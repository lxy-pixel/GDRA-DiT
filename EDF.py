#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import skew
from diffusion_utils_transformer import build_model, lambda_from_u, alpha_from_lambda, sigma_from_lambda, lambda_encoding

# ===================== 基础配置 =====================
ROOT_DIR = "/share/liuxiaoyu/diffusion-code-single-sd/transformer/r1/add-data"

MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "DiT_sd_v2_4channel.pt")
REF_NC     = os.path.join(ROOT_DIR, "data_reshaped", "24-11-25-4-sd-m.nc")
TP_PATH    = os.path.join(ROOT_DIR, "data_reshaped", "tp_norm.nc")
T2M_PATH   = os.path.join(ROOT_DIR, "data_reshaped", "t2m_norm.nc")
DEM_PATH   = "/share/liuxiaoyu/diffusion-code-single-sd/data/land-dayu0.npy"
STAT_PATH  = "/share/liuxiaoyu/diffusion-code-single-sd/cache/sd_pixel_stats_with_land_mask.npz"

OUT_DIR    = os.path.join(ROOT_DIR, "outputs", "forecast_transformer_v2_4channel")

VAR_NAME = "sd"
MAX_PHYSICAL_SD = 50.0
EPS = 1e-6

def to_1x1_hw(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(arr).float().to(device)
    if t.ndim == 2: t = t[None, None, :, :]
    elif t.ndim == 3: t = t[None, 0:1, :, :]
    return t

@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0") # 推理阶段通常单卡就够了
    
    ENSEMBLE_SIZE = 10
    N_HOURS = 75
    START_TIME = "2025-01-28T21:00:00"
    ETA = 1.0
    NOISE_SCALE = 1.1

    print(f"🚀 [DiT 4通道 EDF 预报] 输出目录: {OUT_DIR}")

    # 1. 加载地形与统计量
    d = np.load(STAT_PATH)
    mean_t, std_t, land_mask_t = to_1x1_hw(d["mean"], device), to_1x1_hw(d["std"], device), to_1x1_hw(d["land_mask"], device)
    
    dem_raw = np.load(DEM_PATH).astype(np.float32)
    dem_raw = dem_raw[0] if dem_raw.ndim == 3 else dem_raw
    dem_norm = 2.0 * (np.clip(dem_raw, 0.0, 2420.0) / 2420.0) - 1.0
    dem_t = to_1x1_hw(dem_norm, device).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()
    H, W = dem_raw.shape

    # 2. 加载模型
    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 3. 加载所有数据场
    ds_ref = xr.open_dataset(REF_NC)
    ds_tp = xr.open_dataset(TP_PATH)
    ds_t2m = xr.open_dataset(T2M_PATH)
    lats, lons = ds_ref["latitude"].values, ds_ref["longitude"].values

    start_t = pd.to_datetime(START_TIME)
    init_prev_time = start_t - pd.Timedelta(hours=1)
    
    x_prev_raw = np.nan_to_num(np.squeeze(ds_ref[VAR_NAME].sel(time=init_prev_time).values), nan=0.0)
    x_prev_t = torch.from_numpy(x_prev_raw).float().to(device)[None, None, :, :]
    current_x_prev_batch = ((x_prev_t - mean_t) / (std_t + EPS)).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()

    u_seq = torch.arange(1.0, -1e-9, -1e-3, device=device)
    l_seq, a_seq, s_seq = lambda_from_u(u_seq), alpha_from_lambda(lambda_from_u(u_seq)), sigma_from_lambda(lambda_from_u(u_seq))

    cur_t = start_t
    for hour in range(N_HOURS):
        print(f"\n--- DiT EDF Forecast | {cur_t} ({hour+1}/{N_HOURS}) ---")

        # 🚀 获取当前时刻的强迫场 (tp, t2m) 并扩展维度
        tp_raw = ds_tp['tp_norm'].sel(time=cur_t).values
        t2m_raw = ds_t2m['t2m_norm'].sel(time=cur_t).values
        tp_t = torch.from_numpy(tp_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)
        t2m_t = torch.from_numpy(t2m_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)

        x = torch.randn(ENSEMBLE_SIZE, 1, H, W, device=device) * NOISE_SCALE

        for i in tqdm(range(len(u_seq) - 1), desc="Sampling", leave=False):
            at, st = a_seq[i].view(-1, 1, 1, 1), s_seq[i].view(-1, 1, 1, 1)
            at_next, st_next = a_seq[i + 1].view(-1, 1, 1, 1), s_seq[i + 1].view(-1, 1, 1, 1)
            emb = lambda_encoding(l_seq[i].expand(ENSEMBLE_SIZE))

            # 🚀 5 通道拼接
            model_in = torch.cat([x, current_x_prev_batch, dem_t, tp_t, t2m_t], dim=1)
            model_in = torch.nan_to_num(model_in, nan=0.0) # 安全锁
            
            eps_pred = model(model_in, emb)
            x0_pred = torch.clamp((x - st * eps_pred) / (at + EPS), -10.0, 10.0)
            sigma = ETA * torch.sqrt((st_next**2 - (at_next / (at + EPS))**2 * st**2).clamp(min=0))
            c2 = torch.sqrt((st_next**2 - sigma**2).clamp(min=0))
            x = at_next * x0_pred + c2 * eps_pred + sigma * (torch.randn_like(x) * NOISE_SCALE)

        # 后处理与保存
        ens_post = torch.clamp(x * (std_t + EPS) + mean_t, 0.0, MAX_PHYSICAL_SD)
        ens_masked = (ens_post * land_mask_t.expand_as(ens_post)).squeeze(1).cpu().numpy().astype(np.float32)
        mean_np, std_np = np.mean(ens_masked, axis=0), np.std(ens_masked, axis=0)
        print(f"  > Mean Spread: {np.mean(std_np):.4f} m")

        out_path = os.path.join(OUT_DIR, f"forecast_output_{cur_t.strftime('%Y%m%d_%H%M')}.nc")
        ds_out = xr.Dataset(
            coords={"member": np.arange(ENSEMBLE_SIZE, dtype=np.int32), "latitude": lats, "longitude": lons, "time": pd.to_datetime(cur_t)},
            data_vars={"ens": (("member", "latitude", "longitude"), ens_masked), "mean": (("latitude", "longitude"), mean_np), "std": (("latitude", "longitude"), std_np)}
        )
        ds_out.to_netcdf(out_path)
        current_x_prev_batch = ((ens_post - mean_t) / (std_t + EPS)).contiguous()
        cur_t += pd.Timedelta(hours=1)

    ds_ref.close(); ds_tp.close(); ds_t2m.close()
    print("\n--- DiT 4通道 EDF 预报完成! ---")

if __name__ == "__main__":
    main()