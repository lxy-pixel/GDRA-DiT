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
OBS_PATH   = "/share/liuxiaoyu/diffusion-code-single-sd/data/zdgc-sd.nc"

OUT_DIR    = os.path.join(ROOT_DIR, "outputs", "assim_transformer_v2_4channel")

VAR_NAME = "sd"
MAX_PHYSICAL_SD = 50.0
EPS = 1e-6
VALIDATION_FRACTION = 0.2
OMEGA_STEPS = set(range(800, -1, -10))
REWIND_TAU = 10
REPAINT_ROUNDS = 3

def to_1x1_hw(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(arr).float().to(device)
    if t.ndim == 2: t = t[None, None, :, :]
    elif t.ndim == 3: t = t[None, 0:1, :, :]
    return t

# --- 重构的去噪算子 (加入 tp, t2m) ---
def _denoise_step(x, idx, a_seq, s_seq, l_seq, model, x_prev, dem, tp, t2m, ETA, NOISE_SCALE):
    at, st = a_seq[idx].view(-1,1,1,1), s_seq[idx].view(-1,1,1,1)
    at_next, st_next = a_seq[idx+1].view(-1,1,1,1), s_seq[idx+1].view(-1,1,1,1)
    emb = lambda_encoding(l_seq[idx].expand(x.shape[0]))
    
    # 5 通道拼接
    model_in = torch.cat([x, x_prev, dem, tp, t2m], dim=1)
    model_in = torch.nan_to_num(model_in, nan=0.0)
    eps_pred = model(model_in, emb)
    
    x0_pred = torch.clamp((x - st * eps_pred) / (at + EPS), -10.0, 10.0)
    sigma = ETA * torch.sqrt((st_next**2 - (at_next/(at+EPS))**2 * st**2).clamp(min=0))
    c2 = torch.sqrt((st_next**2 - sigma**2).clamp(min=0))
    return at_next * x0_pred + c2 * eps_pred + sigma * torch.randn_like(x) * NOISE_SCALE

def _forward_step(x, idx, tau, a_seq, s_seq, NOISE_SCALE):
    target_idx = max(0, idx - tau)
    ratio = a_seq[target_idx] / (a_seq[idx] + EPS)
    sigma_fwd = torch.sqrt((s_seq[target_idx]**2) - (ratio**2) * (s_seq[idx]**2))
    return ratio * x + sigma_fwd.clamp(min=0) * torch.randn_like(x) * NOISE_SCALE

@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")
    
    ENSEMBLE_SIZE = 10; N_HOURS = 75; START_TIME = "2025-01-28T21:00:00"
    ETA = 1.0; NOISE_SCALE = 1.1

    print(f"🚀 [DiT 4通道 CLIN 同化] 输出目录: {OUT_DIR}")

    d = np.load(STAT_PATH)
    mean_t, std_t, land_mask_t = to_1x1_hw(d["mean"], device), to_1x1_hw(d["std"], device), to_1x1_hw(d["land_mask"], device)
    
    dem_raw = np.load(DEM_PATH).astype(np.float32)
    dem_raw = dem_raw[0] if dem_raw.ndim == 3 else dem_raw
    dem_norm = 2.0 * (np.clip(dem_raw, 0.0, 2420.0) / 2420.0) - 1.0
    dem_t = to_1x1_hw(dem_norm, device).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()
    H, W = dem_raw.shape

    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    ds_ref, ds_tp, ds_t2m, ds_obs = xr.open_dataset(REF_NC), xr.open_dataset(TP_PATH), xr.open_dataset(T2M_PATH), xr.open_dataset(OBS_PATH)
    lats, lons = ds_ref["latitude"].values, ds_ref["longitude"].values

    start_t = pd.to_datetime(START_TIME)
    init_prev_time = start_t - pd.Timedelta(hours=1)
    x_prev_raw = np.nan_to_num(np.squeeze(ds_ref[VAR_NAME].sel(time=init_prev_time).values), nan=0.0)
    current_x_prev_batch = ((torch.from_numpy(x_prev_raw).float().to(device)[None, None, :, :] - mean_t) / (std_t + EPS)).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()

    u_seq = torch.arange(1.0, -1e-9, -1e-3, device=device)
    l_seq, a_seq, s_seq = lambda_from_u(u_seq), alpha_from_lambda(lambda_from_u(u_seq)), sigma_from_lambda(lambda_from_u(u_seq))

    cur_t = start_t
    for hour in range(N_HOURS):
        print(f"\n--- DiT Assimilation | {cur_t} ({hour+1}/{N_HOURS}) ---")

        # 获取强迫场
        tp_raw = ds_tp['tp_norm'].sel(time=cur_t).values
        t2m_raw = ds_t2m['t2m_norm'].sel(time=cur_t).values
        tp_t = torch.from_numpy(tp_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)
        t2m_t = torch.from_numpy(t2m_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)

        # 🚀 基于地理网格的空间分层抽样 (解决复杂地形过拟合)
        all_coords, all_vals = [], []
        try:
            obs_cur = ds_obs.sel(DateTime=cur_t, method="nearest", tolerance=pd.Timedelta(minutes=30))
            v_arr = np.asarray(obs_cur["Snow_Depth"].values)
            ok = ~np.isnan(v_arr)
            if ok.sum() > 0:
                sds, olats, olons = v_arr[ok], obs_cur["Lat"].values[ok], obs_cur["Lon"].values[ok]
                for lt, ln, val in zip(olats, olons, sds):
                    y, x = int(np.argmin(np.abs(lats - lt))), int(np.argmin(np.abs(lons - ln)))
                    if 0 <= y < H and 0 <= x < W:
                        all_coords.append((y, x)); all_vals.append(float(val))
        except: pass

        n = len(all_coords)
        a_map, a_mask, v_map, v_mask = np.zeros_like(dem_raw), np.zeros_like(dem_raw), np.zeros_like(dem_raw), np.zeros_like(dem_raw)
        n_v = 0

        if n > 0:
            np.random.seed(42)
            strata = {}
            for i, (y, x) in enumerate(all_coords):
                grid_id = (y // 20, x // 20)
                if grid_id not in strata: strata[grid_id] = []
                strata[grid_id].append(i)
            
            val_indices = []
            for grid_id, idxs in strata.items():
                np.random.shuffle(idxs)
                num_v = max(1, int(len(idxs) * VALIDATION_FRACTION)) if len(idxs) >= 2 else 0
                val_indices.extend(idxs[:num_v])
            val_indices = set(val_indices)

            for i, ((y, x), v) in enumerate(zip(all_coords, all_vals)):
                if i in val_indices: v_map[y,x], v_mask[y,x] = v, 1.0; n_v += 1
                else: a_map[y,x], a_mask[y,x] = v, 1.0
        
        y_a = ((torch.from_numpy(a_map).to(device).view(1,1,H,W) - mean_t)/std_t).expand(ENSEMBLE_SIZE,-1,-1,-1)
        m_a = torch.from_numpy(a_mask).to(device).view(1,1,H,W).expand_as(y_a)

        x = torch.randn(ENSEMBLE_SIZE, 1, H, W, device=device) * NOISE_SCALE

        for i in tqdm(range(len(u_seq)-1), desc="Sampling", leave=False):
            t_idx = len(u_seq)-1-i
            # 基础去噪 (带 tp, t2m)
            x = _denoise_step(x, i, a_seq, s_seq, l_seq, model, current_x_prev_batch, dem_t, tp_t, t2m_t, ETA, NOISE_SCALE)

            # 观测注入重绘
            if n > n_v and t_idx in OMEGA_STEPS:
                x_obs_noisy = a_seq[i+1]*y_a + s_seq[i+1]*torch.randn_like(x)*NOISE_SCALE
                for _ in range(REPAINT_ROUNDS):
                    x = m_a * x_obs_noisy + (1 - m_a) * x
                    x = _forward_step(x, i, REWIND_TAU, a_seq, s_seq, NOISE_SCALE)
                    for r_idx in range(max(0, i - REWIND_TAU), i):
                        x = _denoise_step(x, r_idx, a_seq, s_seq, l_seq, model, current_x_prev_batch, dem_t, tp_t, t2m_t, ETA, NOISE_SCALE)

        ens_phys = torch.clamp(torch.nan_to_num(x * (std_t + EPS) + mean_t, nan=0.0), 0.0, MAX_PHYSICAL_SD)
        ens_masked = (ens_phys * land_mask_t).squeeze(1).cpu().numpy().astype(np.float32)
        ana_mean, ana_std = np.nanmean(ens_masked, axis=0), np.nanstd(ens_masked, axis=0)
        
        out_path = os.path.join(OUT_DIR, f"assim_output_{cur_t.strftime('%Y%m%d_%H%M')}.nc")
        ds_out = xr.Dataset(
            coords={"latitude": lats, "longitude": lons, "time": cur_t, "member": np.arange(ENSEMBLE_SIZE)},
            data_vars={
                "ens": (("member", "latitude", "longitude"), ens_masked),
                "mean": (("latitude", "longitude"), ana_mean), "std": (("latitude", "longitude"), ana_std),
                "assim_obs_val": (("latitude", "longitude"), a_map.astype(np.float32)),
                "validation_obs_mask": (("latitude", "longitude"), v_mask.astype(np.float32)),
                "validation_obs_val": (("latitude", "longitude"), v_map.astype(np.float32))
            }
        )
        ds_out.to_netcdf(out_path)
        current_x_prev_batch = ((ens_phys - mean_t) / (std_t + EPS)).contiguous()
        cur_t += pd.Timedelta(hours=1)
        torch.cuda.empty_cache()

    ds_ref.close(); ds_obs.close(); ds_tp.close(); ds_t2m.close()
    print("\n--- DiT CLIN 同化完成! ---")

if __name__ == "__main__":
    main()