def _denoise_step(x, idx, a_seq, s_seq, l_seq, model, x_prev, dem, tp, t2m, ETA, NOISE_SCALE):
    at, st = a_seq[idx].view(-1,1,1,1), s_seq[idx].view(-1,1,1,1)
    at_next, st_next = a_seq[idx+1].view(-1,1,1,1), s_seq[idx+1].view(-1,1,1,1)
    emb = lambda_encoding(l_seq[idx].expand(x.shape[0]))
    # 5通道输入：噪声x、上一时刻积雪、地形、降水、温度
    model_in = torch.cat([x, x_prev, dem, tp, t2m], dim=1)
    model_in = torch.nan_to_num(model_in, nan=0.0)
    eps_pred = model(model_in, emb)
    # EDF去噪公式
    x0_pred = torch.clamp((x - st * eps_pred) / (at + EPS), -10.0, 10.0)
    sigma = ETA * torch.sqrt((st_next**2 - (at_next/(at+EPS))**2 * st**2).clamp(min=0))
    c2 = torch.sqrt((st_next**2 - sigma**2).clamp(min=0))
    return at_next * x0_pred + c2 * eps_pred + sigma * torch.randn_like(x) * NOISE_SCALE

def _forward_step(x, idx, tau, a_seq, s_seq, NOISE_SCALE):
    target_idx = max(0, idx - tau)
    ratio = a_seq[target_idx] / (a_seq[idx] + EPS)
    sigma_fwd = torch.sqrt((s_seq[target_idx]**2) - (ratio**2) * (s_seq[idx]**2))
    # 前向扩散加噪（重绘回退用）
    return ratio * x + sigma_fwd.clamp(min=0) * torch.randn_like(x) * NOISE_SCALE
@torch.no_grad()
def main():
    device = torch.device("cuda:0")
    ENSEMBLE_SIZE = 10; N_HOURS = 75; START_TIME = "2025-01-28T21:00:00"
    ETA = 1.0; NOISE_SCALE = 1.1
    REWIND_TAU = 10; REPAINT_ROUNDS = 3; OMEGA_STEPS = set(range(800, -1, -10))
    VALIDATION_FRACTION = 0.2

    # 1. 加载统计量、地形、DiT模型
    d = np.load(STAT_PATH)
    mean_t, std_t, land_mask_t = to_1x1_hw(d["mean"], device), to_1x1_hw(d["std"], device), to_1x1_hw(d["land_mask"], device)
    dem_raw = np.load(DEM_PATH).astype(np.float32)
    dem_norm = 2.0 * (np.clip(dem_raw, 0.0, 2420.0) / 2420.0) - 1.0
    dem_t = to_1x1_hw(dem_norm, device).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()
    H, W = dem_raw.shape
    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 打开数据文件、初始化滚动初值
    ds_ref, ds_tp, ds_t2m, ds_obs = xr.open_dataset(REF_NC), xr.open_dataset(TP_PATH), xr.open_dataset(T2M_PATH), xr.open_dataset(OBS_PATH)
    lats, lons = ds_ref["latitude"].values, ds_ref["longitude"].values
    start_t = pd.to_datetime(START_TIME)
    init_prev_time = start_t - pd.Timedelta(hours=1)
    x_prev_raw = np.nan_to_num(np.squeeze(ds_ref[VAR_NAME].sel(time=init_prev_time).values), nan=0.0)
    current_x_prev_batch = ((torch.from_numpy(x_prev_raw).float().to(device)[None, None, :, :] - mean_t) / (std_t + EPS)).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()

    # 预计算扩散序列
    u_seq = torch.arange(1.0, -1e-9, -1e-3, device=device)
    l_seq, a_seq, s_seq = lambda_from_u(u_seq), alpha_from_lambda(lambda_from_u(u_seq)), sigma_from_lambda(lambda_from_u(u_seq))
    cur_t = start_t

    # 逐小时同化主循环
    for hour in range(N_HOURS):
        # 读取当前时刻强迫场tp/t2m
        tp_raw = ds_tp['tp_norm'].sel(time=cur_t).values
        t2m_raw = ds_t2m['t2m_norm'].sel(time=cur_t).values
        tp_t = torch.from_numpy(tp_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)
        t2m_t = torch.from_numpy(t2m_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)

        # 读取站点观测+分层抽样划分同化集/验证集
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
        if n > 0:
            # 20×20网格分层抽样拆分同化/验证观测
            strata = {}
            for i, (y, x) in enumerate(all_coords):
                grid_id = (y // 20, x // 20)
                if grid_id not in strata: strata[grid_id] = []
                strata[grid_id].append(i)
            val_indices = set()
            for grid_id, idxs in strata.items():
                np.random.shuffle(idxs)
                num_v = max(1, int(len(idxs) * VALIDATION_FRACTION)) if len(idxs) >= 2 else 0
                val_indices.update(idxs[:num_v])
            # 填充同化观测场、验证观测场
            for i, ((y, x), v) in enumerate(zip(all_coords, all_vals)):
                if i in val_indices: v_map[y,x], v_mask[y,x] = v, 1.0
                else: a_map[y,x], a_mask[y,x] = v, 1.0
        # 归一化观测场
        y_a = ((torch.from_numpy(a_map).to(device).view(1,1,H,W) - mean_t)/std_t).expand(ENSEMBLE_SIZE,-1,-1,-1)
        m_a = torch.from_numpy(a_mask).to(device).view(1,1,H,W).expand_as(y_a)

        # 初始化噪声
        x = torch.randn(ENSEMBLE_SIZE, 1, H, W, device=device) * NOISE_SCALE

        # 扩散采样 + CLIN重绘同化
        for i in tqdm(range(len(u_seq)-1), desc="Sampling", leave=False):
            t_idx = len(u_seq)-1-i
            # 基础EDF去噪
            x = _denoise_step(x, i, a_seq, s_seq, l_seq, model, current_x_prev_batch, dem_t, tp_t, t2m_t, ETA, NOISE_SCALE)
            # 关键重绘同化逻辑：指定时间步注入观测
            if n > len(val_indices) and t_idx in OMEGA_STEPS:
                x_obs_noisy = a_seq[i+1]*y_a + s_seq[i+1]*torch.randn_like(x)*NOISE_SCALE
                for _ in range(REPAINT_ROUNDS):
                    x = m_a * x_obs_noisy + (1 - m_a) * x
                    x = _forward_step(x, i, REWIND_TAU, a_seq, s_seq, NOISE_SCALE)
                    for r_idx in range(max(0, i - REWIND_TAU), i):
                        x = _denoise_step(x, r_idx, a_seq, s_seq, l_seq, model, current_x_prev_batch, dem_t, tp_t, t2m_t, ETA, NOISE_SCALE)

        # 后处理：反归一化、物理约束、陆地掩膜
        ens_phys = torch.clamp(torch.nan_to_num(x * (std_t + EPS) + mean_t, nan=0.0), 0.0, MAX_PHYSICAL_SD)
        ens_masked = (ens_phys * land_mask_t).squeeze(1).cpu().numpy().astype(np.float32)
        ana_mean, ana_std = np.nanmean(ens_masked, axis=0), np.nanstd(ens_masked, axis=0)

        # 输出同化nc（集合、均值、离散度、同化/验证观测）
        ds_out = xr.Dataset(
            coords={"latitude": lats, "longitude": lons, "time": cur_t, "member": np.arange(ENSEMBLE_SIZE)},
            data_vars={
                "ens": (("member", "latitude", "longitude"), ens_masked),
                "mean": (("latitude", "longitude"), ana_mean),
                "std": (("latitude", "longitude"), ana_std),
                "assim_obs_val": (("latitude", "longitude"), a_map.astype(np.float32)),
                "validation_obs_mask": (("latitude", "longitude"), v_mask.astype(np.float32)),
                "validation_obs_val": (("latitude", "longitude"), v_map.astype(np.float32))
            }
        )
        ds_out.to_netcdf(os.path.join(OUT_DIR, f"assim_output_{cur_t.strftime('%Y%m%d_%H%M')}.nc"))
        # 滚动更新下一时刻初值
        current_x_prev_batch = ((ens_phys - mean_t) / (std_t + EPS)).contiguous()
        cur_t += pd.Timedelta(hours=1)
        torch.cuda.empty_cache()
