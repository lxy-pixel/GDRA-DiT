@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")
    ENSEMBLE_SIZE = 10
    N_HOURS = 75
    START_TIME = "2025-01-28T21:00:00"
    ETA = 1.0
    NOISE_SCALE = 1.1

    # 加载像素统计、地形DEM
    d = np.load(STAT_PATH)
    mean_t, std_t, land_mask_t = to_1x1_hw(d["mean"], device), to_1x1_hw(d["std"], device), to_1x1_hw(d["land_mask"], device)
    dem_raw = np.load(DEM_PATH).astype(np.float32)
    dem_raw = dem_raw[0] if dem_raw.ndim == 3 else dem_raw
    dem_norm = 2.0 * (np.clip(dem_raw, 0.0, 2420.0) / 2420.0) - 1.0
    dem_t = to_1x1_hw(dem_norm, device).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()
    H, W = dem_raw.shape

    # 加载DiT模型
    model = build_model(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 读取参考积雪、tp、t2m强迫场
    ds_ref = xr.open_dataset(REF_NC)
    ds_tp = xr.open_dataset(TP_PATH)
    ds_t2m = xr.open_dataset(T2M_PATH)
    lats, lons = ds_ref["latitude"].values, ds_ref["longitude"].values
    start_t = pd.to_datetime(START_TIME)
    init_prev_time = start_t - pd.Timedelta(hours=1)

    # 初始化上一时刻积雪场
    x_prev_raw = np.nan_to_num(np.squeeze(ds_ref[VAR_NAME].sel(time=init_prev_time).values), nan=0.0)
    x_prev_t = torch.from_numpy(x_prev_raw).float().to(device)[None, None, :, :]
    current_x_prev_batch = ((x_prev_t - mean_t) / (std_t + EPS)).expand(ENSEMBLE_SIZE, -1, -1, -1).contiguous()

    # 预生成扩散时序序列 u->lambda->alpha->sigma
    u_seq = torch.arange(1.0, -1e-9, -1e-3, device=device)
    l_seq, a_seq, s_seq = lambda_from_u(u_seq), alpha_from_lambda(lambda_from_u(u_seq)), sigma_from_lambda(lambda_from_u(u_seq))
    cur_t = start_t

    # 逐小时滚动预报循环
    for hour in range(N_HOURS):
        # 读取当前时刻tp、t2m强迫
        tp_raw = ds_tp['tp_norm'].sel(time=cur_t).values
        t2m_raw = ds_t2m['t2m_norm'].sel(time=cur_t).values
        tp_t = torch.from_numpy(tp_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)
        t2m_t = torch.from_numpy(t2m_raw).float().to(device).view(1, 1, H, W).expand(ENSEMBLE_SIZE, -1, -1, -1)

        # 初始化扩散噪声
        x = torch.randn(ENSEMBLE_SIZE, 1, H, W, device=device) * NOISE_SCALE

        # EDF扩散采样迭代
        for i in range(len(u_seq) - 1):
            at, st = a_seq[i].view(-1, 1, 1, 1), s_seq[i].view(-1, 1, 1, 1)
            at_next, st_next = a_seq[i+1].view(-1, 1, 1, 1), s_seq[i+1].view(-1, 1, 1, 1)
            emb = lambda_encoding(l_seq[i].expand(ENSEMBLE_SIZE))
            # 5通道模型输入拼接
            model_in = torch.cat([x, current_x_prev_batch, dem_t, tp_t, t2m_t], dim=1)
            model_in = torch.nan_to_num(model_in, nan=0.0)
            eps_pred = model(model_in, emb)
            # 预测干净场
            x0_pred = torch.clamp((x - st * eps_pred) / (at + EPS), -10.0, 10.0)
            # EDF更新公式
            sigma = ETA * torch.sqrt((st_next**2 - (at_next / (at + EPS))**2 * st**2).clamp(min=0))
            c2 = torch.sqrt((st_next**2 - sigma**2).clamp(min=0))
            x = at_next * x0_pred + c2 * eps_pred + sigma * (torch.randn_like(x) * NOISE_SCALE)

        # 反归一化+物理截断+陆地掩膜
        ens_post = torch.clamp(x * (std_t + EPS) + mean_t, 0.0, MAX_PHYSICAL_SD)
        ens_masked = (ens_post * land_mask_t.expand_as(ens_post)).squeeze(1).cpu().numpy().astype(np.float32)
        mean_np, std_np = np.mean(ens_masked, axis=0), np.std(ens_masked, axis=0)

        # 输出单时次nc
        ds_out = xr.Dataset(
            coords={"member": np.arange(ENSEMBLE_SIZE), "latitude": lats, "longitude": lons, "time": cur_t},
            data_vars={
                "ens": (("member", "latitude", "longitude"), ens_masked),
                "mean": (("latitude", "longitude"), mean_np),
                "std": (("latitude", "longitude"), std_np)
            }
        )
        ds_out.to_netcdf(os.path.join(OUT_DIR, f"forecast_output_{cur_t.strftime('%Y%m%d_%H%M')}.nc"))
        # 滚动更新下一时刻初值
        current_x_prev_batch = ((ens_post - mean_t) / (std_t + EPS)).contiguous()
        cur_t += pd.Timedelta(hours=1)
