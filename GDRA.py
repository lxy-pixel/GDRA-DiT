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
