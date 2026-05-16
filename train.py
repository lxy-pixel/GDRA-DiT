import os
import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader, Subset
from diffusion_utils_transformer import build_model, lambda_from_u, alpha_from_lambda, sigma_from_lambda, lambda_encoding

# --- 🚀 路径配置 (使用全雪季数据) ---
ROOT_DIR = "/share/liuxiaoyu/diffusion-code-single-sd/transformer/r1/add-data"
NC_PATH   = os.path.join(ROOT_DIR, "data_reshaped", "24-11-25-4-sd-m.nc") # 全雪季雪深
TP_PATH   = os.path.join(ROOT_DIR, "data_reshaped", "tp_norm.nc")         # 全雪季降水
T2M_PATH  = os.path.join(ROOT_DIR, "data_reshaped", "t2m_norm.nc")        # 全雪季温度
DEM_PATH  = "/share/liuxiaoyu/diffusion-code-single-sd/data/land-dayu0.npy"
STAT_PATH = "/share/liuxiaoyu/diffusion-code-single-sd/cache/sd_pixel_stats_with_land_mask.npz" 
SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "DiT_sd_v2_4channel.pt")    # 存为新权重名字

BATCH_SIZE = 16  # 如果显存 OOM，请降到 32 或 16
MAX_EPOCHS = 100
LR = 1e-4

def load_stats():
    d = np.load(STAT_PATH)
    return d["mean"][None, :, :], d["std"][None, :, :], d["land_mask"][None, :, :]

class MultiChannelSnowDataset(Dataset):
    def __init__(self):
        mean_map, std_map, _ = load_stats()
        
        # 1. 加载雪深
        print("加载雪深数据...")
        with xr.open_dataset(NC_PATH) as ds:
            raw_sd = np.nan_to_num(ds['sd'].values.astype(np.float32), nan=0.0)
        self.sd_data = (raw_sd - mean_map) / (std_map + 1e-6) # 形状 (T, 120, 120)

        # 2. 加载气象强迫 (降水和温度)
        print("加载降水与温度数据...")
        with xr.open_dataset(TP_PATH) as ds_tp, xr.open_dataset(T2M_PATH) as ds_t2m:
            self.tp_data = ds_tp['tp_norm'].values.astype(np.float32)
            self.t2m_data = ds_t2m['t2m_norm'].values.astype(np.float32)

        # 3. 加载地形
        dem = np.load(DEM_PATH).astype(np.float32)
        if dem.ndim == 3: dem = dem[0]
        dem_norm = 2.0 * (np.clip(dem, 0, 2420) / 2420.0) - 1.0
        self.dem = torch.from_numpy(dem_norm).unsqueeze(0) # (1, 120, 120)
        
        print(f"数据加载完毕！总时间步: {len(self.sd_data)}")

    def __len__(self): 
        # 因为需要 t-1，所以长度是 T-1
        return len(self.sd_data) - 1
    
    def __getitem__(self, idx):
        t = idx + 1 # 当前目标时刻
        
        # 目标值: x_t (1, 120, 120)
        curr_sd = torch.from_numpy(self.sd_data[t]).unsqueeze(0)
        
        # 条件 1: x_t-1 (1, 120, 120)
        prev_sd = torch.from_numpy(self.sd_data[idx]).unsqueeze(0)
        
        # 条件 3 & 4: tp_t 和 t2m_t (1, 120, 120)
        curr_tp = torch.from_numpy(self.tp_data[t]).unsqueeze(0)
        curr_t2m = torch.from_numpy(self.t2m_data[t]).unsqueeze(0)
        
        return curr_sd, prev_sd, self.dem, curr_tp, curr_t2m

def train():
    device = torch.device("cuda:0")
    model = build_model(device)
    
    # 启用多卡
    if torch.cuda.device_count() > 1:
        print(f"🚀 发现 {torch.cuda.device_count()} 张可用显卡，启用多卡并行训练！")
        model = torch.nn.DataParallel(model)
        
    # 稍微降低一点学习率，求稳
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    
    # ❌ 删除了 scaler (不再使用混合精度)
    
    ds = MultiChannelSnowDataset()
    train_len = int(len(ds) * 0.9)
    train_loader = DataLoader(Subset(ds, range(train_len)), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(f"🚀 开始 4 通道 DiT 训练 (纯 FP32 稳定模式)...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for curr_sd, prev_sd, dem, tp, t2m in train_loader:
            curr_sd, prev_sd, dem = curr_sd.to(device), prev_sd.to(device), dem.to(device)
            tp, t2m = tp.to(device), t2m.to(device)
            B = curr_sd.size(0)
            
            u = torch.rand(B, device=device)
            lmbd = lambda_from_u(u)
            a, s = alpha_from_lambda(lmbd).view(B, 1, 1, 1), sigma_from_lambda(lmbd).view(B, 1, 1, 1)
            
            eps = torch.randn_like(curr_sd)
            noised_x = a * curr_sd + s * eps

            # ❌ 删除了 autocast 混合精度上下文
            model_in = torch.cat([noised_x, prev_sd, dem, tp, t2m], dim=1)
            
            # 🛡️ 绝对安全锁：清除任何潜在的 NaN 或 Inf
            model_in = torch.nan_to_num(model_in, nan=0.0, posinf=10.0, neginf=-10.0)
            
            pred = model(model_in, lambda_encoding(lmbd))
            loss = torch.mean((pred - eps)**2)
            
            # 🛡️ 异常拦截：如果 loss 还是 nan，跳过这个 batch 保护模型
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ 警告: 发现 NaN Loss，已跳过该 Batch！")
                optim.zero_grad()
                continue

            optim.zero_grad()
            # 纯 FP32 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            total_loss += loss.item()
            valid_batches += 1

        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('nan')
        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.6f}")
        
        if epoch % 5 == 0:
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            # 处理 DataParallel 的 state_dict 保存问题
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, SAVE_PATH)
            print(f"  Checkpoint saved: {SAVE_PATH}")

if __name__ == "__main__":
    train()