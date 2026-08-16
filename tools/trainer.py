import sys

sys.path.append('../losses')
sys.path.append('../data/datasets/datapipeline')
from losses import SSIM, calculate_loss

calc_SSIM = SSIM(data_range=1.)

from tqdm import tqdm # Nhớ import thêm tqdm ở đầu file

import torch
from torch.amp import autocast, GradScaler # AMP thế hệ mới
from tqdm import tqdm

def train_model(model, optim, all_losses, train_loader, metrics, local_rank=None):
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {metrics.get('epoch', 0)}", leave=False)
    
    # 1. Xác định thiết bị và khởi tạo Scaler (Dù BF16 ít bị tràn số hơn FP16, 
    # nhưng dùng Scaler vẫn là thói quen tốt để đảm bảo ổn định tuyệt đối)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Lưu ý: Với BF16 thuần túy, đôi khi không cần Scaler, nhưng để code linh hoạt, ta cứ khai báo.
    scaler = GradScaler(enabled=(device_type == 'cuda'))

    for high_batch, low_batch in train_loader:
        high_batch = high_batch.to(local_rank)
        low_batch = low_batch.to(local_rank)

        optim.zero_grad()

        # 2. Bật Autocast với kiểu dữ liệu bfloat16
        # Đây là nơi phép màu xảy ra: Forward pass sẽ chạy ở 16-bit
        with autocast(device_type=device_type, dtype=torch.bfloat16):
            output_dict = model(low_batch)
            optim_loss, _ = calculate_loss(all_losses, output_dict, high_batch, get_individual_losses=True)

        # 3. Backward pass và Step với Scaler
        # Nó sẽ tự động quản lý việc ép kiểu ngược lại để cập nhật trọng số
        scaler.scale(optim_loss).backward()
        scaler.step(optim)
        scaler.update()

        pbar.set_postfix({'loss': f"{optim_loss.item():.4f}"})
        pbar.update(1)

    pbar.close()
    return model, optim, metrics