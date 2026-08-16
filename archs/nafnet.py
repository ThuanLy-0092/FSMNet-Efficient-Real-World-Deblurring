import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .nafnet_utils.local_arch import Local_Base
    from .nafnet_utils.arch_util import LayerNorm2d
except:
    pass # Bỏ qua lỗi nếu chạy file này độc lập để test FLOPs

# =====================================================================
# --- UPGRADE 1: FREQUENCY ATTENTION (LỌC TẦN SỐ BẰNG SỐ PHỨC) ---
# =====================================================================
class FrequencyAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        # Khởi tạo trọng số SỐ PHỨC: kích thước [1, c, 1, 1, 2] 
        # (Số 2 ở cuối đại diện cho Phần Thực và Phần Ảo của sóng)
        self.complex_weight = nn.Parameter(torch.randn(1, c, 1, 1, 2, dtype=torch.float32) * 0.02)
        
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        B, C, H, W = x2.shape
        
        # Biến đổi FFT ra tensor số phức
        x2_fft = torch.fft.rfft2(x2.float(), norm='backward')
        
        # Ép kiểu trọng số thành dạng số phức để nhân với sóng hình ảnh
        weight = torch.view_as_complex(self.complex_weight)
        
        # Nhấn chìm nhiễu, kích thích cạnh nét (Lọc cả Biên độ và Góc pha)
        x2_out = x2_fft * weight
        
        # Biến đổi ngược lại miền không gian pixel
        x2_out = torch.fft.irfft2(x2_out, s=(H, W), norm='backward').type_as(x2)
        
        return (x1 * x2_out).contiguous()

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return (x1 * x2).contiguous()

class LayerNormFunction(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.LayerNorm(c)
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

# =====================================================================
# --- KHỐI FSM-BLOCK (Sử dụng cho toàn bộ Encoder / Decoder) ---
# =====================================================================
class FSMBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # Large Kernel 7x7 giúp bao trọn vệt mờ dài
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=7, padding=3, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # Sử dụng Frequency Attention chuẩn SOTA
        self.fg = FrequencyAttention(dw_channel // 2)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.sg_ffn = SimpleGate()
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNormFunction(c)
        self.norm2 = LayerNormFunction(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp) 
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.fg(x)      
        x = x * self.sca(x) 
        x = self.conv3(x) 
        x = self.dropout1(x)
        y = inp + x * self.beta 

        x = self.conv4(self.norm2(y)) 
        x = self.sg_ffn(x)  
        x = self.conv5(x) 
        x = self.dropout2(x)
        x = y + x * self.gamma
        return x 

# =====================================================================
# --- UPGRADE 2: TRANSPOSED ATTENTION (TĂNG LÊN 4 HEADS) ---
# =====================================================================
class TransposedAttention(nn.Module):
    def __init__(self, dim, num_heads=4): # Tăng lên 4 heads để soi đa góc độ
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(B, C, H, W)

        return self.project_out(out).contiguous()

# =====================================================================
# --- UPGRADE 3: CROSS-GATED VISION E-BRANCHFORMER (XỬ LÝ SONG SONG) ---
# =====================================================================
class VisionEBranchformerBlock(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        self.norm = LayerNormFunction(c)
        
        # Nhánh 1: LOCAL CNN (Bắt chi tiết)
        hidden_dim = int(c * expand)
        self.branch_cnn = nn.Sequential(
            nn.Conv2d(c, hidden_dim, kernel_size=1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, c, kernel_size=1)
        )
        
        # Nhánh 2: GLOBAL ATTENTION (Bắt ngữ cảnh)
        self.branch_attn = TransposedAttention(dim=c, num_heads=4)
        
        self.merge_conv = nn.Conv2d(c * 2, c * 2, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones((1, c, 1, 1)) * 1e-4, requires_grad=True)

    def forward(self, inp):
        x = self.norm(inp)
        
        # Trích xuất 2 nhánh hoàn toàn song song
        feat_local = self.branch_cnn(x)     
        feat_global = self.branch_attn(x)   
        
        # CROSS-GATING: Hai nhánh đánh giá và sửa lỗi chéo cho nhau
        gate_local = torch.sigmoid(feat_local)
        gate_global = torch.sigmoid(feat_global)
        
        feat_local_enhanced = feat_local * gate_global
        feat_global_enhanced = feat_global * gate_local
        
        # Gộp thông tin đã được làm sạch
        merged = torch.cat([feat_local_enhanced, feat_global_enhanced], dim=1)
        merged = self.merge_conv(merged)
        
        # Cổng lọc cuối cùng (SimpleGate module)
        m1, m2 = merged.chunk(2, dim=1)
        out = m1 * m2
        
        return inp + out * self.layer_scale

# =====================================================================
# --- CẤU TRÚC CHÍNH: NAFNET (LẮP GHÉP CÁC KHỐI SOTA) ---
# =====================================================================
class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=2, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[FSMBlock(chan) for _ in range(num)])
            )
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        # ĐÁY U-NET: CHỨA CÁC KHỐI VISION E-BRANCHFORMER 
        self.middle_blks = nn.Sequential(
            *[VisionEBranchformerBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[FSMBlock(chan) for _ in range(num)])
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return {'output': x[:, :, :H, :W]}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x
    
class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__=='__main__':
    from ptflops import get_model_complexity_info

    # UPGRADE 4: Tăng middle_blk_num lên 2 để có Bottleneck sâu hơn
    net = NAFNet(img_channel=3, width=16,
                 middle_blk_num=2, enc_blk_nums=[1,1,1,28], dec_blk_nums=[1,1,1,1])

    macs, params = get_model_complexity_info(net, input_res=(3, 1200, 1920), print_per_layer_stat=False, verbose=False)

    print(f"FSM-Net V2 (SOTA) Complexity -> MACs: {macs}, Params: {params}")