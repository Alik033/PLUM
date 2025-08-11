import argparse
import os

import numpy as np

#import cv2
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.fft as fft

from pdb import set_trace as stx
import numbers

from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################################################################################################################
################################################---------------------Domanin Translation----------------------------####################################################################
########################################################################################################################################################################################
######----------------------image to low-contrast----------------------------------------------------
def i2lc(inp):

    batch_tensor = inp

    # Define the contrast reduction factor (less than 1 reduces contrast)
    contrast_factor = 0.5  # Adjust this value as needed

    # Compute the mean intensity for each image in the batch
    # Mean is computed per image, across height and width, for each channel
    mean_intensity = batch_tensor.mean(dim=(2, 3), keepdim=True)  # Shape: (N, C, 1, 1)

    # Apply contrast reduction to each image in the batch
    low_contrast_batch = mean_intensity + contrast_factor * (batch_tensor - mean_intensity)

    # Clamp values to ensure they remain in the valid range [0, 1]
    low_contrast_batch = low_contrast_batch.clamp(0, 1)

    return low_contrast_batch

#####----------------------image to hazy---------------------------------------------------------------
def i2hz(inp):

    # Example batch of image tensors (N x C x H x W), normalized to [0, 1]
    batch_tensor = inp

    # Define atmospheric light (A) and transmission factor (t)
    # Atmospheric light (A): Typically a bright value, close to white
    A = torch.tensor([0.8, 0.8, 0.8]).view(1, 3, 1, 1)  # Shape: (1, C, 1, 1)
    A = A.expand_as(batch_tensor)  # Broadcast to match batch shape

    A = A.to(device)

    # Transmission factor (t): Simulate haze thickness (uniform or random)
    t = torch.rand(batch_tensor.shape[0], 1, 1, 1) * 0.5 + 0.5  # Uniform t in [0.5, 1.0]
    t = t.expand_as(batch_tensor)  # Broadcast to match batch shape
    t = t.to(device)

    # Create hazy images using the formula: I_hazy = t * I + (1 - t) * A
    hazy_batch = t * batch_tensor + (1 - t) * A

    # Clamp values to ensure they remain in the range [0, 1]
    hazy_batch = hazy_batch.clamp(0, 1)

    return hazy_batch

######-------------------------image to blurry------------------------------------------------------------
# Function to create a Gaussian kernel
def gaussian_kernel(size, sigma):
    # Create a 2D Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * 
                     np.exp(- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    kernel /= kernel.sum()  # Normalize the kernel
    return torch.tensor(kernel, dtype=torch.float32)

# Function to apply Gaussian blur to a single image
def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)  # Expand to apply to 3 channels
    kernel = kernel.to(device)

    # Apply convolution using F.conv2d (requires input shape: [batch_size, in_channels, height, width])
    image = image.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)
    blurred_image = F.conv2d(image, kernel, padding=kernel_size//2, groups=3)  # Apply to 3 channels independently

    return blurred_image.squeeze(0)  # Remove batch dimension


def i2bl(inp):
    # Example: Batch of image tensors (N x C x H x W), normalized to [0, 1]
    batch_tensor = inp  # 8 images, 3 channels, 256x256

    # Apply Gaussian blur to each image in the batch
    blurred_batch = torch.stack([apply_gaussian_blur(img, kernel_size=5, sigma=1.0) for img in batch_tensor])

    # Ensure that the blurred images are in the range [0, 1]
    blurred_batch = blurred_batch.clamp(0, 1)

    return blurred_batch

########---------------------------image to color-distortion--------------------------------------------------------------------
# Function to make an image bluish
def make_bluish(image):
    # Increase the blue channel (3rd channel) by scaling
    image[:, 2, :, :] = image[:, 2, :, :] * 1.5  # Multiply blue channel
    image = image.clamp(0, 1)  # Clamp values to [0, 1] range
    return image

# Function to make an image greenish
def make_greenish(image):
    # Increase the green channel (2nd channel) by scaling
    image[:, 1, :, :] = image[:, 1, :, :] * 1.5  # Multiply green channel
    image = image.clamp(0, 1)  # Clamp values to [0, 1] range
    return image

def i2gb(inp):
    batch_tensor = inp  # 8 images, 3 channels, 256x256
    batch_size,_,_,_ = inp.shape

    # Split the batch into two halves: one for bluish, one for greenish
    half_batch_size = batch_size // 2
    bluish_batch = batch_tensor[:half_batch_size]
    greenish_batch = batch_tensor[half_batch_size:]

    # Apply the color modifications
    bluish_batch = make_bluish(bluish_batch)
    greenish_batch = make_greenish(greenish_batch)

    # Recombine both halves back into a single batch
    modified_batch = torch.cat((bluish_batch, greenish_batch), dim=0)

    return modified_batch
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################


## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(p=0.1)
        


    def forward(self, x, y):
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(y))
        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # print(out.shape)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.dropout(out)
        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.q1X1_1 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)

        self.c_attn = Cross_Attention(dim, num_heads, bias)

    def forward(self, x):
        f_x = x
        x = x + self.attn(self.norm1(x))
        # x = x + self.ffn(self.norm2(x))
        
        x_fft = fft.fftn(f_x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        qf = fft.ifftn(x_fft3,dim=(-2, -1)).real

        ca = self.c_attn(self.norm3(x), self.norm3(qf))

        x = f_x + self.ffn(self.norm2(ca))
        # x = ca + f_x

        return x
########################################################################
class P_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, p_size):
        super(P_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.lin_l1 = torch.nn.Linear(512, dim)
        self.prompt1 = nn.Parameter(torch.rand(1, dim, p_size, p_size))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

#################################################################################
# class C_Attention(nn.Module):
#     def __init__(self, dim, num_heads, kernel, pad, bias):
#         super(C_Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
#         self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=kernel, stride=1, padding=pad, groups=dim*2, bias=bias)
#         self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=kernel, stride=1, padding=pad, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
#     def forward(self, x, y):
#         b,c,h,w = x.shape

#         q = self.v_dwconv(self.v(x))

#         kv = self.qk_dwconv(self.qk(y))
#         k,v = kv.chunk(2, dim=1)  

        
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)
        
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out

#################################################################################
class D_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, p_size = 32):
        super(D_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.prompt1 = nn.Parameter(torch.rand(1, dim, p_size, p_size))
        self.prompt2 = nn.Parameter(torch.rand(1, dim, p_size, p_size))

        self.q1X1_1 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(dim, dim , kernel_size=1, bias=False)

        self.c_attn = Cross_Attention(dim, num_heads, bias)
        self.norm3 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):

        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)


        q = q * self.prompt1
        k = k * self.prompt1
        v = v * self.prompt1
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        sb_o = x + out

        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft2 = x_fft2 * self.prompt2
        x_fft3 = self.q1X1_2(x_fft2)
        fb_o = fft.ifftn(x_fft3,dim=(-2, -1)).real

        x = self.c_attn(self.norm3(sb_o),self.norm3(fb_o)) + x

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k,s,p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class En_Net(nn.Module):

    def __init__(self):
        super(En_Net, self).__init__()   

        self.layer1 = Conv2D_pxp(3, 16, 3, 1, 1)
        #--------------Encoder-------------------------------
        self.T1 = TransformerBlock(dim=16, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias')
        self.d1 = Downsample(16)
        self.T2 = TransformerBlock(dim=16*2, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias')
        self.d2 = Downsample(16*2)
        self.T3 = TransformerBlock(dim=16*4, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias')
        self.d3 = Downsample(16*4)
        self.T4 = TransformerBlock(dim=16*8, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias')

        #-------------Decoder--------------------------------
        self.up1 = Upsample(16*8)
        self.rc1 = nn.Conv2d(16*8, 16*4, kernel_size=1, bias=False)
        self.T5 = D_TransformerBlock(dim=16*4, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias', p_size = 64)
        self.up2 = Upsample(16*4)
        self.rc2 = nn.Conv2d(16*4, 16*2, kernel_size=1, bias=False)
        self.T6 = D_TransformerBlock(dim=16*2, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias', p_size = 128)
        self.up3 = Upsample(16*2)
        self.rc3 = nn.Conv2d(16*2, 16, kernel_size=1, bias=False)
        self.T7 = D_TransformerBlock(dim=16, num_heads=8, ffn_expansion_factor= 2.66, bias=False, LayerNorm_type='WithBias', p_size= 256)

        #-------------Image reconstruction-------------------
        # self.layer_f = Conv2D_pxp(16, 16, 3, 1, 1)
        self.layerf = Conv2D_pxp(16, 3, 3, 1, 1)


    def forward(self, input_x):
        lc = i2lc(input_x)
        hz = i2hz(input_x)
        bl = i2bl(input_x)
        gb = i2gb(input_x)

        x = self.layer1(lc)
        y = self.layer1(hz)
        z = self.layer1(bl)
        a = self.layer1(gb)
        b = self.layer1(input_x)

        #------------Encoder Layer 1--------------------------
        x_t1 = self.T1(x)
        x_chunks1 = torch.chunk(x_t1, chunks=8, dim=1)
        y_t1 = self.T1(y)
        y_chunks1 = torch.chunk(y_t1, chunks=8, dim=1)
        x_t1 = torch.cat((y_chunks1[0], torch.cat((x_chunks1[1:]),dim=1)),dim=1)
        y_t1 = torch.cat((x_chunks1[0], torch.cat((y_chunks1[1:]),dim=1)),dim=1)
        z_t1 = self.T1(z)
        z_chunks1 = torch.chunk(z_t1, chunks=8, dim=1)
        a_t1 = self.T1(a)
        a_chunks1 = torch.chunk(a_t1, chunks=8, dim=1)
        z_t1 = torch.cat((a_chunks1[0], torch.cat((z_chunks1[1:]),dim=1)),dim=1)
        a_t1 = torch.cat((z_chunks1[0], torch.cat((a_chunks1[1:]),dim=1)),dim=1)
        b_t1 = self.T1(b)
        add_fea1 = x_t1 + y_t1 + z_t1 + a_t1 + b_t1
        x_t1 = self.d1(x_t1)
        y_t1 = self.d1(y_t1)
        z_t1 = self.d1(z_t1)
        a_t1 = self.d1(a_t1)
        b_t1 = self.d1(b_t1)

        #------------Encoder Layer 2--------------------------
        x_t1 = self.T2(x_t1)
        y_t1 = self.T2(y_t1)
        y_chunks1 = torch.chunk(y_t1, chunks=8, dim=1)
        z_t1 = self.T2(z_t1)
        z_chunks1 = torch.chunk(z_t1, chunks=8, dim=1)
        y_t1 = torch.cat((z_chunks1[0], torch.cat((y_chunks1[1:]),dim=1)),dim=1)
        z_t1 = torch.cat((y_chunks1[0], torch.cat((z_chunks1[1:]),dim=1)),dim=1)
        a_t1 = self.T2(a_t1)
        a_chunks1 = torch.chunk(a_t1, chunks=8, dim=1)
        b_t1 = self.T2(b_t1)
        b_chunks1 = torch.chunk(b_t1, chunks=8, dim=1)
        a_t1 = torch.cat((b_chunks1[0], torch.cat((a_chunks1[1:]),dim=1)),dim=1)
        b_t1 = torch.cat((a_chunks1[0], torch.cat((b_chunks1[1:]),dim=1)),dim=1)
        add_fea2 = x_t1 + y_t1 + z_t1 + a_t1 + b_t1
        x_t1 = self.d2(x_t1)
        y_t1 = self.d2(y_t1)
        z_t1 = self.d2(z_t1)
        a_t1 = self.d2(a_t1)
        b_t1 = self.d2(b_t1)

        #------------Encoder Layer 3--------------------------
        x_t1 = self.T3(x_t1)
        x_chunks1 = torch.chunk(x_t1, chunks=8, dim=1)
        y_t1 = self.T3(y_t1)
        y_chunks1 = torch.chunk(y_t1, chunks=8, dim=1)
        x_t1 = torch.cat((y_chunks1[0], torch.cat((x_chunks1[1:]),dim=1)),dim=1)
        y_t1 = torch.cat((x_chunks1[0], torch.cat((y_chunks1[1:]),dim=1)),dim=1)
        z_t1 = self.T3(z_t1)
        z_chunks1 = torch.chunk(z_t1, chunks=8, dim=1)
        a_t1 = self.T3(a_t1)
        a_chunks1 = torch.chunk(a_t1, chunks=8, dim=1)
        z_t1 = torch.cat((a_chunks1[0], torch.cat((z_chunks1[1:]),dim=1)),dim=1)
        a_t1 = torch.cat((z_chunks1[0], torch.cat((a_chunks1[1:]),dim=1)),dim=1)
        add_fea3 = x_t1 + y_t1 + z_t1 + a_t1 + b_t1
        b_t1 = self.T3(b_t1)
        x_t1 = self.d3(x_t1)
        y_t1 = self.d3(y_t1)
        z_t1 = self.d3(z_t1)
        a_t1 = self.d3(a_t1)
        b_t1 = self.d3(b_t1)

        ########################################################################################## 
        add_fea = x_t1 + y_t1 + z_t1 + a_t1 + b_t1
        ##########################################################################################

        ##########################################################################################
        add_fea = self.T4(add_fea)
        # add_fea = add_fea + l4_prompt
        
        ##########################################################################################
        #-----------Decoder 1-------------------------------------
        dec_inp0 = self.up1(add_fea)
        dec_inp0 = dec_inp0 + add_fea3
        dec_inp0 = self.T5(dec_inp0) 
        
        #-----------Decoder 2-------------------------------------
        dec_inp0 = self.up2(dec_inp0)
        dec_inp0 = dec_inp0 + add_fea2
        dec_inp0 = self.T6(dec_inp0) 

        #-----------Decoder 3-------------------------------------
        dec_inp0 = self.up3(dec_inp0)
        dec_inp0 = dec_inp0 + add_fea1
        dec_inp0 = self.T7(dec_inp0)

        ###############################################################################################
        #-----------Image Reconstruction---------------------------
        x_f = self.layerf(dec_inp0) + input_x
        # print("Hi")

        return x_f