## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090


import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
import time


##########################################################################
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



class resblock(nn.Module):
    def __init__(self, dim):

        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


##########################################################################
## Resizing modules
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


##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

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



##########################################################################
## Prompt to Feature Interaction
class PromptToFeature(nn.Module):
    def __init__(self, interaction_mode='cross_attention', interaction_opt=None):
        super(PromptToFeature, self).__init__()
        self.interaction_mode = interaction_mode
        self.interaction_opt = interaction_opt
        
        if self.interaction_mode == 'cross_attention':
            feat_dim = interaction_opt['feat_dim']
            prompt_dim = interaction_opt['prompt_dim']
            num_heads = interaction_opt['head']
            self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads)
            self.norm = nn.LayerNorm(feat_dim)
            self.prompt_map = nn.Linear(prompt_dim, feat_dim)
        elif self.interaction_mode == 'convgate':
            in_feat = interaction_opt['in_feat']
            out_feat = interaction_opt['out_feat']
            self.gate = nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, feat, prompt):
            if self.interaction_mode == 'cross_attention':
                B, C, H, W = feat.shape
                # 將 feat 和 prompt 轉換為 [L, N, E] 格式，L = H * W, N = B, E = C
                feat_ = rearrange(feat, 'b c h w -> (h w) b c')
                prompt_ = rearrange(prompt, 'b c h w -> (h w) b c')
                # 映射 prompt_ 的通道數
                prompt_ = self.prompt_map(prompt_.transpose(0, 1)).transpose(0, 1)  # [N, L, E] -> [L, N, E]
                feat_ = self.norm(feat_.transpose(0, 1)).transpose(0, 1)  # 確保與 prompt_ 格式一致
                feat_, _ = self.cross_attn(feat_, prompt_, prompt_)
                # 轉回 [B, C, H, W]
                feat = rearrange(feat_, '(h w) b c -> b c h w', h=H, w=W)
                return feat
            elif self.interaction_mode == 'convgate':
                gate = self.gate(prompt)
                return feat * gate
            else:
                raise ValueError(f"Unknown interaction mode: {self.interaction_mode}")



##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, high_prompt_dim=5, low_prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192, task_classes=5):
        super(PromptGenBlock,self).__init__()
        # self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.degradation_param = nn.Parameter(torch.rand(1, prompt_len // 2, high_prompt_dim, prompt_size, prompt_size))
        self.basic_param = nn.Parameter(torch.rand(1, prompt_len // 2, low_prompt_dim, prompt_size, prompt_size))
        # self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.degradation_linear_layer = nn.Linear(lin_dim, high_prompt_dim)
        self.basic_linear_layer = nn.Linear(lin_dim, low_prompt_dim)
        self.attn = nn.MultiheadAttention(embed_dim=low_prompt_dim, num_heads=4)

        # self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv3x3 = nn.Conv2d(low_prompt_dim, low_prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.degradation_map = nn.Conv2d(high_prompt_dim, low_prompt_dim, kernel_size=1, bias=False)

    def forward(self, x, degradation_class=None):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        # prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        degradation_weights = F.softmax(self.degradation_linear_layer(emb), dim=1)
        degradation_prompt = degradation_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.degradation_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        degradation_prompt = torch.sum(degradation_prompt, dim=1)
        degradation_prompt = F.interpolate(degradation_prompt, (H, W), mode="bilinear")
        degradation_prompt = self.degradation_map(degradation_prompt)

        basic_weights = F.softmax(self.basic_linear_layer(emb), dim=1)
        basic_prompt = basic_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.basic_param.unsqueeze(0).repeat(B, 1, 1, 1, 1, 1).squeeze(1)
        basic_prompt = torch.sum(basic_prompt, dim=1)
        basic_prompt = F.interpolate(basic_prompt, (H, W), mode="bilinear")
        
        
        # 轉換為適合 MultiheadAttention 的形狀 [L, N, E]
        degradation_prompt_ = rearrange(degradation_prompt, 'b c h w -> (h w) b c')
        basic_prompt_ = rearrange(basic_prompt, 'b c h w -> (h w) b c')

        # 進行提示間交互，使用 degradation_prompt 作為 query，basic_prompt 作為 key 和 value
        attn_output, _ = self.attn(degradation_prompt_, basic_prompt_, basic_prompt_)
        
        # 將交互結果轉回 [B, C, H, W]
        fused_prompt = rearrange(attn_output, '(h w) b c -> b c h w', h=H, w=W)

        prompt = self.conv3x3(fused_prompt)
        return prompt





##########################################################################
##---------- PromptIR -----------------------

class PromptIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        num_blocks=[6,8,8,10], 
        num_refinement_blocks=6,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        decoder=False,
        task_classes=5,  # 新增：支持多任務退化類型
        interaction_mode='cross_attention',  # 新增：提示與特徵交互模式
    ):
        super(PromptIR, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder
        self.task_classes = task_classes
        
        # 提示生成模塊
        if self.decoder:
            self.prompt1 = PromptGenBlock(high_prompt_dim=2, low_prompt_dim=64, prompt_len=3, prompt_size=64, lin_dim=96, task_classes=task_classes)
            self.prompt2 = PromptGenBlock(high_prompt_dim=2, low_prompt_dim=128, prompt_len=3, prompt_size=32, lin_dim=192, task_classes=task_classes)
            self.prompt3 = PromptGenBlock(high_prompt_dim=2, low_prompt_dim=320, prompt_len=3, prompt_size=16, lin_dim=384, task_classes=task_classes)
        
        # 提示與特徵交互模塊
        if self.decoder:
            self.interaction_level1 = PromptToFeature(
                interaction_mode=interaction_mode,
                interaction_opt={'feat_dim': dim, 'prompt_dim': 64, 'head': heads[0]}
            )
            self.interaction_level2 = PromptToFeature(
                interaction_mode=interaction_mode,
                interaction_opt={'feat_dim': int(dim*2**1), 'prompt_dim': 128, 'head': heads[1]}
            )
            self.interaction_level3 = PromptToFeature(
                interaction_mode=interaction_mode,
                interaction_opt={'feat_dim': int(dim*2**2), 'prompt_dim': 320, 'head': heads[2]}
            )

        self.chnl_reduce1 = nn.Conv2d(64, 64, kernel_size=1, bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128, 128, kernel_size=1, bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320, 256, kernel_size=1, bias=bias)

        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64, dim, kernel_size=1, bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2

        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128, int(dim*2**1), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3

        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256, int(dim*2**2), kernel_size=1, bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))  # From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**2))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224, int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  # From Level 2 to Level 1

        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64, int(dim*2**1), kernel_size=1, bias=bias)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                    
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, noise_emb=None, degradation_class=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            dec3_param = self.prompt3(latent, degradation_class)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        # 在跳躍連接處使用提示調節
        if self.decoder:
            out_enc_level3 = self.interaction_level3(out_enc_level3, dec3_param)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3, degradation_class)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        # 在跳躍連接處使用提示調節
        if self.decoder:
            out_enc_level2 = self.interaction_level2(out_enc_level2, dec2_param)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2, degradation_class)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        # 在跳躍連接處使用提示調節
        if self.decoder:
            out_enc_level1 = self.interaction_level1(out_enc_level1, dec1_param)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
