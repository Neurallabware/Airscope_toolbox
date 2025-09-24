from .attention import Attention, DropPath, Mlp
from .embedding import BERTEmbedding
import torch.nn as nn
import torch
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module for dual streams"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Query from stream1, Key/Value from stream2
        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        # Query from stream2, Key/Value from stream1  
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        
        # Cross-attention: x1 attends to x2
        q1 = self.q1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv2 = self.kv2(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k2, v2 = kv2[0], kv2[1]
        
        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        
        x1_attended = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1_attended = self.proj1(x1_attended)
        x1_attended = self.proj_drop(x1_attended)
        x1_out = self.norm1(x1 + x1_attended)
        
        # Cross-attention: x2 attends to x1
        q2 = self.q2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv1 = self.kv1(x1).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]
        
        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        
        x2_attended = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x2_attended = self.proj2(x2_attended)
        x2_attended = self.proj_drop(x2_attended)
        x2_out = self.norm2(x2 + x2_attended)
        
        return x1_out, x2_out


class TemporalTransformer(nn.Module):
    """Transformer based temporal encoder"""

    def __init__(self, input_dim=256, seq_length=196, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1):
        super(TemporalTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.num_heads = num_heads

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask=None, return_sequence=False):
        # x shape: (batch, seq_length, input_dim)
        B, N, _ = x.shape

        x = self.input_projection(x)  # (B, N, embed_dim)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N)
                x = block(x, mask=mask_expanded)
            else:
                x = block(x)

        x = self.norm(x)
        
        if return_sequence:
            return x  # Return full sequence for cross-attention
        
        # Global pooling with mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            x = x * mask_expanded  # 0 for padding position

            valid_lengths = mask.sum(dim=1, keepdim=True).float()  # (B, 1)
            x_sum = x.sum(dim=1)  # (B, embed_dim)
            x_avg = x_sum / valid_lengths.clamp(min=1.0)  # (B, embed_dim)
        else:
            x_avg = x.mean(dim=1)  # (B, embed_dim)

        return x_avg


class DualStreamBERT(nn.Module):
    """Dual stream Transformer with multiple fusion strategies"""

    def __init__(self, input_dim=256, seq_length=196, embed_dim=512, depth=6, num_heads=8,
                 mlp_ratio=4., num_classes=2, drop_ratio=0.1, fusion_strategy='concat'):
        super(DualStreamBERT, self).__init__()
        
        self.fusion_strategy = fusion_strategy
        assert fusion_strategy in ['concat', 'add', 'cross_attention'], \
            "fusion_strategy must be one of: 'concat', 'add', 'cross_attention'"

        # Two independent Transformer encoders
        self.stream1 = TemporalTransformer(
            input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_ratio=drop_ratio
        )
        self.stream2 = TemporalTransformer(
            input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_ratio=drop_ratio
        )

        # Fusion-specific components
        if fusion_strategy == 'concat':
            fusion_dim = embed_dim * 2
            self.fusion_layers = nn.Sequential(
                nn.Linear(fusion_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(embed_dim // 2, num_classes)
            )
        
        elif fusion_strategy == 'add':
            # Element-wise addition requires same dimensionality
            fusion_dim = embed_dim
            self.fusion_layers = nn.Sequential(
                nn.Linear(fusion_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(embed_dim // 2, num_classes)
            )
        
        elif fusion_strategy == 'cross_attention':
            # Cross-attention fusion
            self.cross_attention = CrossAttentionFusion(
                dim=embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio
            )
            fusion_dim = embed_dim * 2  # Concatenate after cross-attention
            self.fusion_layers = nn.Sequential(
                nn.Linear(fusion_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(drop_ratio),
                nn.Linear(embed_dim // 2, num_classes)
            )

    def forward(self, mouse1_data, mouse2_data, mouse1_mask=None, mouse2_mask=None):
        
        if self.fusion_strategy == 'cross_attention':
            # Get sequence representations for cross-attention
            stream1_seq = self.stream1(mouse1_data, mouse1_mask, return_sequence=True)  # (B, N, embed_dim)
            stream2_seq = self.stream2(mouse2_data, mouse2_mask, return_sequence=True)  # (B, N, embed_dim)
            
            # Apply cross-attention
            stream1_attended, stream2_attended = self.cross_attention(stream1_seq, stream2_seq)
            
            # Global pooling after cross-attention
            if mouse1_mask is not None:
                mask1_expanded = mouse1_mask.unsqueeze(-1).float()
                stream1_features = (stream1_attended * mask1_expanded).sum(dim=1) / mouse1_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            else:
                stream1_features = stream1_attended.mean(dim=1)
                
            if mouse2_mask is not None:
                mask2_expanded = mouse2_mask.unsqueeze(-1).float()
                stream2_features = (stream2_attended * mask2_expanded).sum(dim=1) / mouse2_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            else:
                stream2_features = stream2_attended.mean(dim=1)
            
            # Concatenate cross-attended features
            fused_features = torch.cat([stream1_features, stream2_features], dim=1)
            
        else:
            # Get global representations
            stream1_features = self.stream1(mouse1_data, mouse1_mask)  # (batch, embed_dim)
            stream2_features = self.stream2(mouse2_data, mouse2_mask)  # (batch, embed_dim)
            
            if self.fusion_strategy == 'concat':
                # Concatenation fusion
                fused_features = torch.cat([stream1_features, stream2_features], dim=1)
                
            elif self.fusion_strategy == 'add':
                # Element-wise addition fusion
                fused_features = stream1_features + stream2_features

        logits = self.fusion_layers(fused_features)
        return logits


# Example usage and comparison
def create_model_variants(input_dim=256, seq_length=196, embed_dim=512, num_classes=2):
    """Create models with different fusion strategies for comparison"""
    
    models = {}
    
    # Concatenation fusion (original)
    models['concat'] = DualStreamBERT(
        input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim,
        num_classes=num_classes, fusion_strategy='concat'
    )
    
    # Element-wise addition fusion
    models['add'] = DualStreamBERT(
        input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim,
        num_classes=num_classes, fusion_strategy='add'
    )
    
    # Cross-attention fusion
    models['cross_attention'] = DualStreamBERT(
        input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim,
        num_classes=num_classes, fusion_strategy='cross_attention'
    )
    
    return models


# Test function to compare fusion strategies
def test_fusion_strategies():
    """Test all three fusion strategies with dummy data"""
    
    batch_size = 4
    seq_length = 196
    input_dim = 256
    
    # Create dummy data
    mouse1_data = torch.randn(batch_size, seq_length, input_dim)
    mouse2_data = torch.randn(batch_size, seq_length, input_dim)
    mouse1_mask = torch.ones(batch_size, seq_length).bool()
    mouse2_mask = torch.ones(batch_size, seq_length).bool()
    
    models = create_model_variants()
    
    print("Testing different fusion strategies:")
    for strategy, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(mouse1_data, mouse2_data, mouse1_mask, mouse2_mask)
            print(f"{strategy:15} - Output shape: {logits.shape}, Parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_fusion_strategies()