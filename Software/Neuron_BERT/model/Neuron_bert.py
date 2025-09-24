import torch
import torch.nn as nn
# from attention import Attention, DropPath, Mlp
# from embedding import BERTEmbedding

from .attention import Attention, DropPath, Mlp
from .embedding import BERTEmbedding

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


class Neuron_BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim=256, seq_length=256, embed_dim=512,
                 depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, drop_ratio=0.1, attn_drop_ratio=0.1, drop_path_ratio=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,

                 ):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(Neuron_BERT, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embed_dim

        self.num_heads = num_heads

        self.Bert_embedding = BERTEmbedding(input_dim=input_dim, seq_length=seq_length, embed_dim=embed_dim, dropout=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], act_layer=act_layer,
                 norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            act_layer(),
            nn.Dropout(drop_ratio),
            nn.Linear(embed_dim//2, embed_dim // 4),
            act_layer(),
            nn.Dropout(drop_ratio),
            nn.Linear(embed_dim // 4, 2))

        self.apply(_init_weights)

    def forward(self, x, segment_info):

        B, N, C = x.shape

        valid_mask = (segment_info != 0)  # [B, seq_len+1]

        # 创建注意力掩码 [B, 1, seq_len+1, seq_len+1]
        attention_mask = valid_mask.unsqueeze(1).unsqueeze(2) & valid_mask.unsqueeze(1).unsqueeze(3)

        # embedding the indexed sequence to sequence of vectors
        x = self.Bert_embedding(x, segment_info)

        # running over multiple transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)
        # extract cls token x[:, 0]
        x = self.output_linear(x[:, 0])

        return x


def _init_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


