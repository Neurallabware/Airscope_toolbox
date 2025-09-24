import torch.nn as nn

import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, seq_length, embed_dim, dropout=0.1, learnable_pos_embed=True):
        """
        :param embed_dim: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.learnable_pos_embed = learnable_pos_embed

        self.input_projection = nn.Linear(input_dim, embed_dim)

        # 1, 1, embedding_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if learnable_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_length + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=seq_length + 1)

        # B, seq_len -> B, seq_len, embedding （vocab_dim 为3）
        self.segment = SegmentEmbedding(embed_size=embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_dim

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, segment_label):

        ## TODO: segment_label size is B, length+1, 1
        # sequence: B, length, input_size  segment_label: B, length+1, 1
        B, length, input_d = x.shape
        assert length==self.seq_length and self.input_dim==input_d , \
            f"Input sequence size ({length}*{input_d}) doesn't match model ({self.seq_length}*{self.input_dim})."

        # B, length, input_size -> B, length, input_size
        x = self.input_projection(x)

        cls_token = self.cls_token.expand(B, -1, -1)

        # B, length, input_size -> B, length+1, input_size
        x = torch.cat([cls_token, x], dim=1)

        x = x + self.segment(segment_label)
        if self.learnable_pos_embed:
            x += self.pos_embed
        else:
            x += self.pos_embed(x)

        return self.dropout(x)