import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_ratio=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)

        # 最初のpe初期化は残す。
        # このpeに直接sin/cosの値を埋め込んでいく形にする
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        # torch.log を使用 (math.log よりも推奨)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # (d_model/2,)

        # ここで直接peの偶数・奇数インデックスに値を代入
        # position (max_len, 1) と div_term (d_model/2,) がブロードキャストされる
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 不要になった div_term_full の計算と、2回目の pe 初期化は削除します。
        # div_term_full = torch.zeros(d_model)
        # div_term_full[0::2] = div_term # 偶数インデックス
        # div_term_full[1::2] = div_term # 奇数インデックス (Transformerでは同じ値を使う)

        # Transformerの入力が (batch_size, seq_len, dim) なので、
        # Positional Encodingは (1, max_len, d_model) にするのが最も汎用的
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_dim)
        # self.pe: (1, max_len, embedding_dim)
        # seq_len に合わせて self.pe をスライスする
        # x.size(1) は seq_len
        x = x + self.pe[:, :x.size(1), :] # ブロードキャストにより batch_size 次元にコピーされる
        return self.dropout(x)