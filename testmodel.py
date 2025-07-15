import  torch
import pytorch_lightning as pl
from torch import nn
from typing import Optional
from layer.PositionalEncoding import PositionalEncoding
class TestModel(pl.LightningModule):
    def __init__(self, vocab_size: int ,
                 embedding_dim:int,
                 nhead: int ,
                 num_encoder_layers: int,
                 num_decoder_layers:int ,
                 dropout_ratio: float=0.3,
                 pad_token_id: int = 0,
                 max_seq_len: int = 1024,
                 lr:float = 0.001,
                 weight_decay: float = 0.0001
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.embedding=nn.Embedding(embedding_dim,vocab_size,padding_idx=pad_token_id)
        self.positional_encoding = PositionalEncoding(
            d_model=embedding_dim,
            dropout_ratio=dropout_ratio,
            max_len=self.hparams.max_seq_len
        )

        self.transformer = nn.Transformer(vocab_size, embedding_dim, nhead, num_encoder_layers, num_decoder_layers, batch_first=True, dropout=dropout_ratio)
        self.fc= nn.Linear(embedding_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad_token_id)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """モデルの順伝播を定義します。"""
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # look_ahead_mask #tgt.size(1)はseq_len
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.device)

        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        output = self.fc(output)
        return output
