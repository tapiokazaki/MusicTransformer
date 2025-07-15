# model.py (修正後)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, Tuple
from layer.PositionalEncoding import PositionalEncoding
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class MyMusicTransformerModel(pl.LightningModule):
    """
    MORTMのトークナイザーと連携する音楽Transformerモデル。(デコーダーのみ版)
    """

    # <<< 変更点: __init__の引数から num_encoder_layers を削除し、dim_feedforward を追加 >>>
    def __init__(self, vocab_size: int, pad_token_id: int = 0, embedding_dim: int = 512,
                 nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048,
                 lr: float = 0.001, weight_decay: float = 0.0001,
                 max_seq_len: int = 1024, dropout_ratio: float = 0.1):

        super().__init__()
        self.save_hyperparameters() #パラメータ
        self.embedding = nn.Embedding(self.hparams.vocab_size, self.hparams.embedding_dim,
                                      padding_idx=self.pad_token_id)

        self.positional_encoding = PositionalEncoding(
            d_model=self.hparams.embedding_dim,
            dropout_ratio=self.hparams.dropout_ratio,
            max_len=self.hparams.max_seq_len
        )

        # <<< 変更点: nn.Transformer を nn.TransformerEncoder に変更 >>>
        # エンコーダーレイヤーを定義
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hparams.embedding_dim,
            nhead=self.hparams.nhead,
            dim_feedforward=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout_ratio,
            batch_first=True
        )
        # エンコーダーレイヤーを積み重ねてTransformerエンコーダー（デコーダーとして使用）
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.hparams.num_layers
        )
        # <<< 変更点ここまで >>>

        self.fc = nn.Linear(self.hparams.embedding_dim, self.hparams.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    # <<< 変更点: forwardメソッドの引数を簡略化 >>>
    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)

        #TransformerEncoderの呼び出し、generate_square_subsequent_mask は呼び出し側で作成している。
        output = self.transformer(
            src=x,
            mask=attn_mask,
            src_key_padding_mask=padding_mask
        )
        output = self.fc(output)
        return output

    # <<< 変更点: _common_step をデコーダーのみの入力形式に修正 >>>
    def _common_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # srcとtgtは同じなので、片方だけ使用する
        input_sequence, _ = batch

        # デコーダーへの入力と、正解ラベルを生成
        input_for_decoder = input_sequence[:, :-1].contiguous()
        target_labels = input_sequence[:, 1:].contiguous()

        # マスクを生成
        padding_mask = (input_for_decoder == self.pad_token_id)
        tgt_seq_len = input_for_decoder.size(1)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

        output = self.forward(
            input_for_decoder,
            padding_mask=padding_mask,
            attn_mask=attn_mask
        )

        loss = self.criterion(output.view(-1, output.size(-1)), target_labels.view(-1))
        return loss

    # training_step, validation_step, test_step は _common_step を呼び出すだけなので変更不要
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._common_step(batch)
        self.log('test_loss', loss, on_epoch=True, logger=True) #->tensorbordにlogを記録
        return loss

    # <<< 変更点: predict_stepをデコーダーのみの形式に修正 >>>
    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # DataLoaderから渡されるバッチを受け取る
        # バッチがタプルやリストの場合、最初の要素を開始シーケンスとして使用
        if isinstance(batch, (list, tuple)):
            start_sequence, *_ = batch
        else:
            start_sequence = batch

        generated_sequence = start_sequence.to(self.device)
        # 終了条件となるトークンIDのリスト
        eos_tokens = [585, 586]

        self.eval()
        with torch.no_grad():
            # max_seq_lenに達するまで、または終了トークンが生成されるまでループ
            # 現在のシーケンス長から、あと何トークン生成できるかを計算
            for _ in range(self.hparams.max_seq_len - generated_sequence.size(1)):
                tgt_seq_len = generated_sequence.size(1)
                # 後続のトークンを見ないようにするためのマスクを生成
                causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

                # forwardに渡すのは生成中のシーケンスのみ
                output = self.forward(
                    generated_sequence,
                    attn_mask=causal_mask
                )

                # 次のトークンを予測
                next_token_logits = output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # 生成したトークンをシーケンスの末尾に追加
                generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

                # 終了条件をチェック
                if next_token.item() in eos_tokens:
                    print(f"終了トークン {next_token.item()} が生成されたため停止します。")
                    break

        # predict()はモデルの状態を元に戻さないので、train()の呼び出しは不要
        return generated_sequence

    # configure_optimizers は変更不要
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            'scheduler': CosineAnnealingWarmRestarts(
                optimizer, T_0=3, T_mult=1, eta_min=1e-6
            ),
            'interval': 'epoch', 'monitor': 'val_loss',
        }
        return [optimizer], [scheduler]