import os
from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl
# MORTMトークナイザーで変換されたシーケンスを保持するクラス
from mortm.train.datasets import MORTM_SEQDataset, LearningProgress

# --- Lightning DataModule の定義 ---
class MusicDataModule(pl.LightningDataModule):
    """MORTMトークン化された音楽データセットを管理するLightning DataModule。"""
    def __init__(self,
                 data_dir: str,  # トークン化されたNPZファイルがあるディレクトリ
                 mortm_tokenizer_vocab_path: str,  # トークナイザーの語彙ファイルがあるパス
                 batch_size: int = 32,
                 max_seq_len: int = 1024,  # モデルの positional_encoding とも関連する最大シーケンス長
                 min_seq_len: int = 32,  # データセットに含める最小シーケンス長
                 train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                 num_workers: int = 4,
                 pad_token_id: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.mortm_tokenizer_vocab_path = mortm_tokenizer_vocab_path  # パディングトークンID取得用
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.train_ratio, self.val_ratio, self.test_ratio = train_val_test_split
        self.num_workers = num_workers
        self.pad_token_id = pad_token_id
        self.train_dataset: Optional[MORTM_SEQDataset] = None
        self.val_dataset: Optional[MORTM_SEQDataset] = None
        self.test_dataset: Optional[MORTM_SEQDataset] = None
        self.progress_manager: LearningProgress = LearningProgress()
    def setup(self, stage: Optional[str] = None):
        """
        データセットをロードし、トレーニング、バリデーション、テストセットに分割します。
        fit, validate, test, predictの各ステージに応じて呼び出されます。
        """
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None: return  # 既にセットアップ済みなら何もしない

        # 全NPZファイルをリストアップ
        npz_files = [str(f) for f in os.listdir(self.data_dir) if f.endswith('.npz')]  # ★ 修正案 ★
        if not npz_files: raise FileNotFoundError(f"データディレクトリ '{self.data_dir}' にNPZファイルが見つかりません。")

        # MORTM_SEQDatasetのインスタンスを生成
        full_dataset_instance = MORTM_SEQDataset(
            progress=self.progress_manager,  # 正しい型のprogress_managerが渡される
            positional_length=self.max_seq_len,#->1024
            min_length=self.min_seq_len #->32
        )

        # NPZファイルを読み込み、データセットに追加
        print(f"NPZファイルをロード中: {len(npz_files)} 個")
        total_added_seqs = 0
        for i, npz_file in enumerate(npz_files):
            npz_path = os.path.join(self.data_dir, str(npz_file))
            try:
                music_seq_data = np.load(npz_path)
                added_count = full_dataset_instance.add_data(music_seq_data)
                total_added_seqs += added_count
                print(
                    f"  {i + 1}/{len(npz_files)}: {npz_file} から {added_count} シーケンス追加済み. 合計: {total_added_seqs}",
                    end='\r')
            except Exception as e:
                print(f"\n警告: {npz_file} のロード中にエラー: {e} スキップします。")
        print(f"\n全NPZファイルのロード完了。合計 {total_added_seqs} シーケンスをデータセットに追加しました。")

        # データセットを分割
        total_sequences = len(full_dataset_instance)
        train_size = int(total_sequences * self.train_ratio)
        val_size = int(total_sequences * self.val_ratio)
        test_size = total_sequences - train_size - val_size  # 残りをテストに

        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset_instance, [train_size, val_size, test_size]
        )
        print(f"データセット分割: Train({len(self.train_dataset)}), Val({len(self.val_dataset)}), Test({len(self.test_dataset)})")

    def _collate_fn_pad(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DataLoaderのcollate_fnとして使用し、バッチ内のシーケンスをパディングします。
        srcとtgt（入力とターゲットラベル）を生成します。
        """
        # pad_sequence を使って一括でパディング
        # batch はリストなので、pad_sequence の引数には合致する
        # pad_sequence はデフォルトで batch_first=False なので、transposeで形状を調整
        padded_batch = torch.nn.utils.rnn.pad_sequence(
            batch,
            batch_first=True,  # (batch_size, max_len, ...) 形状にする
            padding_value=self.pad_token_id
        )

        # src (エンコーダー入力) と tgt_input (デコーダー入力) を独立したテンソルとして生成
        src_batch = padded_batch
        tgt_input_batch = padded_batch.clone()

        # モデルの_common_stepの期待する形式に合わせて返す
        return src_batch, tgt_input_batch

    def train_dataloader(self) -> DataLoader:
        """トレーニング用DataLoaderを返します。"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn_pad,  # self.pad_token_id を渡さないように修正済
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """バリデーション用DataLoaderを返します。"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn_pad,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """テスト用DataLoaderを返します。"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn_pad,
            pin_memory=True
        )

    def predict_dataloader(self) -> DataLoader:
        """予測用DataLoaderを返します。"""
        return self.test_dataloader()