import torch
import os
import yaml
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from mortm.de_convert import ct_token_to_midi
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MIDI2Seq
from model import MyMusicTransformerModel
from calc_vocab_size import get_vocab_size_from_npz

# --- 設定ファイルの読み込み (変更なし) ---
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.yaml")
if not os.path.exists(config_path): raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
data_config = config['data']

base_dir = os.path.dirname(os.path.abspath(__file__))
tokenized_data_dir = os.path.join(base_dir, data_config['tokenized_data_dir'])
print("処理を開始します")

# --- トークナイザーとMIDIコンバーターの準備 (変更なし) ---
converter_midi_dir = os.path.join(base_dir, "converter_midi")
tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))
midi_file_name = "test_midi/test_midi.mid"
converter = MIDI2Seq(
    tokenizer,
    directory=converter_midi_dir,
    file_name=midi_file_name,
    program_list=[0],
    split_measure=8
)
converter.convert()
seq = converter.aya_node[1][:-1]
print("シーケンスの読み込みが完了しました")

# <<< 変更点: モデルの読み込み方法を修正 >>>
# .ckptファイルからモデルを正しく復元するには .load_from_checkpoint を使います
checkpoint_path = "checkpoints/music-transformer-epoch=02-val_loss=2.7500.ckpt"
model = MyMusicTransformerModel.load_from_checkpoint(checkpoint_path)

# <<< 変更点: PyTorch LightningのTrainerを使ってpredictを呼び出す >>>
# 1. 開始シーケンスをテンソルに変換
input_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0)  # (1, seq_len) の形状にする

# 2. DataLoaderを作成
# predict_stepはDataLoaderからデータをバッチで受け取るため、
# 開始シーケンスを1つだけ持つデータセットとデータローダーを作成します。
predict_dataset = TensorDataset(input_tensor)
predict_dataloader = DataLoader(predict_dataset, batch_size = 1)

# 3. Trainerを初期化して .predict() を実行
print("生成を開始します...")
# accelerator="auto" でGPUが利用可能なら自動で選択
trainer = pl.Trainer(accelerator="auto", logger=False)
# trainer.predict() で model.predict_step() が呼び出される
generated_batch = trainer.predict(model, predict_dataloader)

# 4. 結果の取り出し
# trainer.predictはリストで結果を返す。今回はバッチが1つなので最初の要素を取得
generated_sequence_tensor = generated_batch[0]
final_seq = generated_sequence_tensor.cpu().np().flatten() # (1, seq_len) -> (seq_len,)

print("生成が完了しました。")
print(final_seq)

# --- MIDIへの変換と保存 (変更なし) ---
tokenizer.rev_mode()
midi_seq = ct_token_to_midi()
midi_seq = ct_token_to_midi(torch.tensor(final_seq), tokenizer, save_directory="output.mid", program=0)
# MIDIの保存処理をここに追加してください
# 例: midi_seq.dump("output.mid")