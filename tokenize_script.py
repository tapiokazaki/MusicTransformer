import os
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_TOKEN
from mortm.convert import MIDI2Seq
from datasets import load_from_disk
import time

# --- 定数の定義 ---
OUTPUT_NPZ_DIR = "tokenized_data"
VOCAB_SAVE_DIR = "tokenizer_vocab"
GRAND_PIANO_DATASET_PATH = "grand_piano_solo_dataset"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # スクリプトの実行ディレクトリ

# --- 1. ディレクトリの準備 ---
os.makedirs(os.path.join(BASE_DIR, OUTPUT_NPZ_DIR), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, VOCAB_SAVE_DIR), exist_ok=True)
print("ディレクトリ準備完了。")

# --- 2. Tokenizerの初期化 ---
tokenizer = Tokenizer(music_token=get_token_converter(TO_TOKEN))
tokenizer.save(os.path.join(BASE_DIR, VOCAB_SAVE_DIR))
print("Tokenizer初期化＆語彙保存完了。")

# --- 3. データセットのロード ---
grand_piano_data = load_from_disk(os.path.join(BASE_DIR, GRAND_PIANO_DATASET_PATH))
num_midi_files = grand_piano_data.num_rows
print(f"データセットロード完了。対象MIDIファイル数: {num_midi_files} 件")

# --- 4. MIDIファイルのトークン化と保存 ---
print("全MIDIファイルのトークン化を開始。")
start_time = time.time()

for i, entry in enumerate(grand_piano_data):
    print(entry)
    midi_location = entry['location']
    midi_file_name = os.path.basename(midi_location)
    actual_midi_file_abs_path = os.path.join(BASE_DIR, "dataset", midi_location)
    converter_midi_dir = os.path.dirname(actual_midi_file_abs_path)

    print(f"処理中 ({i+1}/{num_midi_files}): {midi_file_name} ", end='\r')

    # MIDIファイルが存在しない場合はスキップ (エラーハンドリングを最小化)
    if not os.path.exists(actual_midi_file_abs_path):
        print(f"\n警告: ファイルが見つかりません。スキップします: {midi_file_name}")
        continue

    converter = MIDI2Seq(
        tokenizer,
        directory=converter_midi_dir,
        file_name=midi_file_name,
        program_list=[0],
        split_measure=8
    )
    converter.convert()
    converter.save(os.path.join(BASE_DIR, OUTPUT_NPZ_DIR)) # 成功/失敗の戻り値は無視

end_time = time.time()
total_time = end_time - start_time

# --- 5. 処理結果のサマリー ---
print("\n\nトークン化処理完了。")
print(f"全処理時間: {total_time:.2f} 秒 ({total_time / 60:.2f} 分)")
print(f"トークン化されたデータは以下に保存されました: {os.path.join(BASE_DIR, OUTPUT_NPZ_DIR)}")