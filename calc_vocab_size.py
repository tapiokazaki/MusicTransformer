import os
import numpy as np
# --- ヘルパー関数: 語彙サイズの動的取得 (変更なし) ---
def get_vocab_size_from_npz(tokenized_data_dir: str) -> int:
    """
    トークン化されたNPZファイルから語彙サイズを推定します。
    NPZファイル内の最大トークンID + 1 を語彙サイズとします。
    """
    npz_files = [f for f in os.listdir(tokenized_data_dir) if f.endswith('.npz')]
    if not npz_files:
        raise FileNotFoundError(f"'{tokenized_data_dir}' にNPZファイルが見つかりません。")

    max_token_id = -1
    for npz_file in npz_files:
        npz_path = os.path.join(tokenized_data_dir, npz_file)
        try:
            data = np.load(npz_path)
            for key in data.files:
                seq = data[key]
                if seq.size > 0:
                    current_max = np.max(seq)
                    if current_max > max_token_id:
                        max_token_id = current_max
        except Exception as e:
            print(f"警告: {npz_file} の読み込み中にエラーが発生しました。スキップします。詳細: {e}")
            continue

    if max_token_id == -1:
        raise ValueError("NPZファイルから有効なトークンIDを検出できませんでした。")

    return int(max_token_id) + 1