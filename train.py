import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping # EarlyStoppingを追加
from pytorch_lightning.loggers import TensorBoardLogger
from model import MyMusicTransformerModel
from datamodule import MusicDataModule
from calc_vocab_size import get_vocab_size_from_npz
import torch

# A100のTensor Coreを最大限に活用するために追加
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high') # ここを追加！

# --- メインのトレーニング関数 ---
def main():
    print("--- 音楽Transformerモデルのトレーニングを開始 ---")
    # 1. YAML設定ファイルの読み込み
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config/config.yaml")
    if not os.path.exists(config_path): raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 設定値を変数に展開
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    callbacks_config = config['callbacks']
    logger_config = config['logger']

    embedding_dim = model_config['embedding_dim']
    dim_feedforward_dynamic = embedding_dim * 4
    model_config['dim_feedforward'] = dim_feedforward_dynamic

    base_dir = os.path.dirname(os.path.abspath(__file__))

    tokenized_data_dir = os.path.join(base_dir, data_config['tokenized_data_dir'])
    mortm_tokenizer_vocab_path = os.path.join(base_dir, data_config['tokenizer_vocab_path'])

    # VOCAB_SIZE を動的に取得
    try:
        vocab_size_dynamic = get_vocab_size_from_npz(tokenized_data_dir)
        print(f"動的に取得された語彙サイズ (VOCAB_SIZE): {vocab_size_dynamic}")
        model_config['vocab_size'] = vocab_size_dynamic
    except (FileNotFoundError, ValueError) as e:
        print(f"致命的なエラー: 語彙サイズを動的に取得できませんでした。configのデフォルト値を使用します。詳細: {e}")
        print("NPZファイルが 'tokenized_data/' ディレクトリに正しく保存されているか確認してください。")
        if 'vocab_size' not in model_config or model_config['vocab_size'] is None:
            raise ValueError("語彙サイズが指定されていません。configファイルに設定するか、NPZファイルを配置してください。")
        print(f"Configファイルから取得された語彙サイズ (VOCAB_SIZE): {model_config['vocab_size']}")

    # 2. データモジュールの初期化
    music_data_module = MusicDataModule(
        data_dir=tokenized_data_dir,
        mortm_tokenizer_vocab_path=mortm_tokenizer_vocab_path,
        batch_size=training_config['batch_size'],
        max_seq_len=model_config['max_seq_len'],
        min_seq_len=model_config['min_seq_len'],
        num_workers=os.cpu_count(),
        pad_token_id=data_config['pad_token_id']
    )

    # 3. モデルの初期化
    model = MyMusicTransformerModel(
        vocab_size=model_config['vocab_size'],
        pad_token_id=data_config['pad_token_id'],
        embedding_dim=model_config['embedding_dim'],
        nhead=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        lr=training_config['learning_rate'],
        max_seq_len=model_config['max_seq_len'],
        dropout_ratio=model_config['dropout_ratio']
    )

    # 4. コールバックの設定
    checkpoint_callback = ModelCheckpoint(
        monitor=callbacks_config['checkpoint']['monitor'],
        dirpath=callbacks_config['checkpoint']['dirpath'],
        filename=callbacks_config['checkpoint']['filename'],
        save_top_k=callbacks_config['checkpoint']['save_top_k'],
        mode=callbacks_config['checkpoint']['mode'],
        # save_last=True # YAML設定にないが追加しても良い
    )

    lr_monitor = LearningRateMonitor(logging_interval=callbacks_config['lr_monitor_logging_interval'])

    # ★ EarlyStopping コールバックの追加 ★
    early_stopping_callback = EarlyStopping(
        monitor=callbacks_config['early_stopping']['monitor'],
        mode=callbacks_config['early_stopping']['mode'],
        patience=callbacks_config['early_stopping']['patience'],
        verbose=True # 早期終了時にメッセージを出力
    )

    # 5. ロガーの設定
    logger = TensorBoardLogger(logger_config['save_dir'], name=logger_config['name'])

    # 6. トレーナーの初期化と学習の開始
    trainer = pl.Trainer(
        accelerator=training_config['accelerator'],
        devices=training_config['devices'],
        max_epochs=training_config['max_epochs'],
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback], #EarlyStopping
        logger=logger,
        log_every_n_steps=50,
        precision=training_config['precision'], # ★ YAMLで 'bf16-mixed'-> flash-attn
        gradient_clip_val=training_config['gradient_clip_val'],
        gradient_clip_algorithm=training_config['gradient_clip_algorithm'],
    )

    print(f"学習を開始します (アクセラレーター: {training_config['accelerator']}, エポック数: {training_config['max_epochs']})")
    trainer.fit(model, datamodule=music_data_module)
    print("--- 学習完了 ---")

if __name__ == "__main__":
    main()