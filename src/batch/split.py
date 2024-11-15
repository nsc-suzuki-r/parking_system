from PIL import Image
import os
import glob
import shutil  # ファイル移動のためにインポート


def split_image(filepath, target_folder, models_and_outputs):
    # 画像を開く
    image = Image.open(filepath)
    width, height = image.size
    segment_height = height // 6  # 6分割する高さ
    segments = []

    # ファイル名と日付パスを取得
    original_filename = os.path.basename(filepath)  # 例: frame_0002.jpg
    date_str = (
        os.path.dirname(filepath).replace("data/train/raw/", "").replace("/", "_")
    )  # 日付ディレクトリを "2024_11_06" に変換

    # 保存用のファイル名を生成（例: 2024_11_06_frame_0002.jpg）
    base_filename = f"{date_str}_{original_filename}"

    # 各モデル名に応じて分割して保存
    for i, (model_name, _) in enumerate(models_and_outputs.items()):
        box = (0, i * segment_height, width, (i + 1) * segment_height)
        segment = image.crop(box)

        # 各セグメントの保存先を設定
        model_folder = os.path.join(target_folder, model_name)
        os.makedirs(model_folder, exist_ok=True)

        segment_path = os.path.join(
            model_folder, base_filename
        )  # 動的なファイル名を使用
        segment.save(segment_path)
        segments.append((segment_path, model_name, base_filename))

    return segments


def process_all_images(output_folder, target_folder, models_and_outputs):
    # outputフォルダ内の全ての画像ファイルを取得
    all_images = glob.glob(os.path.join(output_folder, "**", "*.jpg"), recursive=True)

    # 分割済みの元画像を移動するディレクトリ（sample_dataと同じ構造）
    processed_base_dir = os.path.join(output_folder, "../processed")

    for filepath in all_images:
        # 各画像を分割して保存
        segments = split_image(filepath, target_folder, models_and_outputs)
        print(f"{filepath} の分割結果: {segments}")

        # 元の画像の相対パスを取得し、processed_data内で同じ構造にする
        relative_path = os.path.relpath(filepath, output_folder)
        processed_path = os.path.join(processed_base_dir, relative_path)

        # processed_data内のディレクトリ構造を再現
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

        # 元の画像を processed_data 内の対応するディレクトリに移動
        shutil.move(filepath, processed_path)
        print(f"元の画像 {filepath} を {processed_path} に移動しました")


# 使用例
output_folder = "data/train/raw"
target_folder = "data/train/split"
models_and_outputs = {
    "takeda_a": None,  # 画像名を動的に設定するため、値は任意でOK
    "takeda_b": None,
    "takeda_c": None,
    "takeda_d": None,
    "rittai_p": None,
    "bottom": None,
}

# 実行
process_all_images(output_folder, target_folder, models_and_outputs)
