import os
import subprocess
from datetime import datetime
from PIL import Image
import shutil  # ファイル移動のためにインポート
from dotenv import load_dotenv

# .envファイルの内容を読み込む
load_dotenv()

# YouTubeライブのURL
youtube_url = os.getenv("YOUTUBE_URL")


# 分割関数
def split_image(filepath, target_folder, models_and_outputs):
    image = Image.open(filepath)
    width, height = image.size
    segment_height = height // 6  # 6分割
    segments = []

    original_filename = os.path.basename(filepath)
    date_str = (
        os.path.dirname(filepath).replace("data/train/raw/", "").replace("/", "_")
    )
    base_filename = f"{date_str}_{original_filename}"

    for i, (model_name, _) in enumerate(models_and_outputs.items()):
        box = (0, i * segment_height, width, (i + 1) * segment_height)
        segment = image.crop(box)

        model_folder = os.path.join(target_folder, model_name)
        os.makedirs(model_folder, exist_ok=True)

        segment_path = os.path.join(model_folder, base_filename)
        segment.save(segment_path)
        segments.append((segment_path, model_name, base_filename))

    return segments


# フレーム取得＆分割処理
def capture_split(
    youtube_url, output_folder, target_folder, models_and_outputs
):
    # 現在の日付に基づいて出力ディレクトリを作成
    current_date = datetime.now().strftime("%Y/%m/%d")
    output_dir = os.path.join(output_folder, current_date)
    os.makedirs(output_dir, exist_ok=True)

    # 現在時刻に基づいてファイル名を作成
    current_time = datetime.now().strftime("%H%M%S")
    output_path = os.path.join(output_dir, f"frame_{current_time}.jpg")

    # streamlinkとffmpegを使って1フレームを取得
    command = (
        f'streamlink -O "{youtube_url}" best | '
        f'ffmpeg -y -i - -frames:v 1 "{output_path}"'
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"フレームが正常に保存されました: {output_path}")

        # 分割処理を実行
        split_segments = split_image(output_path, target_folder, models_and_outputs)
        print(f"分割完了: {split_segments}")

        # 元の画像を「processed」に移動
        processed_dir = os.path.join(output_folder, "../processed", current_date)
        os.makedirs(processed_dir, exist_ok=True)
        shutil.move(
            output_path, os.path.join(processed_dir, f"frame_{current_time}.jpg")
        )
        print(f"元の画像を {processed_dir} に移動しました")

    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")


# 使用例
output_folder = "data/train/raw"
target_folder = "data/train/split"
models_and_outputs = {
    "takeda_a": None,
    "takeda_b": None,
    "takeda_c": None,
    "takeda_d": None,
    "rittai_p": None,
    "bottom": None,
}

# 実行
capture_split(youtube_url, output_folder, target_folder, models_and_outputs)
