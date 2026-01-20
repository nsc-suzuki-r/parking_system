import os
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# src フォルダをモジュール検索パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.file import clear_existing_files
from utils.image import split_image
from utils.visistory_api import run_predictions

# .envファイルの内容を読み込む
load_dotenv()


def _capture_frame_from_youtube(youtube_url, output_path):
    """YouTubeライブストリームから1フレームを取得して保存する

    Args:
        youtube_url: YouTube動画のURL
        output_path: フレーム保存先パス

    Raises:
        RuntimeError: フレーム取得に失敗した場合
    """
    # ステップ1: yt-dlpで直接ストリームURLを取得
    print("yt-dlpでストリームURLを取得中...")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--cookies",
                "cookies.txt",
                "--get-url",
                "-f",
                "best[ext=mp4]",
                youtube_url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp error: {result.stderr}")

        stream_url = result.stdout.strip()
        if not stream_url:
            raise RuntimeError("ストリームURLの取得に失敗しました")

        print(f"ストリームURL取得完了")

    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timeout: ストリームURL取得に30秒以上かかりました")

    # ステップ2: ffmpegで直接ストリームURLからフレームを取得
    print("ffmpegでフレームを保存中...")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                stream_url,
                "-frames:v",
                "1",
                "-timeout",
                "10000000",  # 10秒のタイムアウト（マイクロ秒）
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"ffmpeg stderr: {result.stderr}")
            raise RuntimeError(f"ffmpeg error: {result.stderr}")

        print("ffmpeg処理完了")

    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timeout: フレーム保存に30秒以上かかりました")

    # フレームが保存されたかを確認
    print("フレーム保存の確認中...")
    if not os.path.exists(output_path):
        raise RuntimeError("フレームの保存に失敗しました")


def capture_split_predict_and_send(
    youtube_url, output_folder, target_folder, models_and_outputs
):
    """YouTubeライブからフレーム取得、分割、予測、API送信を実行

    Args:
        youtube_url: YouTube動画のURL
        output_folder: フレーム保存フォルダ
        target_folder: 分割後の出力フォルダ
        models_and_outputs: モデルと出力ファイルのマッピング辞書
    """
    try:
        print("出力ディレクトリの準備中...")
        # 出力ディレクトリを作成
        clear_existing_files(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # 現在時刻に基づいてファイル名を作成
        current_time = datetime.now().strftime("%H%M%S")
        output_path = os.path.join(output_folder, f"frame_{current_time}.jpg")

        # フレーム取得
        _capture_frame_from_youtube(youtube_url, output_path)
        print(f"フレームが正常に保存されました: {output_path}")

        # 画像分割
        split_segments = split_image(output_path, target_folder, models_and_outputs)
        print(f"分割完了: {len(split_segments)}個のセグメント")

        # 予測実行
        prediction_results = run_predictions(split_segments)
        print(f"予測結果: {prediction_results}")

        return prediction_results

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"エラーが発生しました: {e}")
        raise


# 使用例
if __name__ == "__main__":
    print("処理を開始します...")
    youtube_url = os.getenv("YOUTUBE_URL")

    output_folder = "data/upload"
    target_folder = "data/target"
    models_and_outputs = {
        "models/parking_model_takeda_a.pth": "takeda_a.jpg",
        "models/parking_model_takeda_b.pth": "takeda_b.jpg",
        "models/parking_model_takeda_c.pth": "takeda_c.jpg",
        "models/parking_model_takeda_d.pth": "takeda_d.jpg",
        "models/parking_model_rittai_p.pth": "rittai_p.jpg",
        "models/parking_model_bottom.pth": "bottom.jpg",
    }

    # 実行
    print("フレーム取得、分割、予測、送信を実行中...")
    capture_split_predict_and_send(
        youtube_url, output_folder, target_folder, models_and_outputs
    )
