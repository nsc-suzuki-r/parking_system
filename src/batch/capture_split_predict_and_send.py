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
    # yt-dlpプロセス（クッキーファイルを使用）
    yt_dlp_process = subprocess.Popen(
        [
            "yt-dlp",
            "--cookies",
            "cookies.txt",
            "--socket-timeout",
            "30",
            "-o",
            "-",
            "-f",
            "best[ext=mp4]",
            youtube_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # ffmpegプロセス
    ffmpeg_process = subprocess.Popen(
        ["ffmpeg", "-y", "-i", "pipe:0", "-frames:v", "1", output_path],
        stdin=yt_dlp_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # yt-dlpの標準出力をffmpegにパイプ
    yt_dlp_process.stdout.close()

    # 両プロセスの終了を待つ
    ffmpeg_process.communicate()
    yt_dlp_stdout, yt_dlp_stderr = yt_dlp_process.communicate()

    # フレームが保存されたかを確認
    if not os.path.exists(output_path):
        if yt_dlp_process.returncode != 0:
            error_msg = (
                yt_dlp_stderr.decode("utf-8", errors="ignore")
                if yt_dlp_stderr
                else "Unknown error"
            )
            raise RuntimeError(f"yt-dlp error: {error_msg}")
        else:
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
        # 出力ディレクトリを作成
        clear_existing_files(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # 現在時刻に基づいてファイル名を作成
        current_time = datetime.now().strftime("%H%M%S")
        output_path = os.path.join(output_folder, f"frame_{current_time}.jpg")

        # フレーム取得
        _capture_frame_from_youtube(youtube_url, output_path)
        print(f"✓ フレームが正常に保存されました: {output_path}")

        # 画像分割
        split_segments = split_image(output_path, target_folder, models_and_outputs)
        print(f"✓ 分割完了: {len(split_segments)}個のセグメント")

        # 予測実行
        prediction_results = run_predictions(split_segments)
        print(f"✓ 予測結果: {prediction_results}")

        return prediction_results

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"✗ エラーが発生しました: {e}")
        raise


# 使用例
if __name__ == "__main__":
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
    capture_split_predict_and_send(
        youtube_url, output_folder, target_folder, models_and_outputs
    )
