import os
import sys

# src フォルダをモジュール検索パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import subprocess
from datetime import datetime
from dotenv import load_dotenv
from utils.file import clear_existing_files
from utils.image import split_image
from utils.visistory_api import run_predictions

# .envファイルの内容を読み込む
load_dotenv()

# YouTubeライブのURL
youtube_url = os.getenv("YOUTUBE_URL")


# フレーム取得、分割、予測、API送信の処理
def capture_split_predict_and_send(
    youtube_url, output_folder, target_folder, models_and_outputs
):
    # 出力ディレクトリを作成
    clear_existing_files(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # 現在時刻に基づいてファイル名を作成
    current_time = datetime.now().strftime("%H%M%S")
    output_path = os.path.join(output_folder, f"frame_{current_time}.jpg")
    temp_video = os.path.join(output_folder, f"temp_{current_time}.mp4")

    # yt-dlpでライブストリームの動画をファイルにダウンロード
    # （パイプを使わない → タイムアウト回避）
    max_retries = 1

    for attempt in range(max_retries):
        try:
            print(f"\n試行 {attempt + 1}/{max_retries}: フレーム取得中...")

            # Step 1: yt-dlpでファイルにダウンロード
            download_cmd = (
                f'yt-dlp --cookies cookies.txt -o "{temp_video}" '
                f'-f "best[ext=mp4]" {youtube_url}'
            )
            print(f"実行コマンド: {download_cmd[:100]}...")

            result = subprocess.run(
                download_cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            # ダウンロード失敗チェック
            if result.returncode != 0:
                print(f"ダウンロードエラー (コード: {result.returncode})")
                if result.stdout:
                    print(f"stdout: {result.stdout[:500]}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:500]}")

                if "403" in result.stderr or "Sign in" in result.stderr:
                    print("エラー: YouTubeが認証を要求しています")
                    print("→ cookieを更新してください:")
                    print(
                        "  yt-dlp --cookies-from-browser chrome --save-cookies cookies.txt 'https://www.youtube.com'"
                    )
                    raise Exception("Authentication required: 403 Forbidden")
                raise Exception(f"Download failed")

            # ファイル確認
            if not os.path.exists(temp_video) or os.path.getsize(temp_video) < 1000:
                print(
                    f"ダウンロードファイルサイズが不足: {os.path.getsize(temp_video) if os.path.exists(temp_video) else 0} bytes"
                )
                raise Exception("Downloaded file is invalid")

            print(f"ダウンロード成功: {os.path.getsize(temp_video)} bytes")

            # Step 2: ffmpegでフレーム抽出
            extract_cmd = (
                f'ffmpeg -y -i "{temp_video}" -frames:v 1 -q:v 2 "{output_path}" '
                f"-loglevel error 2>/dev/null"
            )
            print(f"実行コマンド: {extract_cmd[:100]}...")

            result = subprocess.run(
                extract_cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            # クリーンアップ
            if os.path.exists(temp_video):
                os.remove(temp_video)

            # フレーム確認
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                print(f"フレーム抽出エラー")
                if result.stdout:
                    print(f"stdout: {result.stdout[:500]}")
                if result.stderr:
                    print(f"stderr: {result.stderr[:500]}")
                raise Exception("Frame extraction failed")

            print(f"フレームが正常に保存されました: {output_path}")

            # 分割処理を実行
            split_segments = split_image(output_path, target_folder, models_and_outputs)
            print(f"分割完了: {split_segments}")

            # 分割結果を使って予測とAPI送信を実行
            prediction_results = run_predictions(split_segments)
            print(f"予測結果: {prediction_results}")
            break  # 成功したら終了

        except subprocess.TimeoutExpired as e:
            if os.path.exists(temp_video):
                os.remove(temp_video)
            print(f"\n【タイムアウト発生】")
            print(f"試行 {attempt + 1}/{max_retries}")
            print(f"タイムアウト時間: 10秒")
            print(f"実行していたコマンド: {str(e.cmd)[:200]}")
            if attempt < max_retries - 1:
                print(f"→ 次を試行します...\n")
            else:
                print(f"→ 最大試行回数に達しました")
                raise
        except Exception as e:
            if os.path.exists(temp_video):
                os.remove(temp_video)
            print(f"\n【エラー発生】")
            print(f"試行 {attempt + 1}/{max_retries}")
            print(f"エラー内容: {e}")
            if attempt < max_retries - 1:
                print(f"→ 次を試行します...\n")
            else:
                print(f"→ 最大試行回数に達しました")
                raise


# 使用例
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
