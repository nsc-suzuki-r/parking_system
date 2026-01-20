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

    # yt-dlpでライブストリームの動画データを取得し、ffmpegで1フレームを保存
    # ffmpegの出力をサイレントにしてエラーメッセージを抑制
    max_retries = 1  # 無限ループを防ぐため、本番環境では最大1回のみ実行

    for attempt in range(max_retries):
        try:
            command = (
                f'yt-dlp --cookies cookies.txt -o - -f "best[ext=mp4]" {youtube_url} 2>/dev/null | '
                f'ffmpeg -y -loglevel error -i pipe:0 -frames:v 1 "{output_path}" 2>/dev/null'
            )

            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )

            # ファイルが正常に作成されたか確認
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"フレームが正常に保存されました: {output_path}")

                # 分割処理を実行
                split_segments = split_image(
                    output_path, target_folder, models_and_outputs
                )
                print(f"分割完了: {split_segments}")

                # 分割結果を使って予測とAPI送信を実行
                prediction_results = run_predictions(split_segments)
                print(f"予測結果: {prediction_results}")
                break  # 成功したら終了
            else:
                # 403エラーなど認証エラーの場合
                if "403" in result.stderr or "bot" in result.stderr.lower():
                    print("エラー: YouTubeが認証を要求しています（403）")
                    print("→ cookieを更新してください:")
                    print(
                        "  yt-dlp --cookies-from-browser chrome --save-cookies cookies.txt 'https://www.youtube.com'"
                    )
                    raise Exception("Authentication required: 403 Forbidden")
                else:
                    raise Exception("Frame extraction failed")

        except subprocess.TimeoutExpired:
            print(f"タイムアウト: フレーム取得に時間がかかりすぎました")
            raise
        except Exception as e:
            print(f"エラーが発生しました: {e}")
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
