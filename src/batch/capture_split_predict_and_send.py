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
youtube_cookie_sid_key = os.getenv("YOUTUBE_COOKIE_SID_KEY")
youtube_cookie_hsid_key = os.getenv("YOUTUBE_COOKIE_HSID_KEY")
youtube_cookie_ssid_key = os.getenv("YOUTUBE_COOKIE_SSID_KEY")
youtube_cookie_sapisid_key = os.getenv("YOUTUBE_COOKIE_SAPISID_KEY")
youtube_cookie_secure_1psidts_key = os.getenv("YOUTUBE_COOKIE_SECURE_1PSIDTS_KEY")
youtube_cookie_secure_1papisid_key = os.getenv("YOUTUBE_COOKIE_SECURE_1PAPISID_KEY")
youtube_cookie_secure_1psid_key = os.getenv("YOUTUBE_COOKIE_SECURE_1PSID_KEY")
youtube_cookie_secure_1psidcc_key = os.getenv("YOUTUBE_COOKIE_SECURE_1PSIDCC_KEY")
youtube_cookie_secure_3psidts_key = os.getenv("YOUTUBE_COOKIE_SECURE_3PSIDTS_KEY")
youtube_cookie_secure_3papisid_key = os.getenv("YOUTUBE_COOKIE_SECURE_3PAPISID_KEY")
youtube_cookie_secure_3psid_key = os.getenv("YOUTUBE_COOKIE_SECURE_3PSID_KEY")
youtube_cookie_secure_3psidcc_key = os.getenv("YOUTUBE_COOKIE_SECURE_3PSIDCC_KEY")
youtube_cookie_sid_value = os.getenv("YOUTUBE_COOKIE_SID_VALUE")
youtube_cookie_hsid_value = os.getenv("YOUTUBE_COOKIE_HSID_VALUE")
youtube_cookie_ssid_value = os.getenv("YOUTUBE_COOKIE_SSID_VALUE")
youtube_cookie_sapisid_value = os.getenv("YOUTUBE_COOKIE_SAPISID_VALUE")
youtube_cookie_secure_1psidts_value = os.getenv("YOUTUBE_COOKIE_SECURE_1PSIDTS_VALUE")
youtube_cookie_secure_1papisid_value = os.getenv("YOUTUBE_COOKIE_SECURE_1PAPISID_VALUE")
youtube_cookie_secure_1psid_value = os.getenv("YOUTUBE_COOKIE_SECURE_1PSID_VALUE")
youtube_cookie_secure_1psidcc_value = os.getenv("YOUTUBE_COOKIE_SECURE_1PSIDCC_VALUE")
youtube_cookie_secure_3psidts_value = os.getenv("YOUTUBE_COOKIE_SECURE_3PSIDTS_VALUE")
youtube_cookie_secure_3papisid_value = os.getenv("YOUTUBE_COOKIE_SECURE_3PAPISID_VALUE")
youtube_cookie_secure_3psid_value = os.getenv("YOUTUBE_COOKIE_SECURE_3PSID_VALUE")
youtube_cookie_secure_3psidcc_value = os.getenv("YOUTUBE_COOKIE_SECURE_3PSIDCC_VALUE")


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

    # streamlinkとffmpegを使って1フレームを取得
    command = (
        f'streamlink --http-cookie "{youtube_cookie_sid_key}={youtube_cookie_sid_value}" '
        f'--http-cookie "{youtube_cookie_hsid_key}={youtube_cookie_hsid_value}" '
        f'--http-cookie "{youtube_cookie_ssid_key}={youtube_cookie_ssid_value}" '
        f'--http-cookie "{youtube_cookie_sapisid_key}={youtube_cookie_sapisid_value}" '
        f'--http-cookie "{youtube_cookie_secure_1psidts_key}={youtube_cookie_secure_1psidts_value}" '
        f'--http-cookie "{youtube_cookie_secure_1papisid_key}={youtube_cookie_secure_1papisid_value}" '
        f'--http-cookie "{youtube_cookie_secure_1psid_key}={youtube_cookie_secure_1psid_value}" '
        f'--http-cookie "{youtube_cookie_secure_1psidcc_key}={youtube_cookie_secure_1psidcc_value}" '
        f'--http-cookie "{youtube_cookie_secure_3psidts_key}={youtube_cookie_secure_3psidts_value}" '
        f'--http-cookie "{youtube_cookie_secure_3papisid_key}={youtube_cookie_secure_3papisid_value}" '
        f'--http-cookie "{youtube_cookie_secure_3psid_key}={youtube_cookie_secure_3psid_value}" '
        f'--http-cookie "{youtube_cookie_secure_3psidcc_key}={youtube_cookie_secure_3psidcc_value}" '
        f'-O "{youtube_url}" best | '
        f'ffmpeg -y -i - -frames:v 1 "{output_path}"'
    )
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"フレームが正常に保存されました: {output_path}")

        # 分割処理を実行
        split_segments = split_image(output_path, target_folder, models_and_outputs)
        print(f"分割完了: {split_segments}")

        # 分割結果を使って予測とAPI送信を実行
        print(split_segments)
        prediction_results = run_predictions(split_segments)
        print(f"予測結果: {prediction_results}")

    except subprocess.CalledProcessError as e:
        print(f"エラーが発生しました: {e}")


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
