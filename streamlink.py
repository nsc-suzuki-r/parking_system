import os
import subprocess
from datetime import datetime
import time
from dotenv import load_dotenv

# .envファイルの内容を読み込む
load_dotenv()

# YouTubeライブのURL
youtube_url = os.getenv("YOUTUBE_URL")
# 出力するフレームの間隔（秒）
frame_interval = 600  # 10分ごとに1フレーム

# 初期の日付に基づいて出力ディレクトリを作成
current_date = datetime.now().strftime("%Y/%m/%d")
output_dir = f"sample_data/{current_date}"
os.makedirs(output_dir, exist_ok=True)

try:
    while True:
        # 日付が変わったかをチェック
        new_date = datetime.now().strftime("%Y/%m/%d")
        if new_date != current_date:
            # 新しい日付に基づいたディレクトリを作成
            output_dir = f"sample_data/{new_date}"
            os.makedirs(output_dir, exist_ok=True)
            current_date = new_date  # 日付を更新

        # 現在時刻に基づいてファイル名を作成
        current_time = datetime.now().strftime("%H%M%S")
        output_path = f"{output_dir}/frame_{current_time}.jpg"

        # streamlinkとffmpegを使って1フレームを取得
        command = (
            f'streamlink -O "{youtube_url}" best | '
            f'ffmpeg -y -i - -frames:v 1 "{output_path}"'
        )

        # サブプロセスでコマンドを実行
        subprocess.run(command, shell=True)

        # 次のフレーム抽出まで待機
        time.sleep(frame_interval)

except KeyboardInterrupt:
    print("フレーム抽出が終了しました")
