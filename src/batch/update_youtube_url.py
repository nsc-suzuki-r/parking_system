import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")
TARGET_TITLE = os.getenv("TARGET_TITLE")


def get_live_url(api_key, channel_id, target_title=TARGET_TITLE):
    # チャンネルの最新動画を検索（ライブ配信のみ）
    url = f"https://www.googleapis.com/youtube/v3/search?part=id&channelId={channel_id}&type=video&eventType=live&key={api_key}"
    res = requests.get(url).json()

    for item in res.get("items", []):
        video_id = item["id"]["videoId"]
        detail_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,liveStreamingDetails&id={video_id}&key={api_key}"
        detail = requests.get(detail_url).json()

        if detail.get("items"):
            video_info = detail["items"][0]
            title = video_info["snippet"]["title"]
            live_info = video_info["snippet"]["liveBroadcastContent"]

            # タイトルが一致し、かつライブ配信中の場合
            if live_info == "live" and target_title in title:
                return f"https://www.youtube.com/watch?v={video_id}"

    return None


def write_to_env(live_url, env_path=".env"):
    """URLを .env に書き込む"""
    env_file = Path(env_path)
    lines = []

    # 既存ファイルを読み込み
    if env_file.exists():
        with env_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

    # LIVE_URL の行を探して置き換え or 追加
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("YOUTUBE_URL="):
            lines[i] = f"YOUTUBE_URL={live_url}\n"
            updated = True
            break

    if not updated:
        lines.append(f"YOUTUBE_URL={live_url}\n")

    # 書き込み
    with env_file.open("w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"✅ .env に書き込み完了: {live_url}")


if __name__ == "__main__":
    live_url = get_live_url(API_KEY, CHANNEL_ID)
    if live_url:
        write_to_env(live_url)
    else:
        print("現在ライブ配信は行われていません。")
