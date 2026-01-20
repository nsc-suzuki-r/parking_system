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
    # HLS セグメント 403 エラーを回避するため複数フォーマットを試行

    # フォーマット選択の優先順序（高品質から低品質へ）
    format_options = [
        "best[ext=mp4]",  # 標準
        "best[vcodec=h264]",  # H.264推奨
        "worst",  # 最後の手段
    ]

    success = False
    last_error = None

    for format_opt in format_options:
        if success:
            break

        # yt-dlpでライブストリームの動画データを取得
        command = (
            f"yt-dlp --cookies cookies.txt --socket-timeout 10 "
            f'-o - -f "{format_opt}" {youtube_url} 2>&1 | '
            f"tee /tmp/yt-dlp-log.txt | "
            f'ffmpeg -y -loglevel error -i pipe:0 -frames:v 1 "{output_path}" 2>&1'
        )

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=False, timeout=60
            )

            # yt-dlpのログを確認（バイナリデータのため別途処理）
            if os.path.exists("/tmp/yt-dlp-log.txt"):
                try:
                    with open(
                        "/tmp/yt-dlp-log.txt", "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        log_content = f.read()

                    # 403エラー検出
                    if (
                        "403 Forbidden" in log_content
                        or "HTTP error 403" in log_content
                    ):
                        last_error = f"Format '{format_opt}': HTTP 403 Forbidden"
                        print(f"⚠️  {last_error}")
                        continue

                    # 他のエラー検出
                    if "ERROR" in log_content or "failed" in log_content.lower():
                        last_error = f"Format '{format_opt}': {log_content[:200]}"
                        print(f"⚠️  {last_error}")
                        continue
                except Exception:
                    pass

            # フレームファイル確認
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                print(f"✓ フレーム取得成功（フォーマット: {format_opt}）")
                success = True
                break
            else:
                last_error = (
                    f"Format '{format_opt}': Frame file not created or too small"
                )
        except subprocess.TimeoutExpired:
            last_error = f"Format '{format_opt}': Timeout"
            print(f"⚠️  {last_error}")
        except Exception as e:
            last_error = f"Format '{format_opt}': {str(e)}"
            print(f"⚠️  {last_error}")

    if not success:
        print("\n【エラー】すべてのフォーマットで失敗しました")
        print(f"最後のエラー: {last_error}")
        print("\n→ Cookie を更新してください:")
        print("  開発環境で: Cookie extension で cookies.txt をエクスポート")
        print("  本番環境で実行:")
        print("  cd /opt/visitory-parking-system/parking_system")
        print("  scp user@dev:/path/to/cookies.txt .")
        raise Exception(f"All format attempts failed: {last_error}")

    # ファイル確認
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"\n【エラー】フレーム取得に失敗しました")
        raise Exception("Frame extraction failed")

    print(f"✓ フレームが正常に保存されました: {output_path}")

    # 分割処理を実行
    split_segments = split_image(output_path, target_folder, models_and_outputs)
    print(f"✓ 分割完了: {split_segments}")

    # 分割結果を使って予測とAPI送信を実行
    prediction_results = run_predictions(split_segments)
    print(f"✓ 予測結果: {prediction_results}")

    return prediction_results


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
if __name__ == "__main__":
    try:
        capture_split_predict_and_send(
            youtube_url, output_folder, target_folder, models_and_outputs
        )
    except Exception as e:
        print(f"\n【致命的エラー】{e}")
        sys.exit(1)
