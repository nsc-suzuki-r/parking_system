import requests
import json
import os
import subprocess
from utils.predict import load_model, predict


def run_predictions(segments):
    results = {}
    visitory_url = os.getenv("VISITORY_URL")
    visitory_headers = {
        "Authorization": os.getenv("VISITORY_AUTH"),
        "Content-Type": "application/json",
    }
    parking_lot_ids = {
        "takeda_a.jpg": os.getenv("PARKING_LOT_TAKEDA_A"),
        "takeda_b.jpg": os.getenv("PARKING_LOT_TAKEDA_B"),
        "takeda_c.jpg": os.getenv("PARKING_LOT_TAKEDA_C"),
        "takeda_d.jpg": os.getenv("PARKING_LOT_TAKEDA_D"),
        "rittai_p.jpg": os.getenv("PARKING_LOT_RITTAI_P"),
    }

    # モデルのキャッシュ
    model_cache = {}

    for segment_path, model_path, output_name in segments:
        # モデルのロード（キャッシュして再利用）
        if model_path not in model_cache:
            model, device = load_model(model_path)
            model_cache[model_path] = (model, device)
        else:
            model, device = model_cache[model_path]

        # 画像の予測を実行
        try:
            result = predict(segment_path, model, device)
            results[output_name] = result
        except Exception as e:
            raise Exception(f"Error running prediction for {output_name}: {e}")

        # 駐車場のIDが設定されている場合、センサーの状態を更新
        if parking_lot_ids.get(output_name) is not None:
            update_sensor_status(
                visitory_url,
                visitory_headers,
                parking_lot_ids[output_name],
                results[output_name],
            )

    return results


def update_sensor_status(visitory_url, headers, parking_lot_id, status):
    body = {
        "id": parking_lot_id,
        "value": status,
    }

    response = requests.post(visitory_url, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        raise Exception(
            f"Failed to update sensor: {response.status_code}, {response.text}"
        )
