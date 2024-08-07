from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
)
import os
import subprocess
import glob
import uuid
from PIL import Image
import requests
import json
from dotenv import load_dotenv

# .envファイルの内容を読み込む
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "input"
app.config["TARGET_FOLDER"] = "target"

# ファイルが保存されるディレクトリが存在しない場合は作成
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["TARGET_FOLDER"], exist_ok=True)

# モデルファイル名と対応する出力名
models_and_outputs = {
    "parking_model_takeda_a.pth": "takeda_a.jpg",
    "parking_model_takeda_b.pth": "takeda_b.jpg",
    "parking_model_takeda_c.pth": "takeda_c.jpg",
    "parking_model_takeda_d.pth": "takeda_d.jpg",
    "parking_model_rittai_p.pth": "rittai_p.jpg",
    "parking_model_bottom.pth": "bottom.jpg",
}

# Visitory APIの設定
visitory_url = os.getenv("VISITORY_URL")
visitory_headers = {
    "Authorization": os.getenv("VISITORY_AUTH"),
    "Content-Type": "application/json",
}

# 駐車場のIDマッピング
parking_lot_ids = {
    "takeda_a.jpg": os.getenv("PARKING_LOT_TAKEDA_A"),
    "takeda_b.jpg": os.getenv("PARKING_LOT_TAKEDA_B"),
    "takeda_c.jpg": os.getenv("PARKING_LOT_TAKEDA_C"),
    "takeda_d.jpg": os.getenv("PARKING_LOT_TAKEDA_D"),
    "rittai_p.jpg": os.getenv("PARKING_LOT_RITTAI_P"),
    "bottom.jpg": os.getenv("PARKING_LOT_BOTTOM"),
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            clear_existing_files(app.config["UPLOAD_FOLDER"])
            filename, filepath = save_file(file, app.config["UPLOAD_FOLDER"])
            segments = split_image(
                filepath, app.config["TARGET_FOLDER"], models_and_outputs
            )
            results = run_predictions(
                segments,
                models_and_outputs,
                visitory_url,
                visitory_headers,
                parking_lot_ids,
            )
            return jsonify({"results": results}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File upload failed"}), 500


@app.route("/target/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["TARGET_FOLDER"], filename)


def clear_existing_files(folder):
    for existing_file in glob.glob(os.path.join(folder, "*")):
        os.remove(existing_file)


def save_file(file, folder):
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(folder, filename)
    file.save(filepath)
    return filename, filepath


def split_image(filepath, target_folder, models_and_outputs):
    image = Image.open(filepath)
    width, height = image.size
    segment_height = height // 6 - 4
    segments = []

    for i, (model_path, output_name) in enumerate(models_and_outputs.items()):
        box = (0, i * segment_height, width, (i + 1) * segment_height)
        segment = image.crop(box)
        segment_path = os.path.join(target_folder, output_name)
        segment.save(segment_path)
        segments.append((segment_path, model_path, output_name))

    return segments


def run_predictions(
    segments, models_and_outputs, visitory_url, visitory_headers, parking_lot_ids
):
    results = {}

    for segment_path, model_path, output_name in segments:
        result = subprocess.run(
            ["python3", "predict.py", segment_path, model_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise Exception(
                f"Error running prediction script for {output_name}: {result.stderr}"
            )

        results[output_name] = result.stdout.strip()

        if parking_lot_ids[output_name] is not None:
            update_sensor_status(
                visitory_url,
                visitory_headers,
                parking_lot_ids[output_name],
                results[output_name],
            )

    return results


def update_sensor_status(visitory_url, visitory_headers, parking_lot_id, status):
    body = {
        "gatewayid": parking_lot_id,
        "value": status,
        "type": "parking-lot",
    }

    response = requests.post(
        visitory_url, headers=visitory_headers, data=json.dumps(body)
    )

    if response.status_code != 200:
        raise Exception(
            f"Failed to update sensor status: {response.status_code}, {response.text}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
