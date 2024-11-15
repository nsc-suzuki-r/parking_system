from flask import request, jsonify
from utils.file_utils import clear_existing_files, save_file
from utils.image_processing import split_image
from utils.visistory_api import run_predictions
import os


# モデルファイル名と対応する出力名
models_and_outputs = {
    "models/parking_model_takeda_a.pth": "takeda_a.jpg",
    "models/parking_model_takeda_b.pth": "takeda_b.jpg",
    "models/parking_model_takeda_c.pth": "takeda_c.jpg",
    "models/parking_model_takeda_d.pth": "takeda_d.jpg",
    "models/parking_model_rittai_p.pth": "rittai_p.jpg",
    "models/parking_model_bottom.pth": "bottom.jpg",
}


def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            folder = os.getenv("UPLOAD_FOLDER")
            clear_existing_files(folder)
            filename, filepath = save_file(file, folder)
            segments = split_image(
                filepath, os.getenv("TARGET_FOLDER"), models_and_outputs
            )
            results = run_predictions(segments)
            return jsonify({"results": results}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File upload failed"}), 500
