from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    make_response,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import os
import subprocess
import glob
import uuid
from PIL import Image

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file:
            # inputディレクトリ内の既存の画像を削除
            for existing_file in glob.glob(
                os.path.join(app.config["UPLOAD_FOLDER"], "*")
            ):
                os.remove(existing_file)

            # ファイル名をUUIDに変換
            filename = f"{uuid.uuid4()}.png"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # 画像を6分割
            image = Image.open(filepath)
            width, height = image.size
            segment_height = height // 6 - 4

            # 分割した画像の保存
            for i, (model_path, output_name) in enumerate(models_and_outputs.items()):
                box = (0, i * segment_height, width, (i + 1) * segment_height)
                segment = image.crop(box)
                segment_path = os.path.join(app.config["TARGET_FOLDER"], output_name)
                segment.save(segment_path)

            # 各モデルで予測を実行
            results = {}
            print(f"{ models_and_outputs.items() }")

            for model_path, output_name in models_and_outputs.items():
                print(f"{ model_path }")
                print(f"{ output_name }")
                # modelの予測を行うスクリプトを実行
                result = subprocess.run(
                    [
                        "python3",
                        "predict.py",
                        os.path.join(app.config["TARGET_FOLDER"], output_name),
                        model_path,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )

                if result.returncode != 0:
                    return (
                        jsonify(
                            {
                                "error": f"Error running prediction script for {output_name}",
                                "details": result.stderr,
                            }
                        ),
                        500,
                    )

                results[output_name] = result.stdout.strip()

            return jsonify({"results": results}), 200

        return jsonify({"error": "File upload failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/target/<filename>")
def uploaded_file(filename):
    print(filename)
    return send_from_directory(app.config["TARGET_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
