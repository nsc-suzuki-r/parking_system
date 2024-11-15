from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# ルートのインポート
from routes.index import index
from routes.upload import upload_file

app.add_url_rule("/", "index", index)
app.add_url_rule("/upload", "upload_file", upload_file, methods=["POST"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
