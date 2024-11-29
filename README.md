# parking_system

## README

### 概要

このプロジェクトは、駐車場の状況を予測するためのFlaskベースのウェブアプリケーションです。ユーザーが画像をアップロードすると、事前に学習された機械学習モデルが画像を解析し、駐車場の空き状況を分類します。また、大量の画像データを効率的に処理するバッチ処理機能も提供しています。

#### 主な機能

1. ウェブアプリケーションを通じた画像のアップロードと即時予測。
2. バッチ処理スクリプトによる自動解析と結果出力。
3. 学習済みモデルの使用およびカスタムデータでの再学習機能。

- `app.py`：メインのFlaskアプリケーション。
- `predict.py`：学習済みモデルを使用して予測を行うスクリプト。
- `train.py`：モデルを学習するためのスクリプト。
- `capture_split.py`: 画像収集スクリプト。
- `capture_split_predict_and_send.py`: API送信スクリプト。

### 前提条件

- streamlink==5.6.1
- ffmpeg-python==0.2.0
- Flask==2.0.1
- torch==1.9.0
- torchvision==0.10.0
- Pillow==8.2.0
- python-dotenv==0.19.2
- yt-dlp==2024.10.22

### インストール

1. リポジトリをクローンします：

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. 必要なパッケージをインストールします：

   ```bash
   pip install -r requirements.txt
   ```

### ディレクトリ構成

```plaintext
├── README.md
├── cookies.txt
├── cookies.txt.sample
├── data
│   ├── target
│   │   ├── bottom.jpg
│   │   ├── rittai_p.jpg
│   │   ├── takeda_a.jpg
│   │   ├── takeda_b.jpg
│   │   ├── takeda_c.jpg
│   │   └── takeda_d.jpg
│   ├── train
│   │   ├── processed
│   │   │   └── 2024
│   │   ├── raw
│   │   │   └── 2024
│   │   └── split
│   │       ├── bottom
│   │       ├── rittai_p
│   │       ├── takeda_a
│   │       ├── takeda_b
│   │       ├── takeda_c
│   │       └── takeda_d
│   └── upload
│       └── frame_172609.jpg
├── dataset
│   ├── bottom
│   │   ├── train
│   │   │   ├── crowded
│   │   │   ├── empty
│   │   │   └── full
│   │   └── val
│   │       ├── crowded
│   │       ├── empty
│   │       └── full
│   ├── rittai_p
│   │   ├── train
│   │   │   ├── crowded
│   │   │   ├── empty
│   │   │   └── full
│   │   └── val
│   │       ├── crowded
│   │       ├── empty
│   │       └── full
│   ├── takeda_a
│   │   ├── train
│   │   │   ├── crowded
│   │   │   ├── empty
│   │   │   └── full
│   │   └── val
│   │       ├── crowded
│   │       ├── empty
│   │       └── full
│   ├── takeda_b
│   │   ├── train
│   │   │   ├── crowded
│   │   │   ├── empty
│   │   │   └── full
│   │   └── val
│   │       ├── crowded
│   │       ├── empty
│   │       └── full
│   ├── takeda_c
│   │   ├── train
│   │   │   ├── crowded
│   │   │   ├── empty
│   │   │   └── full
│   │   └── val
│   │       ├── crowded
│   │       ├── empty
│   │       └── full
│   └── takeda_d
│       ├── train
│       │   ├── crowded
│       │   ├── empty
│       │   └── full
│       └── val
│           ├── crowded
│           ├── empty
│           └── full
├── models
│   ├── parking_model_bottom.pth
│   ├── parking_model_rittai_p.pth
│   ├── parking_model_takeda_a.pth
│   ├── parking_model_takeda_b.pth
│   ├── parking_model_takeda_c.pth
│   └── parking_model_takeda_d.pth
├── requirements.txt
└── src
    ├── app.py
    ├── batch
    │   ├── capture_split.py
    │   ├── capture_split_predict_and_send.py
    │   ├── split.py
    │   ├── streamlink.py
    │   └── train.py
    ├── routes
    │   ├── __pycache__
    │   │   ├── classify.cpython-310.pyc
    │   │   ├── index.cpython-310.pyc
    │   │   └── upload.cpython-310.pyc
    │   ├── index.py
    │   └── upload.py
    ├── templates
    │   └── index.html
    └── utils
        ├── __init__.py
        ├── __pycache__
        │   ├── __init__.cpython-310.pyc
        │   ├── file.cpython-310.pyc
        │   ├── file_utils.cpython-310.pyc
        │   ├── image.cpython-310.pyc
        │   ├── image_processing.cpython-310.pyc
        │   ├── predict.cpython-310.pyc
        │   └── visistory_api.cpython-310.pyc
        ├── file.py
        ├── image.py
        ├── predict.py
        └── visistory_api.py

```

### 使用方法

#### アプリケーションの実行

1. Flaskアプリケーションを開始します：

   ```bash
   python3 src/app.py --host=0.0.0.0 --port=5001
   ```

   アプリケーションは `http://0.0.0.0:5001` で利用可能です。

#### 画像のアップロード

1. ウェブブラウザで `http://0.0.0.0:5001` にアクセスします。
2. アップロードフォームを使用して画像をアップロードします。
3. アプリケーションは画像をセグメントに分割し、学習済みモデルを使用して各セグメントを分類します。結果はウェブページに表示されます。

### モデルの学習

モデルを学習するには、`train.py` スクリプトを使用します。データセットは `train.py` 内の `datasets_and_models` リストで指定されたディレクトリに配置する必要があります。

1. 学習スクリプトを実行します：

   ```bash
   python src/batch/train.py
   ```

   これにより、モデルが学習され、指定されたパスに保存されます。

### 予測スクリプト

`predict.py` スクリプトは、Flaskアプリケーションによってセグメント化された画像に対して予測を行うために使用されます。事前に学習されたモデルをロードし、入力画像を処理します。

1. 予測スクリプトを実行します：

   ```bash
   python src/batch/predict.py
   ```

   これにより、画像が予測され、結果が表示されます。

### 画像収集スクリプト

`capture_split.py` スクリプトは、YouTubeライブ動画から1フレームを取得し、指定のセグメントに分割して保存する機能を提供します。また、取得した元のフレームは「processed」ディレクトリに移動されます。主な用途は駐車場の画像解析におけるデータ準備です。
以下に、このスクリプトの各セクションや機能の説明をまとめます。

1. **.env ファイルの設定**
   `.env` ファイルを作成し、以下の内容を記載します。

   ```dotenv
   YOUTUBE_URL=<YouTubeライブ動画のURL>
   ```

2. **.cookies.txt ファイルの設定**
   `.cookies.txt` ファイルを作成します。
   - 作成方法
      - <https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies>
   - youtubeのcookieの有効期限は1年。
   - 1年後に再取得する必要がある。

3. **スクリプト実行**
   以下のコマンドでスクリプトを実行します。

   ```bash
   python src/batch/capture_split.py
   ```

### API送信スクリプト

`capture_split_predict_and_send.py` スクリプトは、YouTubeライブから1フレームを取得し、それを分割、機械学習モデルで予測し、結果をAPIに送信する一連のプロセスを自動化します。

1. **.env ファイルの設定**
   `.env` ファイルを作成し、以下の内容を記載します。

   ```dotenv
   YOUTUBE_URL=<YouTubeライブ動画のURL>
   ```

2. **.cookies.txt ファイルの設定**
   `.cookies.txt` ファイルを作成します。
   - 作成方法
      - <https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies>
   - youtubeのcookieの有効期限は1年。
   - 1年後に再取得する必要がある。

3. **スクリプト実行**
   以下のコマンドでスクリプトを実行します。

   ```bash
   python src/batch/capture_split_predict_and_send.py
   ```
