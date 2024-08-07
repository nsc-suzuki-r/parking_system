# parking_system

## README

### 概要

このプロジェクトは、駐車場の状況を予測するために、画像をアップロードして機械学習モデルで処理するFlaskウェブアプリケーションです。アップロードされた画像をセグメントに分割し、各セグメントを事前に学習されたモデルで分類します。プロジェクトには以下の3つの主要なコンポーネントが含まれます：

- `app.py`：メインのFlaskアプリケーション。
- `predict.py`：学習済みモデルを使用して予測を行うスクリプト。
- `train.py`：モデルを学習するためのスクリプト。

### 前提条件

- Python 3.10.12
- Flask
- Torch
- torchvision
- PIL (Python Imaging Library)

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
.
├── app.py
├── predict.py
├── templates/
│   └── index.html
├── input/
│   └── 元の画像
├── target/
│   └── 分割された画像
├── models/
│   ├── parking_model_takeda_a.pth
│   ├── parking_model_takeda_b.pth
│   ├── parking_model_takeda_c.pth
│   ├── parking_model_takeda_d.pth
│   ├── parking_model_rittai_p.pth
│   └── parking_model_bottom.pth
├── dataset_bottom/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
├── dataset_rittai_p/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
├── dataset_takeda_a/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
├── dataset_takeda_b/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
├── dataset_takeda_c/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
├── dataset_takeda_d/
│   ├── train/
│   │   ├── full/
│   │   │   └── ... (full images)
│   │   ├── crowded/
│   │   │   └── ... (crowded images)
│   │   └── empty/
│   │       └── ... (empty images)
│   └── val/
│       ├── full/
│       │   └── ... (full images)
│       ├── crowded/
│       │   └── ... (crowded images)
│       └── empty/
│           └── ... (empty images)
```

### 使用方法

#### アプリケーションの実行

1. Flaskアプリケーションを開始します：

   ```bash
   python app.py
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
   python train.py
   ```

   これにより、モデルが学習され、指定されたパスに保存されます。

### 予測スクリプト

`predict.py` スクリプトは、Flaskアプリケーションによってセグメント化された画像に対して予測を行うために使用されます。事前に学習されたモデルをロードし、入力画像を処理します。

### ファイルの説明

#### `app.py`

- メインのFlaskアプリケーションファイル。
- ファイルのアップロード、画像のセグメント化、およびモデル予測を処理します。

#### `predict.py`

- モデルをロードし、画像を前処理する関数が含まれています。
- 入力画像セグメントに対して予測を実行します。

#### `train.py`

- モデルを学習するためのスクリプト。
- データセットをロードし、モデルを学習し、学習済みモデルを保存します。

### 注意事項

- `UPLOAD_FOLDER` および `TARGET_FOLDER` ディレクトリが存在すること、またはFlaskアプリケーションによって作成されることを確認してください。
- スクリプト内のパスを必要に応じてディレクトリ構造に合わせて更新してください。
