import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
import os
import sys

EMPTY = "1"
FULL = "6"
CROWDED = "5"


def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


# 画像の前処理
def preprocess_image(image_path, device):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    return image.to(device)


# 画像の予測
def predict(image_path, model, device):
    image = preprocess_image(image_path, device)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        if preds.item() == 0:
            return CROWDED
        elif preds.item() == 1:
            return EMPTY
        else:
            return FULL


# メイン処理
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py <image_path> <model_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"画像ファイル {image_path} が存在しません。")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"モデルファイル {model_path} が存在しません。")
        sys.exit(1)

    model, device = load_model(model_path)
    result = predict(image_path, model, device)
    print(result)
