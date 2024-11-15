import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

# データセットの前処理
data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

# データセット名と出力ファイル名のリスト
datasets_and_models = [
    ("dataset/takeda_a", "models/parking_model_takeda_a.pth"),
    ("dataset/takeda_b", "models/parking_model_takeda_b.pth"),
    ("dataset/takeda_c", "models/parking_model_takeda_c.pth"),
    ("dataset/takeda_d", "models/parking_model_takeda_d.pth"),
    ("dataset/rittai_p", "models/parking_model_rittai_p.pth"),
    ("dataset/bottom", "models/parking_model_bottom.pth"),
]


# トレーニング関数
def train_model(model, criterion, optimizer, num_epochs, dataloaders, dataset_sizes):
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

    return model


# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 各データセットに対してトレーニングを実行
for data_dir, model_path in datasets_and_models:
    print(f"Training on dataset: {data_dir}")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    # モデルのロードと微調整
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # モデルのトレーニング
    model = train_model(
        model,
        criterion,
        optimizer,
        num_epochs=25,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
    )

    # モデルの保存
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
