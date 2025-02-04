import os

import time

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split


from PIL import Image

from tqdm import tqdm

from datetime import datetime

from sklearn.model_selection import StratifiedKFold

from torch.optim.lr_scheduler import StepLR


matplotlib.use('Agg')




class AnomalyDataset(Dataset):

    """MVTec Anomaly Detection Dataset을 PyTorch Dataset으로 구현."""


    def __init__(self, image_paths, labels, target_size=(256, 256), is_train=True):

        self.image_paths = image_paths

        self.labels = labels

        self.target_size = target_size

        self.is_train = is_train


        # 데이터 증강을 위한 변환 정의

        if self.is_train:

            self.transform = transforms.Compose([

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomVerticalFlip(p=0.5),

                transforms.RandomRotation(degrees=2),

                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05),

                transforms.Resize(self.target_size),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])

        else:

            # 검증/테스트 데이터의 기본 변환 정의

            self.transform = transforms.Compose([

                transforms.Resize(self.target_size),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ])


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')


        # 데이터 증강 및 변환 적용

        img = self.transform(img)


        return img, torch.tensor(label, dtype=torch.long)



def load_images_from_folder(folder_path):

    """특정 폴더에서 모든 PNG 이미지 경로와 레이블을 반환."""

    image_paths = []

    labels = []


    for defect_type in os.listdir(folder_path):

        defect_path = os.path.join(folder_path, defect_type)

        if os.path.isdir(defect_path):

            for root, _, files in os.walk(defect_path):

                for file in files:

                    if file.lower().endswith('.png'):

                        image_paths.append(os.path.join(root, file))

                        labels.append(.0 if defect_type.lower() == 'good' else 1.0)


    return image_paths, labels



def prepare_data(base_dir, folder_names):

    """전체 데이터셋에서 이미지 경로와 라벨을 수집."""

    all_image_paths = []

    all_labels = []


    for folder in folder_names:

        for subfolder in ['train', 'test']:

            folder_path = os.path.join(base_dir, folder, subfolder)

            image_paths, labels = load_images_from_folder(folder_path)

            all_image_paths.extend(image_paths)

            all_labels.extend(labels)


    return all_image_paths, all_labels



# CNN 기반 DAE 모델 정의

class DenoisingAutoencoder(nn.Module):

    def __init__(self):

        super(DenoisingAutoencoder, self).__init__()


        # 인코더

        self.encoder = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(64),  # Batch Normalization 추가

            nn.ReLU(),

            torch.nn.Dropout(p=0.1),

            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(16),  # Batch Normalization 추가

            nn.ReLU(),

            torch.nn.Dropout(p=0.1),

            nn.Conv2d(16, 4, kernel_size=4, stride=16, padding=0),

            nn.BatchNorm2d(4),  # Batch Normalization 추가

            nn.ReLU(),

        )


        self.linear_1 = nn.Linear(4*4*4, 1) # channel*channel*RGB 

        #self.linear_2 = nn.Linear(16*16, 1)

        self.flatten = nn.Flatten()


        # 디코더

        # self.decoder = nn.Sequential(

        #     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),

        #     nn.BatchNorm2d(128),  # Batch Normalization 추가

        #     nn.ReLU(),

        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),

        #     nn.BatchNorm2d(64),  # Batch Normalization 추가

        #     nn.ReLU(),

        #     nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),

        #     nn.Sigmoid()

        # )


    def forward(self, x):

        x = self.encoder(x)

        x = self.flatten(x)

        x = self.linear_1(x)

        #x = self.linear_2(x)

        x = x.squeeze()

        # x = self.decoder(x)

        return x


"""print("PyTorch에서 지원하는 CUDA 버전:", torch.version.cuda)


if torch.cuda.is_available():

    print("CUDA is available! GPU:", torch.cuda.get_device_name(0))

else:

    print("CUDA is not available.")


print("CPU 코어 수 (os.cpu_count):", os.cpu_count())"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoisingAutoencoder().to(device)


# 학습 함수 (K-Fold Cross Validation 적용)

def train_dae_with_kfold(model, splits, num_epochs=10, save_path="dae_model.pth"):

    criterion = nn.MSELoss()  # 손실 함수

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # 5 에포크마다 학습률 감소

    scaler =  torch.cuda.amp.GradScaler()  # 혼합 정밀도 학습을 위한 스케일러 추가


    # 손실 및 정확도 기록을 위한 리스트

    epoch_losses = []

    val_losses = []

    batch_losses = []

    train_accuracies = []

    val_accuracies = []

    val_mse_hist = []


    for epoch in range(num_epochs):

        model.train()

        epoch_loss = 0

        correct_train = 0

        total_train = 0

        sigmoid = nn.Sigmoid()

        bce_loss = nn.BCEWithLogitsLoss()

        for fold, (train_dataset, val_dataset) in enumerate(splits):

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True,prefetch_factor=128)

            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True,prefetch_factor=128)



            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}, Fold {fold + 1}") as pbar:

                for imgs, labels in train_loader:

                    imgs = imgs.to(device)

                    labels = labels.to(device).float()

                    noisy_imgs = imgs + 0.2 * torch.randn_like(imgs)

                    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)


                    optimizer.zero_grad()


                    # 혼합 정밀도 학습 적용

                    with torch.amp.autocast('cuda'):

                        outputs = model(noisy_imgs)

                        loss = bce_loss(outputs, labels)


                    # 역전파

                    scaler.scale(loss).backward()

                    scaler.step(optimizer)

                    scaler.update()


                    epoch_loss += loss.item()

                    batch_losses.append(loss.item())  # 매 배치 손실 값을 batch_losses에 추가합니다.


                    # 정확도 계산을 위한 처리

                    #mse = nn.functional.mse_loss(outputs, imgs, reduction='none').mean([1, 2, 3])

                    outputs = sigmoid(outputs) # 모델 결과를 확률로 나타냄. 예) 불량품일 확률 0.7..

                    predictions = (outputs > 0.4).float() # 기준 확률 0.5를 넘을 경우 불량품 1 넘지 못하면 정상제품 0

                    correct_train += (predictions == labels).sum().item()

                    total_train += labels.size(0)


                    # tqdm에 표시할 정보 업데이트

                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

                    pbar.update(1)

                    

            model.eval()

            val_loss = 0

            correct_val = 0

            total_val = 0

            with tqdm(total=len(val_loader), desc=f"Validation ... Epoch {epoch + 1}, Fold {fold + 1}") as pbar:

                with torch.no_grad():

                    for val_imgs, val_labels in val_loader:

                        val_imgs, val_labels = val_imgs.to(device), val_labels.to(device).float()


                        with torch.amp.autocast('cuda'):

                            outputs = model(val_imgs)

                            loss = bce_loss(outputs, val_labels)

                            

                        val_loss += loss.item()


                        outputs = sigmoid(outputs)  # 모델 결과를 확률로 나타냄. 예) 불량품일 확률 0.7..

                        # print(f'outputs: {outputs}')

                        predictions = (outputs > 0.4).float()  # 기준 확률 0.5를 넘을 경우 불량품 1 넘지 못하면 정상제품 0

                        correct_val += (predictions == val_labels).sum().item()

                        total_val += val_labels.size(0)

                        

                        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

                        pbar.update(1)

                        

        # 에포크 손실 계산

        average_epoch_loss = epoch_loss / len(train_loader)

        epoch_losses.append(average_epoch_loss)

        train_accuracy = correct_train / total_train

        train_accuracies.append(train_accuracy)


        # 검증 손실 및 정확도 계산

        


        average_val_loss = val_loss / len(val_loader)

        val_losses.append(average_val_loss)

        val_accuracy = correct_val / total_val

        val_accuracies.append(val_accuracy)


        print(f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


        scheduler.step()  # 학습률 스케줄러 적용


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")


        # 최적의 모델 저장

        if epoch == 0 or average_epoch_loss < min(epoch_losses):

            torch.save(model.state_dict(), save_path)

            print(f"New best model saved at '{save_path}' with Loss: {average_epoch_loss:.4f}")


    return epoch_losses, val_losses, train_accuracies, val_accuracies, batch_losses, val_mse_hist




# 결과 시각화 함수

def show_results(model, dataloader, num_images=5):

    model.eval()

    imgs, _ = next(iter(dataloader))

    noisy_imgs = imgs + 0.2 * torch.randn_like(imgs)

    noisy_imgs = torch.clip(noisy_imgs, 0., 1.).to(device)

    imgs = imgs.to(device)


    with torch.no_grad():

        outputs = model(noisy_imgs)


    plt.figure(figsize=(15, 5))

    for i in range(num_images):

        ax = plt.subplot(3, num_images, i + 1)

        plt.imshow(imgs[i].cpu().permute(1, 2, 0))

        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + num_images)

        plt.imshow(noisy_imgs[i].cpu().permute(1, 2, 0))

        plt.axis("off")


        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)

        plt.imshow(outputs[i].cpu().permute(1, 2, 0))

        plt.axis("off")


    plt.show()



# 양불 판정 함수

def classify_anomalies(model, dataloader, threshold):

    model.eval()

    # criterion = nn.MSELoss(reduction='mean')


    sigmoid = nn.Sigmoid()

    total = len(dataloader.dataset)

    correct = 0


    print("\n양불 판정 결과:")

    with torch.no_grad():

        for imgs, labels in dataloader:

            labels = labels.to(device).float()


            # 노이즈가 제거된 복원 이미지 예측

            outputs = model(imgs)


            # 복원 오차 계산

            outputs = sigmoid(outputs)  # 모델 결과를 확률로 나타냄. 예) 불량품일 확률 0.7..

            predictions = (outputs > 0.1).long()  # 기준 확률 0.5를 넘을 경우 불량품 1 넘지 못하면 정상제품 0

            actual = labels.item()


            # 예측이 실제 레이블과 일치하는 경우

            if predictions == actual:

                correct += 1


            # 결과 출력

            print(f"실제 라벨: {'불량' if actual == 1 else '양품'}, 예측 라벨: {'불량' if predictions == 1 else '양품'}")


    accuracy = correct / total * 100

    print(f"\n전체 정확도: {accuracy:.2f}%")


def split_data_stratified_kfold(dataset, k=5):

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    splits = []

    # `dataset.image_paths`와 `dataset.labels`를 사용하여 분할

    for train_idx, val_idx in skf.split(dataset.image_paths, dataset.labels):

        train_subset = torch.utils.data.Subset(dataset, train_idx)

        val_subset = torch.utils.data.Subset(dataset, val_idx)

        splits.append((train_subset, val_subset))

    return splits


def plot_training_results(epoch_losses, val_losses, train_accuracies, val_accuracies, batch_losses, val_mse_hist):

    plt.figure(figsize=(18, 15))


    # 1. 에포크 손실 그래프 (Training Loss per Epoch)

    plt.subplot(2, 3, 1)

    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss per Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('Training Loss per Epoch')

    plt.legend()

    plt.grid()


    # 2. 검증 손실 그래프 (Validation Loss per Epoch)

    plt.subplot(2, 3, 2)

    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', color='orange', label='Validation Loss per Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('Validation Loss per Epoch')

    plt.legend()

    plt.grid()


    # 3. 학습 정확도 그래프 (Training Accuracy per Epoch)

    plt.subplot(2, 3, 3)

    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='green',

             label='Training Accuracy per Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('Training Accuracy per Epoch')

    plt.legend()

    plt.grid()


    # 4. 검증 정확도 그래프 (Validation Accuracy per Epoch)

    plt.subplot(2, 3, 4)

    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', color='red',

             label='Validation Accuracy per Epoch')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('Validation Accuracy per Epoch')

    plt.legend()

    plt.grid()


    # 5. 배치 손실 그래프 (Batch-wise Training Loss)

    plt.subplot(2, 3, 5)

    plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker='.', label='Training Loss per Batch', alpha=0.6)

    plt.xlabel('Batch')

    plt.ylabel('Loss')

    plt.title('Training Loss per Batch')

    plt.legend()

    plt.grid()


    # 6. 검증 MSE 히스토그램 (Validation MSE Histogram)

    plt.subplot(2, 3, 6)

    plt.hist(val_mse_hist, bins=50, color='purple', alpha=0.7)

    plt.xlabel('MSE')

    plt.ylabel('Frequency')

    plt.title('Validation MSE Histogram')

    plt.grid()


    plt.tight_layout()

    plt.savefig("training_results.png")

    print("손실 그래프가 'training_results.png'에 저장되었습니다.")



def log_message(message):

    """시간을 포함한 로그 메시지를 출력합니다."""

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[{current_time}] {message}")


# 메인 실행 코드

if __name__ == "__main__":

    base_dir = r'/home/user/Desktop/img_data'

    folder_names = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",

"tile", "toothbrush" ]  # "Cable", "Capsule", 등 다른 클래스도 포함 가능


    log_message("데이터 로드 및 전처리")

    image_paths, labels = prepare_data(base_dir, folder_names)

    dataset = AnomalyDataset(image_paths, labels, target_size=(256, 256))


    log_message("K-Fold 데이터 분할")

    k = 2  # 폴드 수 설정

    splits = split_data_stratified_kfold(dataset, k=k)


    # 모델 정의

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenoisingAutoencoder().to(device)


    # 학습 실행 (K-Fold 적용)

    num_epochs = 100

    model_save_path = "/home/user/Desktop/img_data/dae_model.pth"

    epoch_losses, val_losses, train_accuracies, val_accuracies, batch_losses, val_mse_hist = train_dae_with_kfold(

        model, splits, num_epochs=num_epochs, save_path=model_save_path

    )


    # 결과 시각화

    plot_training_results(epoch_losses, val_losses, train_accuracies, val_accuracies, batch_losses, val_mse_hist)


    log_message("저장된 모델 로드 및 테스트")

    model.load_state_dict(torch.load(model_save_path, map_location=device))

    model.to(device)

    print("저장된 모델을 불러왔습니다.")


    # 테스트 데이터 준비 및 양불 판정

    test_dataset_size = int(0.15 * len(dataset))  # 테스트 데이터 비율 설정

    _, _, test_dataset = random_split(dataset, [len(dataset) - 2 * test_dataset_size, test_dataset_size, test_dataset_size])


    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    classify_anomalies(model, test_loader, threshold=0.01)
