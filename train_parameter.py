import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from PIL import Image

# 데이터셋 정의
class CameraPoseDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_paths = []
        self.intrinsics = []
        self.extrinsics = []
        self.transform = transform

        # 데이터 경로에서 각 샘플의 이미지 및 파라미터 파일 경로 수집
        for folder_name in sorted(os.listdir(os.path.join(data_path, 'img'))):
            if folder_name == '0201':
                print("break load img")
                break
            img_folder = os.path.join(data_path, 'img', folder_name)
            parm_folder = os.path.join(data_path, 'parm', folder_name)

            for i in range(16):
                img_path = os.path.join(img_folder, f"{i}.jpg")
                intrinsic_path = os.path.join(parm_folder, f"{i}_intrinsic.npy")
                extrinsic_path = os.path.join(parm_folder, f"{i}_extrinsic.npy")
                
                # 각 파일 경로 리스트에 추가
                self.image_paths.append(img_path)
                self.intrinsics.append(intrinsic_path)
                self.extrinsics.append(extrinsic_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        intrinsic = np.load(self.intrinsics[idx])
        extrinsic = np.load(self.extrinsics[idx])
        
        if self.transform:
            image = self.transform(image)
        
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)
        
        return image, intrinsic, extrinsic

# ResNet 기반 모델 정의
class CameraPoseEstimationModel(nn.Module):
    def __init__(self):
        super(CameraPoseEstimationModel, self).__init__()
        
        # Pre-trained ResNet50 모델 사용
        resnet = models.resnet50(pretrained=True)
        
        # Feature extractor로 사용 (FC layer 제거)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Fully connected layers for intrinsic and extrinsic regression
        self.fc_intrinsic = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 9)  # 3x3 intrinsic matrix
        )
        
        self.fc_extrinsic = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # 3x4 extrinsic matrix
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten

        intrinsic = self.fc_intrinsic(x).view(-1, 3, 3)
        extrinsic = self.fc_extrinsic(x).view(-1, 3, 4)
        
        return intrinsic, extrinsic


# 테스트 함수
def test_model(model, data_path, save_dir='parameter_test_output'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
#----
    # 데이터 경로에서 각 샘플의 이미지 및 파라미터 파일 경로 수집
    for folder_name in sorted(os.listdir(os.path.join(data_path, 'img'))):

        img_folder = os.path.join(data_path, 'img', folder_name)
        parm_folder = os.path.join(save_dir, 'parm', folder_name)
        os.makedirs(parm_folder, exist_ok=True)

        for i in range(2):
            img_path = os.path.join(img_folder, f"{i}.jpg")
            intrinsic_save_path = os.path.join(parm_folder, f"{i}_intrinsic.npy")
            extrinsic_save_path = os.path.join(parm_folder, f"{i}_extrinsic.npy")

            # 테스트 이미지 불러오기 및 전처리
            # test_image_path = test_dataset.image_paths[i]
            test_image = Image.open(img_path).convert("RGB")
            test_image = transform(test_image).unsqueeze(0).cuda()  # 배치 차원 추가 및 CUDA로 이동

            with torch.no_grad():
                # 모델 예측
                pred_intrinsic, pred_extrinsic = model(test_image)

                # 결과 저장
                intrinsic_np = pred_intrinsic.squeeze().cpu().numpy()
                extrinsic_np = pred_extrinsic.squeeze().cpu().numpy()

                np.save(intrinsic_save_path, intrinsic_np)
                np.save(extrinsic_save_path, extrinsic_np)

                print("Predicted Intrinsic Matrix saved at:", intrinsic_save_path)
                print("Predicted Extrinsic Matrix saved at:", extrinsic_save_path)


if __name__ == "__main__":
    
    torch.cuda.empty_cache()

    # 하이퍼파라미터
    batch_size = 4
    num_epochs = 200
    learning_rate = 0.001

    # 데이터 경로 설정
    data_path = "/home/sophie/Desktop/minseok/gps_dataset/real_data"
    model_save_path = '/home/sophie/Desktop/minseok/GPS-Gaussian/result_parameter/result.pth'

    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    model = CameraPoseEstimationModel().cuda()  # 모델을 CUDA로 이동

    # testing, generate parameter with AI
    test = 1
    if test :
        print("Test Mode")
        test_path = "/home/sophie/Desktop/minseok/gps_dataset/test_data"
        
        #model load
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        # 테스트 함수 호출
        test_model(model, test_path)

    else:
        print("Train Mode")
        # 데이터셋 및 데이터로더 준비
        dataset = CameraPoseDataset(data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 모델 초기화 및 학습 설정
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # 학습 루프
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, (images, intrinsic_targets, extrinsic_targets) in enumerate(dataloader):
                # 데이터를 CUDA로 이동
                images = images.cuda()
                intrinsic_targets = intrinsic_targets.cuda()
                extrinsic_targets = extrinsic_targets.cuda()

                # Forward pass
                intrinsic_preds, extrinsic_preds = model(images)
                
                # Loss 계산
                intrinsic_loss = criterion(intrinsic_preds, intrinsic_targets)
                extrinsic_loss = criterion(extrinsic_preds, extrinsic_targets)
                loss = intrinsic_loss + extrinsic_loss
                
                # Backward pass 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 1:
                    print(f"Iteration [{i}/{len(dataloader)}], Loss: {running_loss / i:.4f}")
            scheduler.step()
            print(f"**Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}\n")
            if epoch%50 == 0:
                # 학습된 모델 저장
                torch.save(model.state_dict(), model_save_path)
                print("모델 저장 완료:", model_save_path)

        # 학습된 모델 저장
        torch.save(model.state_dict(), model_save_path)
        print("모델 저장 완료:", model_save_path)

        test_path = "/home/sophie/Desktop/minseok/gps_dataset/test_data"
        dataset_test = CameraPoseDataset(data_path, transform=transform)
        
        # 테스트 함수 호출
        test_image_path = dataset_test.image_paths[0]  # 임의로 첫 번째 이미지를 선택하여 테스트
        test_model(model, test_image_path)
