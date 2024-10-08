import os
import clip
import torch
import joblib  # joblib을 사용해 모델을 저장하고 불러옵니다
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


# 사용자 데이터셋 경로 지정
data_dir = r""  


# 데이터셋 변환 정의 (CLIP 모델의 예상 입력 크기 및 정규화에 맞춰야 함)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지를 224x224로 크기 조정 (CLIP ViT-B/32 입력 크기)
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))  # 정규화 (CLIP의 예상 정규화)
])


# ImageFolder를 사용하여 사용자 데이터셋 로드
train = ImageFolder(os.path.join(data_dir, 'train'), transform=preprocess)
test = ImageFolder(os.path.join(data_dir, 'test'), transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


# 이미지 특징 계산
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)


# 로지스틱 회귀 수행
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)


# 훈련된 모델을 joblib을 사용하여 저장
joblib.dump(classifier, 'model.pkl')  # 모델을 pkl 파일로 저장


print("모델 저장 완료!")


# 로지스틱 회귀 분류기를 사용하여 평가
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"정확도 = {accuracy:.3f}")
