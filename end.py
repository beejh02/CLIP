import os
import clip
import torch
import joblib  # 저장된 모델을 불러오기 위한 joblib
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# 1. CLIP 모델 및 전처리 불러오기
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# 2. Logistic Regression 모델 불러오기
classifier = joblib.load('logistic_regression_model.pkl')  # 저장된 Logistic Regression 모델을 불러옵니다

# 3. 단일 이미지 불러오기
image_path = ""  # 분류할 이미지 파일의 경로 지정
image = Image.open(image_path)

# 4. 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP 모델에 맞게 크기 조정
    transforms.ToTensor(),  # 텐서로 변환
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))  # 정규화
])
image_tensor = preprocess(image).unsqueeze(0).to(device)  # 이미지에 배치 차원 추가 후 GPU/CPU로 전송

# 5. CLIP 모델을 사용해 이미지 특징 추출
with torch.no_grad():
    image_features = model.encode_image(image_tensor).cpu().numpy()

# 6. Logistic Regression 모델을 사용하여 예측
prediction = classifier.predict(image_features)

# 7. 분류 결과 출력
# ImageFolder 사용 시, 카테고리 이름은 dataset.classes에 저장되므로 이를 불러와야 합니다.
# 예시로 train 데이터셋에서 카테고리 목록을 가져옴
data_dir = r""
train = ImageFolder(os.path.join(data_dir, 'train'), transform=preprocess)
categories = train.classes  # ImageFolder로부터 클래스 이름 불러오기

# 예측된 카테고리 출력
predicted_category = categories[prediction[0]]  # 예측된 레이블로 카테고리 이름 가져오기
print(f"해당 이미지는 '{predicted_category}' 카테고리로 분류되었습니다.")
