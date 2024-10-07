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

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Define the directory containing your custom dataset
data_dir = r""  # 사용자 데이터셋 경로 지정

# Define the dataset transformations (must match CLIP model's expected input size and normalization)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (CLIP ViT-B/32 input size)
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                        (0.26862954, 0.26130258, 0.27577711))  # Normalize (CLIP's expected normalization)
])

# Load the custom dataset using ImageFolder
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

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Save the trained model using joblib
joblib.dump(classifier, 'logistic_regression_model.pkl')  # 모델을 pkl 파일로 저장

print("모델이 성공적으로 저장되었습니다.")

# Optional: 나중에 저장된 모델을 불러오는 코드
# classifier = joblib.load('logistic_regression_model.pkl')
# print("저장된 모델을 성공적으로 불러왔습니다.")

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
