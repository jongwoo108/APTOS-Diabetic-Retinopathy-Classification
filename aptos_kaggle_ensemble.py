import os
for d, _, f in os.walk('/kaggle/input'):
    for name in f:
        if name.endswith('.pth'):
            print(os.path.join(d, name))

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

# ============================================================
# 1. 모델
# ============================================================
class APTOSModel(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes - 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# ============================================================
# 2. 전처리
# ============================================================
def crop_image_from_gray(img, tol=7):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if not np.any(mask):
            return img
        return img[np.ix_(np.any(mask, axis=1), np.any(mask, axis=0))]
    return img

def load_ben_color(path, img_size=300, sigmaX=10):
    image = cv2.imread(path)
    if image is None:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

class APTOSTestDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        name = self.df.iloc[idx]['id_code']
        path = os.path.join(self.img_dir, f"{name}.png")
        img = load_ben_color(path, 300, 10)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        img = img.transpose(2, 0, 1)
        return torch.tensor(img, dtype=torch.float32), name

# ============================================================
# 3. 경로
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_DIR = '/kaggle/input/datasets/jongwoo108/aptos-v2-coral'
WEIGHTS = [
    os.path.join(WEIGHT_DIR, 'best_fold0.pth'),
    os.path.join(WEIGHT_DIR, 'best_fold1.pth'),
    os.path.join(WEIGHT_DIR, 'best_fold2.pth'),
]
TEST_CSV = '/kaggle/input/competitions/aptos2019-blindness-detection/test.csv'
TEST_IMG = '/kaggle/input/competitions/aptos2019-blindness-detection/test_images'

# ============================================================
# 4. 3-fold 앙상블 추론
# ============================================================
test_df = pd.read_csv(TEST_CSV)
test_loader = DataLoader(APTOSTestDataset(test_df, TEST_IMG), batch_size=32, shuffle=False)

all_logits = []

for w_path in WEIGHTS:
    model = APTOSModel('efficientnet_b3', 5).to(device)
    ckpt = torch.load(w_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ {os.path.basename(w_path)} 로드 (Epoch {ckpt['epoch']}, Kappa {ckpt['kappa']:.4f})")

    fold_logits = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=os.path.basename(w_path)):
            images = images.to(device)
            logits = model(images)
            fold_logits.append(logits.cpu())

    all_logits.append(torch.cat(fold_logits, dim=0))

# 3개 모델의 로짓을 평균
avg_logits = torch.stack(all_logits).mean(dim=0)
preds = (torch.sigmoid(avg_logits) > 0.5).sum(dim=1).long().numpy()

# ============================================================
# 5. 제출
# ============================================================
submission = pd.DataFrame({'id_code': test_df['id_code'], 'diagnosis': preds})
print("\n--- 예측 분포 ---")
print(submission['diagnosis'].value_counts().sort_index())
submission.to_csv('submission.csv', index=False)
print("🎯 submission.csv 생성 완료!")
