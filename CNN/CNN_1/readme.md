```python
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.Resize((64, 64))
])

dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)
dataloader_train = DataLoader(
    dataset_train, shuffle=True, batch_size=16
)
```

### 데이터 전처리 및 데이터 로더에 저장

* train_transforms: 이미지 데이터에 대한 전처리 및 데이터 증강
* transforms.ToTensor(): 이미지를 PyTorch 텐서로 변환
* transforms.RandomHorizontalFlip(): 이미지를 수평으로 뒤집기
* transforms.RandomRotation(45): 이미지를 무작위로 ±45도 회전
* transforms.RandomAutocontrast(): 이미지의 명암 대비를 자동으로 조정
* transforms.Resize((64, 64)): 모든 이미지를 64x64 픽셀로 크기 조정
* ImageFolder: 주어진 디렉토리에서 이미지를 로드하며, 각 클래스별로 폴더가 정리되어 있어야 함
* DataLoader: dataset_train을 기반으로 데이터 로더를 생성하여 배치 단위로 데이터를 로드할 수 있게 한다
* batch_size=16: 배치사이즈 16
* shuffle=True: 에폭마다 데이터 섞기

<br><br>

```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
```
### 데이터 전처리 및 데이터 로더에 저장

* Net 클래스: 이미지 분류 모델을 정의
* feature_extractor: 합성곱 신경망(CNN)으로 이미지의 특징을 추출
* nn.Conv2d(3, 32, kernel_size=3, padding=1): 3 채널(RGB)의 입력 이미지에 대해 32개의 필터를 사용해 3x3 커널 크기로 합성곱 연산
* nn.ELU(): 활성화 함수로 ELU(Exponential Linear Unit)를 사용
* nn.MaxPool2d(kernel_size=2): 출력의 공간적 크기를 절반으로 줄임
* nn.Conv2d(32, 64, kernel_size=3, padding=1): 64개의 필터를 사용해 합성곱 연산을 한 번 더 수행
* nn.MaxPool2d(kernel_size=2): 출력의 공간적 크기를 절반으로 줄임.
* nn.Flatten(): 2D 이미지를 1D 벡터로 평탄화
* classifier: 평탄화된 특징 벡터를 받아 최종적으로 클래스에 대한 출력을 생성
* nn.Linear(64 * 16 * 16, num_classes): 64개의 16x16 특징 맵을 입력으로 받아, num_classes 개의 출력을 생성하는 완전 연결 계층

<br><br>

```python
net = Net(num_classes=7)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
```
### 학습 설정
* 7개 클래스 분류할 net모델 정의
* loss 함수 정의 : CrossEntropyLoss 사용해서 출력과 실제 레이블 간의 오차 계산
* 옵티마이저 정의 : Adam 옵티마이저 사용하고, 학습률은 0.001

<br><br>

```python
for epoch in range(3):
    running_loss = 0.0
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
```
### 학습
* 3 epoch
* 배치 단위 학습
* loss 계산 및 역전파: 예측 결과와 실제 레이블 간의 loss을 계산하고,gradient 역전파한 뒤 옵티마이저가 파라미터를 업데이트
* loss 누적: 각 배치의 loss을 더해 epoch 동안의 총 loss을 계산
* loss 출력: 각 epoch가 끝날 때마다 평균 loss을 출력


```python
# Define metrics
metric_precision = Precision(task="multiclass", num_classes=7, average="micro")
metric_recall = Recall(task="multiclass", num_classes=7, average="micro")

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1) # 첫번째 값: 최대값, 두번째 값: 해당 인덱스(클래스)
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()
print(f"Precision: {precision}")
print(f"Recall: {recall}")
```
### 평
* _, preds = torch.max(outputs, 1): 출력 중 가장 높은 값을 가진 클래스를 예측으로 선택합니다. torch.max(outputs, 1)은 두 가지 값을 반환하며, 첫 번째는 최대값이고, 두 번째는 해당 값의 인덱스(클래스). 이 코드에서는 클래스 인덱스를 preds로 저장
* metric_precision(preds, labels) 및 metric_recall(preds, labels): 현재 배치의 예측과 실제 레이블을 사용하여 정밀도와 재현율을 업데이트합니다. 이때 내부적으로 지표 계산에 필요한 값을 축적
