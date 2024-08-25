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

* train_transforms: 이미지 데이터에 대한 전처리 및 데이터 증강 <br>
* transforms.ToTensor(): 이미지를 PyTorch 텐서로 변환<br>
* transforms.RandomHorizontalFlip(): 이미지를 수평으로 뒤집기<br>
* transforms.RandomRotation(45): 이미지를 무작위로 ±45도 회전<br>
* transforms.RandomAutocontrast(): 이미지의 명암 대비를 자동으로 조정<br>
* transforms.Resize((64, 64)): 모든 이미지를 64x64 픽셀로 크기 조정<br>
* ImageFolder: 주어진 디렉토리에서 이미지를 로드하며, 각 클래스별로 폴더가 정리되어 있어야 함<br>
* DataLoader: dataset_train을 기반으로 데이터 로더를 생성하여 배치 단위로 데이터를 로드할 수 있게 한다<br>
* batch_size=16: 배치사이즈 16 <br>
* shuffle=True: 에폭마다 데이터 섞기 <br>

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

* Net 클래스: 이미지 분류 모델을 정의합니다.
* feature_extractor: 합성곱 신경망(CNN)으로 이미지의 특징을 추출하는 부분입니다.
* nn.Conv2d(3, 32, kernel_size=3, padding=1): 3 채널(RGB)의 입력 이미지에 대해 32개의 필터를 사용해 3x3 커널 크기로 합성곱 연산을 수행합니다.
* nn.ELU(): 활성화 함수로 ELU(Exponential Linear Unit)를 사용합니다.
* nn.MaxPool2d(kernel_size=2): 최대 풀링을 사용해 출력의 공간적 크기를 절반으로 줄입니다.
* nn.Conv2d(32, 64, kernel_size=3, padding=1): 64개의 필터를 사용해 합성곱 연산을 한 번 더 수행합니다.
* nn.MaxPool2d(kernel_size=2): 또 한 번 최대 풀링을 수행하여 공간적 크기를 줄입니다.
* nn.Flatten(): 2D 이미지를 1D 벡터로 평탄화합니다.
* classifier: 평탄화된 특징 벡터를 받아 최종적으로 클래스에 대한 출력을 생성합니다.
* nn.Linear(64 * 16 * 16, num_classes): 64개의 16x16 특징 맵을 입력으로 받아, num_classes 개의 출력을 생성하는 완전 연결 계층입니다.
