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

# 데이터 전처리 및 데이터 로더에 저장
* train_transforms: 이미지 데이터에 대한 전처리 및 데이터 증강 <br>
<span style="color: red;">-</span> transforms.ToTensor(): 이미지를 PyTorch 텐서로 변환<br>
<span style="color: red;">-</span> transforms.RandomHorizontalFlip(): 이미지를 수평으로 뒤집기<br>
<span style="color: red;">-</span> transforms.RandomRotation(45): 이미지를 무작위로 ±45도 회전<br>
<span style="color: red;">-</span> transforms.RandomAutocontrast(): 이미지의 명암 대비를 자동으로 조정<br>
<span style="color: red;">-</span> transforms.Resize((64, 64)): 모든 이미지를 64x64 픽셀로 크기 조정<br>
<span style="color: red;">-</span> ImageFolder: 주어진 디렉토리(clouds_train)에서 이미지를 로드하며, 각 클래스별로 폴더가 정리되어 있어야 합니다.<br>
<span style="color: red;">-</span> DataLoader: dataset_train을 기반으로 데이터 로더를 생성하여 배치 단위로 데이터를 로드할 수 있게 합니다.<br>
<span style="color: red;">-</span> batch_size=16: 각 배치에 포함되는 이미지 수를 16으로 설정합니다.<br>
<span style="color: red;">-</span> shuffle=True: 에포크마다 데이터를 섞어줍니다.<br>


