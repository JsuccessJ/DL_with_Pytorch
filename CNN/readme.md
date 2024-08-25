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
```python
