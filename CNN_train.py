from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch
from torch.nn as nn

class Net(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.feature_extractor = nn.Sequential(
      nn.Conv2d(3,32,kernel_size = 3, padding =1),
      nn.ELU(),
      nn,MaxPool2d(kernel_size=2),
      nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
      nn.ELU(),
      nn.MaxPool2d(kernel_size = 2),
      nn.Flatten(),
    )
    self.classifier = nn.Linear(64*16*16, num_classes)
  
  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x


# Define transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.Resize((64,64))
])

dataset_train = ImageFolder(
  "clouds_train",
  transform=train_transforms,
)
dataloader_train = DataLoader(
  dataset_train, shuffle=True, batch_size=16
)


# Define the model
net = Net(num_classes=7)
# Define the loss function
criterion = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(3):
    running_loss = 0.0
    # Iterate over training batches
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
