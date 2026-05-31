import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torchvision.models import ResNet18_Weights

print("Loading Dataset...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

print("Classes Found:")
print(dataset.classes)

print("Total Images:", len(dataset))

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

print("Loading ResNet18 Model...")

model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(dataset.classes)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

print("Training Started...\n")

for epoch in range(epochs):

    running_loss = 0.0

    for images, labels in loader:

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {avg_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "model.pth"
)

print("\nModel Saved Successfully!")
print("Training Completed!")