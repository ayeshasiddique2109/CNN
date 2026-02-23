import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create plots folder
if not os.path.exists("plots"):
    os.makedirs("plots")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.CIFAR10('./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10('./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN,self).__init__()

        self.conv1 = nn.Conv2d(3,32,3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,3)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(64*6*6,128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1,64*6*6)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = BasicCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 10

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []

for epoch in range(num_epochs):

    model.train()
    running_loss, correct, total = 0,0,0

    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    train_loss = running_loss/len(train_loader)
    train_acc = 100*correct/total

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    model.eval()
    test_loss, correct, total = 0,0,0

    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            test_loss += loss.item()

            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

    test_loss = test_loss/len(test_loader)
    test_acc = 100*correct/total

    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% Test Acc={test_acc:.2f}%")

print("\nFINAL BASIC MODEL RESULTS")
print(f"Final Training Accuracy: {train_acc_list[-1]:.2f}%")
print(f"Final Testing Accuracy: {test_acc_list[-1]:.2f}%")

# Save Loss Plot
plt.figure()
plt.plot(train_loss_list,label="Train Loss")
plt.plot(test_loss_list,label="Test Loss")
plt.legend()
plt.title("Basic Model Loss")
plt.savefig("plots/basic_loss.png")
plt.close()

# Save Accuracy Plot
plt.figure()
plt.plot(train_acc_list,label="Train Accuracy")
plt.plot(test_acc_list,label="Test Accuracy")
plt.legend()
plt.title("Basic Model Accuracy")
plt.savefig("plots/basic_accuracy.png")
plt.close()