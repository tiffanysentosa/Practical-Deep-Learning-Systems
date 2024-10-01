import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvModule, self).__init__()
        padding = kernel_size // 2  
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3):
        super(InceptionModule, self).__init__()
        self.branch1x1 = ConvModule(
            in_channels, out_channels_1x1, kernel_size=1, stride=1)
        self.branch3x3 = ConvModule(
            in_channels, out_channels_3x3, kernel_size=3, stride=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        outputs = torch.cat([branch1x1, branch3x3], dim=1)
        return outputs

class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_channels_3x3):
        super(DownsampleModule, self).__init__()
        self.branch3x3 = ConvModule(
            in_channels, out_channels_3x3, kernel_size=3, stride=2)
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch_pool = self.branch_pool(x)
        outputs = torch.cat([branch3x3, branch_pool], dim=1)
        return outputs

class InceptionSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionSmall, self).__init__()
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=96,
            kernel_size=3,
            stride=1)
        self.inception_block1 = InceptionModule(
            in_channels=96, out_channels_1x1=32, out_channels_3x3=32)
        self.inception_block2 = InceptionModule(
            in_channels=64, out_channels_1x1=32, out_channels_3x3=48)
        self.downsample1 = DownsampleModule(in_channels=80, out_channels_3x3=80)
        self.inception_block3 = InceptionModule(
            in_channels=160, out_channels_1x1=112, out_channels_3x3=48)
        self.inception_block4 = InceptionModule(
            in_channels=160, out_channels_1x1=96, out_channels_3x3=64)
        self.inception_block5 = InceptionModule(
            in_channels=160, out_channels_1x1=80, out_channels_3x3=80)
        self.inception_block6 = InceptionModule(
            in_channels=160, out_channels_1x1=48, out_channels_3x3=96)
        self.downsample2 = DownsampleModule(in_channels=144, out_channels_3x3=96)
        self.inception_block7 = InceptionModule(
            in_channels=240, out_channels_1x1=176, out_channels_3x3=160)
        self.inception_block8 = InceptionModule(
            in_channels=336, out_channels_1x1=176, out_channels_3x3=160)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(336, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.downsample1(x)
        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.downsample2(x)
        x = self.inception_block7(x)
        x = self.inception_block8(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# DATA PREPARATION

def prepare_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                          download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                         download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return train_dataset, test_dataset, train_loader, test_loader

# LR RANGE TEST

def lr_range_test(train_loader, start_lr, end_lr, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    num_iter = len(train_loader)
    model = InceptionSmall(num_classes=10).to(device)
    total_iterations = num_iter * num_epochs
    learning_rates = np.logspace(
        np.log10(start_lr),
        np.log10(end_lr),
        num=total_iterations)
    train_losses = []
    train_accuracies = []
    lrs = []

    optimizer = optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)
    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for data, target in progress_bar:
            if global_step >= total_iterations:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            lr = learning_rates[global_step]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == target).sum().item()
            accuracy = correct / target.size(0)
            train_accuracies.append(accuracy)
            train_losses.append(loss.item())
            lrs.append(lr)

            global_step += 1

    # Plot Training Loss vs Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, train_losses, color='blue')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Learning Rate (5 Epochs)')
    plt.grid(True)
    plt.show()

    # Plot Training Accuracy vs Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, train_accuracies, color='green')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy vs Learning Rate (5 Epochs)')
    plt.grid(True)
    plt.show()

# CYCLICAL LEARNING RATE

def cyclical_learning_rate(train_loader, test_loader,
                           lr_min, lr_max, num_epochs=5):
    model = InceptionSmall(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr_min, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_steps = num_epochs * len(train_loader)
    scheduler = CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=total_steps // 2,
        mode='exp_range',
        gamma=0.99994)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_loss = val_running_loss / len(test_loader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'g-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# BATCH SIZE TEST

def batch_size_test(train_dataset, test_loader, lr_max):
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]   # T4 GPU, 16GB RAM
    train_losses_batch = []
    test_accuracies = []
    num_epochs = 1

    for batch_size in batch_sizes:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=2)
        model = InceptionSmall(num_classes=10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr_max, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader, desc=f"Batch Size {batch_size}, Epoch {epoch + 1}")
            for data, target in progress_bar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        epoch_loss = running_loss / total_samples
        train_losses_batch.append(epoch_loss)

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        print(f"Batch Size {batch_size}, Training Loss: {epoch_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    plt.figure()
    plt.plot(batch_sizes, train_losses_batch, marker='d', color='green')
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Batch Size')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(batch_sizes, test_accuracies, marker='o', color='red')
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Batch Size')
    plt.grid(True)
    plt.show()


# MAIN FUNCTION

def main():
    train_dataset, test_dataset, train_loader, test_loader = prepare_data()
    batch_size_test(train_dataset, test_loader, 1e-2)

if __name__ == "__main__":
    main()