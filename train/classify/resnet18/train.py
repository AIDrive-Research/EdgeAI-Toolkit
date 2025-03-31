import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 配置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
lr = 0.0001
batch_size = 64
n_classes = 2
pretrain = True
epochs = 300

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集
train_dataset = datasets.ImageFolder(root='/code/edgeai-toolkit/train/classify/resnet18/dataset/train/',
                                     transform=transform_train)
test_dataset = datasets.ImageFolder(root='/code/edgeai-toolkit/train/classify/resnet18/dataset/test/',
                                    transform=transform_test)

# 模型初始化
model = models.resnet18(pretrained=pretrain)
model.fc = nn.Linear(512, n_classes)
model.to(device)

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 训练和评估的通用函数
def run_epoch(model, data_loader, loss_fn, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, corrects, total = 0, 0, 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * inputs.size(0)
        corrects += torch.sum(preds.eq(labels))
        total += labels.size(0)
        print(total)

    avg_loss = total_loss / total
    accuracy = 100 * corrects / total
    return avg_loss, accuracy


# 主函数
def main():
    best_accuracy = 0.0
    for epoch in range(epochs):
        start_time = time.time()

        # 训练和验证
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, loss_fn, train=False)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            print('保存最佳模型！')
            torch.save(model.state_dict(), "best.pth")

        end_time = time.time()
        print(f"第{epoch + 1}轮 | 耗时: {end_time - start_time:.2f}s | "
              f"训练损失: {train_loss:.5f} | 训练准确率: {train_acc:.2f}% | "
              f"验证损失: {test_loss:.5f} | 验证准确率: {test_acc:.2f}%")


# 执行主函数
if __name__ == "__main__":
    main()
