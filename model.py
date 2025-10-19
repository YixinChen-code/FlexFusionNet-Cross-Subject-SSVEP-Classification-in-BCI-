import scipy.io as scio
import numpy as np
import torch
import pandas as pd
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from inception import InceptionBlock
from OMS import OmniScaleCNN_2D
from some_block import ResidualBlock, MultiScaleConvBlock
import torch.nn.functional as F
from scipy.io import savemat
def calculate_itr(acc_matrix, M, t):
    acc_matrix = torch.tensor(acc_matrix, dtype=torch.float32)
    num_subjects = acc_matrix.shape
    itr_matrix = torch.zeros_like(acc_matrix)
    for i in range(num_subjects):
        p = acc_matrix[i].item() / 100
        if p < 1 / M:
            itr_matrix[i] = 0
        elif p == 1:
            itr_matrix[i] = np.log2(M) * (60 / t)
        else:
            itr_matrix[i] = (np.log2(M) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (M - 1))) * (60 / t)
    return itr_matrix


class XXG2net(nn.Module):
    def __init__(self):
        super(XXG2net, self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding='same', bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(),
            nn.Dropout(p=0.10)
        )
        self.separableconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(9, 1), stride=(1, 1), groups=64, bias=False)
        )
        self.separableconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 5), padding='same', groups=128, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), padding='same', bias=False),
            nn.ELU(),
            nn.Dropout(p=0.95)
        )

        self.multiscale_block = MultiScaleConvBlock(64, 64)
        self.residual_block = ResidualBlock(64, 64)
        self.multiscale_block1 = MultiScaleConvBlock(64, 128)
        self.residual_block1 = ResidualBlock(128, 128)

        self.classify = nn.Sequential(
            nn.Linear(in_features=128 * 1 * timewindows, out_features=40, bias=True)
        )
        self.inceptionblock = InceptionBlock(3, 16)
    def forward(self, x):
        x1 = self.firstconv(x)
        x1 = self.multiscale_block(x1)
        x1 = self.residual_block(x1)
        x2 = self.inceptionblock(x)
        x = self.separableconv2(x1+x2)
        x = self.multiscale_block1(x)
        x = self.residual_block1(x)
        x = self.separableconv3(x)
        x = x.view(x.size(0), -1)  
        x = self.classify(x)
        return x

print(torch.cuda.is_available())  
torch.cuda.set_device(0)
device = torch.device("cuda")
torch.manual_seed(1100)
# torch.backends.cudnn.enabled = True  
# torch.backends.cudnn.benchmark = True 
timeexp = 0.8   #选哪个数据集
dataset = 'Benchmark'
if dataset == 'Benchmark':
    path1 = r'F:\Article_Reproduction\SSVEP\DNN_for_Python\Benchmark\y_AllData{:.1f}.mat'.format(timeexp)
    path2 = r'F:\Article_Reproduction\SSVEP\DNN_for_Python\Benchmark\AllData{:.1f}.mat'.format(timeexp)
elif dataset == 'Beta':
    path1 = r'F:\Article_Reproduction\SSVEP\DNN_for_Python\Beta\y_AllData{:.1f}.mat'.format(timeexp)
    path2 = r'F:\Article_Reproduction\SSVEP\DNN_for_Python\Beta\AllData{:.1f}.mat'.format(timeexp)
Datapath1 = scio.loadmat(path1)
Datapath2 = scio.loadmat(path2)
y_AllData = Datapath1['y_AllData'] #(1, 40, 6, 35)
AllData = Datapath2['AllData'] #(9, 50, 3, 40, 6, 35)

channels, timewindows, subbands, totalcharacter, totalblock, totalsubject = AllData.shape
print(channels, timewindows, subbands, totalcharacter, totalblock, totalsubject)

testaccuracy = np.zeros((totalsubject))
testitr = np.zeros((totalsubject))
for s in range(totalsubject):
    allsubject = list(range(totalsubject))
    allsubject.pop(s)  # 从列表中排除指定的subject

    train_x = AllData[..., allsubject]
    train_x = train_x.transpose(5, 4, 3, 2, 0, 1)  # 转置为 (5, 40, 3, 9, 50)
    train_x = train_x.reshape(-1, train_x.shape[3], train_x.shape[4], train_x.shape[5])

    train_y = y_AllData[..., allsubject]  # 维度为 (1, 40, 6, 34)
    train_y = train_y.transpose(3, 2, 1, 0)  # 转置为 (34, 6, 40, 1)
    train_y = train_y.reshape(-1) - 1  # 重新调整为 (8160,)

    test_x = AllData[..., s]  # 维度为 (9, 50, 3, 40, 6)
    test_x = test_x.transpose(4, 3, 2, 0, 1)  # 重新调整为 (6, 40, 9, 50)
    test_x = test_x.reshape(-1, test_x.shape[2], test_x.shape[3], test_x.shape[4])# 重新调整为 (240, 3, 9, 50)

    test_y = y_AllData[..., s]  # 维度为 (1, 40, 6)
    test_y = test_y.transpose(2, 1, 0)
    test_y = test_y.reshape(-1) - 1  # 去掉多余维度并调整标签 (240,)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.long)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # 定义模型

    model = XXG2net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-13, weight_decay=1e-10)

    for epoch in range(80):
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # 清零梯度
            outputs = model(inputs)# 前向传播
            loss = criterion(outputs, labels)
            loss.backward()# 反向传播
            optimizer.step()# 更新参数
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        train_accuracy = 100 * correct / total

        # 计算测试精度
        model.eval()
        correct = 0
        testtotal = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                testtotal += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        test_accuracy = 100 * correct / testtotal
        M = totalcharacter  # 类别数
        t = timewindows/250+0.14  # 时间窗数
        p = test_accuracy / 100 # 噪声率
        itr = (np.log2(M) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (M - 1))) * (60 / t)

        print(
            f"Subject {s + 1}, Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, ITR: {itr:.2f}")
    testaccuracy[s] = test_accuracy  
    testitr[s] = itr  
    print(f"Subject {s + 1}, Final Test Accuracy: {test_accuracy:.2f}, ITR: {itr:.2f}")
average_accuracy=np.mean(testaccuracy)
print("Average accuracy:", average_accuracy)
testaccuracy=torch.tensor(testaccuracy)
testitr=torch.tensor(testitr)
data = {
    "Test Accuracy": testaccuracy,
    "ITR (bits/min)": testitr
}


df = pd.DataFrame(data)


df.insert(0, 'Subject', ['被试{}'.format(i + 1) for i in range(len(df))])
df["Test Accuracy"] = df["Test Accuracy"].round(2)
df["ITR (bits/min)"] = df["ITR (bits/min)"].round(2)

with pd.ExcelWriter('results_{:.1f}.xlsx'.format(timeexp), engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)


