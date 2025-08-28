import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as scio
from torch.utils.data import DataLoader, TensorDataset
from inception import InceptionBlock
from some_block import ResidualBlock, MultiScaleConvBlock
import random

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
        # self.self_attention = SelfAttention(in_channels=128)

        # 改进后的多尺度卷积块
        self.multiscale_block = MultiScaleConvBlock(64, 64)
        # # 添加残差块
        self.residual_block = ResidualBlock(64, 64)

        self.multiscale_block1 = MultiScaleConvBlock(64, 128)
        # # 添加残差块
        self.residual_block1 = ResidualBlock(128, 128)

        # self.se = SqueezeExcitation(in_channels=128)
        self.classify = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_features=128 * 1 * timewindows, out_features=40, bias=True)
            # nn.Softmax(dim=1)
        )
        self.inceptionblock = InceptionBlock(3, 16)
    def forward(self, x):
        features = x
        x1 = self.firstconv(x)
        x1 = self.multiscale_block(x1)
        x1 = self.residual_block(x1)
        x2 = self.inceptionblock(x)
        x = self.separableconv2(x1+x2)
        x = self.multiscale_block1(x)
        x = self.residual_block1(x)
        x = self.separableconv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classify(x)
        return x, features


device = 'cpu'
torch.manual_seed(1100)
# torch.backends.cudnn.enabled = True  # 关闭 CuDNN 加速
# torch.backends.cudnn.benchmark = True  # 开启 CuDNN 自动优化
timeexp = 0.6   #选哪个数据集
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
# 读第几个被试
s = 0   #第一个被试
test_x = AllData[..., s]  # 维度为 (9, 50, 3, 40, 6)
test_x = test_x.transpose(4, 3, 2, 0, 1)  # 重新调整为 (6, 40, 9, 50)
test_x = test_x.reshape(-1, test_x.shape[2], test_x.shape[3], test_x.shape[4])  # 重新调整为 (240, 3, 9, 50)

# 处理测试标签
test_y = y_AllData[..., s]  # 维度为 (1, 40, 6)
test_y = test_y.transpose(2, 1, 0)
test_y = test_y.reshape(-1) - 1  # 去掉多余维度并调整标签 (240,)

test_x = torch.tensor(test_x, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.long)

test_dataset = TensorDataset(test_x, test_y)

Net = XXG2net()   # EEGNet, EEGTCNet, M_FANet   Mynet1
Net.load_state_dict(torch.load('model_weights_subject1.pth', weights_only=True))
Net.to('cpu')
Net.eval()

loader = DataLoader(test_dataset, batch_size=1)

output_features = []
output_y = []
for X, y in loader:
    _, features = Net(X)
    output_features.append(features.flatten(1).clone().detach().numpy())
    output_y.append(y.clone().detach().numpy().reshape(-1))
features = features.flatten(1)
feature_dim = features.size(-1)
output_x = np.array(output_features).reshape(-1, feature_dim)
output_y = np.array(output_y).reshape(-1)

# tsne = TSNE(n_components=2, random_state=42, early_exaggeration=800)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(output_x)

# 获取不同类别的索引
unique_classes = np.unique(output_y)
# label_names = {0: 'Left', 1: 'Right', 2: 'Foot', 3: 'Tongue'}
# 创建从8到15.8，步长为0.2的数组
label_names = np.round(np.arange(8, 16, 0.2), 1)

plt.figure(figsize=(6, 5))
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# ################################### Color ##############################################
# colors = cm.rainbow(np.linspace(0, 1, len(label_names)))
# colors = [cm.rainbow(i/len(set(label_names))) for i in range(len(set(label_names)))]
# colors = ['#D2362C', '#FADF93', '#5374B0', '#00B050']

red = '#FF0000'
shallow_red = '#FF4A55'
green = '#00FF00'
shallow_green = '#55FFAA'
blue = '#0000FF'
shallow_blue = '#558FFF'
orange = '#FFA500'
yellow = '#FFFF00'
purple = '#800080'
cyan = '#00FFFF'
pink = '#FFC0CB'
Magenta = '#FF00FF'
shallow_Magenta = '#EE55FF'
# 将这些颜色组合成一个列表
num_classes = 40
colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(num_classes)]

# #######################################################################################

for cls, color in zip(unique_classes, colors):
    idx = np.where(output_y == cls)
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label_names[cls], color=color, s=40, alpha=0.75)
plt.xticks([])
plt.yticks([])
# plt.xlabel('t-SNE Feature 1',fontsize=20)
# plt.ylabel('t-SNE Feature 2',fontsize=20)
# plt.title('1', fontsize=26)

# 设置边框线的宽度
ax = plt.gca()
spine_width = 1.5  # 示例的边框宽度为2，您可以根据需要调整这个值
ax.spines['top'].set_linewidth(spine_width)
ax.spines['bottom'].set_linewidth(spine_width)
ax.spines['left'].set_linewidth(spine_width)
ax.spines['right'].set_linewidth(spine_width)

plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
plt.grid(False)
plt.legend(fontsize=16)
plt.show()


