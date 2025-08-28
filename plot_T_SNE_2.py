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
from matplotlib import cm, colors

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

        x1 = self.firstconv(x)
        x1 = self.multiscale_block(x1)
        x1 = self.residual_block(x1)
        x2 = self.inceptionblock(x)
        x = self.separableconv2(x1+x2)
        features = x
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
timeexp = 1.0   #选哪个数据集
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
num_classes = 40

# tsne = TSNE(n_components=2, random_state=42, early_exaggeration=800)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(output_x)

# 生成渐变颜色
cmap = plt.colormaps['jet']  # 使用新的 Matplotlib colormap API      试试  turbo rainbow viridis plasma inferno magma jet twilight hsv coolwarm
color_norm = colors.Normalize(vmin=0, vmax=num_classes - 1)  # 正规化颜色范围
scalar_map = cm.ScalarMappable(norm=color_norm, cmap=cmap)

# 绘制 TSNE 图
fig, ax = plt.subplots(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['Times New Roman']

for cls in range(num_classes):
    idx = np.where(output_y == cls)
    ax.scatter(X_tsne[idx, 0],
               X_tsne[idx, 1],
               label=f"Class {cls + 1}",
               color=scalar_map.to_rgba(cls),
               s=310, #点的大小
               edgecolors='black',  # 点的轮廓颜色
               linewidth=0.7,  # 点的轮廓宽度
               alpha=0.9)# 半透明效果

# 隐藏刻度
ax.set_xticks([])
ax.set_yticks([])

# 设置边框线宽度
spine_width = 1.5
ax.spines['top'].set_linewidth(spine_width)
ax.spines['bottom'].set_linewidth(spine_width)
ax.spines['left'].set_linewidth(spine_width)
ax.spines['right'].set_linewidth(spine_width)

# 添加颜色条并匹配绘图
cbar = fig.colorbar(scalar_map, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.set_title('Class', fontsize=20, weight='bold', pad=10)
cbar.ax.tick_params(labelsize=20)# 刻度字体大小

# # 设置点的阴影效果 (近似模拟)
# ax.set_facecolor('#f0f0f0')  # 背景颜色

# 布局调整
fig.tight_layout()
plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.3, 1.03), frameon=False)
# 保存为矢量图格式 (SVG)
plt.savefig("T-SNE_框架之后.svg", format="svg", dpi=300)

plt.show()