import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

save_path = 'training_data.pkl'
with open(save_path, 'rb') as f:
    loaded_data = pickle.load(f)
# 访问加载的数据
loaded_all_labels = loaded_data['all_labels']
loaded_all_predictions = loaded_data['all_predictions']

# 设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'

# 创建从8到15.8，步长为0.2的数组
custom_ticks = np.round(np.arange(8, 16, 0.2), 1)


# 生成混淆矩阵
cm = confusion_matrix(loaded_all_labels, loaded_all_predictions)

# 设置图形大小
fig, ax = plt.subplots(figsize=(20, 12))  # 宽20，高10，比例为2:1

# 显示混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=custom_ticks)
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax)

# 调整布局为长方形
ax.set_aspect(0.65)  # 单元格的宽高比，例如 0.5 表示高度是宽度的一半

# 优化标题和轴标签字体大小
plt.title('Confusion Matrix', fontsize=20, weight='bold', pad=20)
plt.xlabel('Predicted Labels', fontsize=20, weight='bold', labelpad=15)
plt.ylabel('True Labels', fontsize=20, weight='bold', labelpad=15)

# 调整坐标轴刻度字体大小
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12)

# 调整矩阵内部文字的样式和大小
for text in ax.texts:  # 遍历所有单元格中的文本
    text.set_fontsize(10)  # 设置字体大小
    text.set_fontweight('bold')  # 加粗字体

# 调整 colorbar 的字体大小
cbar = disp.im_.colorbar  # 获取 colorbar 实例
cbar.ax.tick_params(labelsize=12)  # 设置 colorbar 刻度字体大小

# 调整刻度线宽度
ax.tick_params(width=1.5)

# # 自动调整布局
# plt.tight_layout()

# 保存为矢量图格式 (SVG)
plt.savefig("confusion_matrix.svg", format="svg", dpi=300)

plt.show()



