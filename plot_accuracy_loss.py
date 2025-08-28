import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from matplotlib import rcParams


save_path = 'training_data.pkl'
with open(save_path, 'rb') as f:
    loaded_data = pickle.load(f)

# 访问加载的数据
loaded_all_labels = loaded_data['all_labels']
loaded_all_predictions = loaded_data['all_predictions']
loaded_train_losses = loaded_data['train_losses']
loaded_train_accuracies = loaded_data['train_accuracies']
loaded_test_accuracies = loaded_data['test_accuracies']

# 设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.weight'] = 'bold'

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 82), loaded_train_accuracies, label='Train Accuracy')
plt.plot(range(1, 82), loaded_test_accuracies, label='Test Accuracy')
# 优化标题和轴标签字体大小
plt.title('Training and Testing Accuracy', fontsize=20, weight='bold', pad=20)
plt.xlabel('Epoch', fontsize=20, weight='bold', labelpad=15)
plt.ylabel('Accuracy (%)', fontsize=20, weight='bold', labelpad=15)
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(range(1, 82), loaded_train_losses, label='Train Loss')
plt.title('Training Loss', fontsize=20, weight='bold', pad=20)
plt.xlabel('Epoch', fontsize=20, weight='bold', labelpad=15)
plt.ylabel('Loss', fontsize=20, weight='bold', labelpad=15)

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.savefig("confusion_matrix.svg", format="svg", dpi=300)




