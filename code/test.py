from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from model import Model
import pandas as pd
import numpy as np
import torch
# 1. 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features1, labels):
        self.features1 = torch.tensor(features1.values.astype(np.float32))
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, index):
        x1 = self.features1[index]
        y = self.labels[index]
        return x1,y

print(torch.cuda.is_available())
# 5. 数据读取和预处理
test = pd.read_csv("dataset/Alg_real_Data_ESM1bEmbedder.csv")
x_test = test.iloc[:, 1:]
x_test_label = test.iloc[:,0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Accuracy Pre Recall F1 ROC Pr_AUC")
test_dataset = CustomDataset(x_test,x_test_label)
test_loader = DataLoader(test_dataset, batch_size=32)
# 加载验证集上表现最好的模型
best_model = Model()
best_model.load_state_dict(torch.load("Draw/deeplearning/best_model/ESM.pt"))
best_model.eval()
best_model.to(device)
with torch.no_grad():
    all_predictions = []
    all_labels = []
    all_auc = []

    for data2, labels in test_loader:
        data2 = data2.to(device)
        labels = labels.to(device)
        final_output = best_model( data2.unsqueeze(1))

        scores = final_output[:,0].tolist()
        all_auc.extend(scores)
        final_output = (final_output.data > 0.5).int()
        all_predictions.extend(final_output.tolist())
        all_labels.extend(labels.tolist())
with open("dataset/test_ESM1b.csv",'a') as f:
    np.savetxt(f, all_predictions)

