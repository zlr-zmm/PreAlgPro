import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, auc
from model import Model
import pandas as pd
import numpy as np
from numpy import interp
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


# 5. 数据读取和预处理
train = pd.read_csv("dataset/Alg_train_protT5.csv")
x_train = train.iloc[:, 1:]
x_train_label = train.iloc[:,0]
# 打印模型结构
num_epochs = 100
batch_size = 32
num_folds = 5
# 7. 定义交叉验证
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Accuracy Pre Recall F1 ROC Pr_AUC")
best_cnn_mcc = 0
best_transformer_mcc = 0
tprs = []
fprs = []
precisions = []
recalls = []
best_acc = 0
best_acc_model = None
mean_fpr_linspace = np.linspace(0, 1, 100)
for fold, (train_index, val_index) in enumerate(kf.split(train)):
    # print(f"Fold: {fold+1}")
    train_ProtT5 = x_train.iloc[train_index]
    train_labels = x_train_label.iloc[train_index]

    val_ProtT5 = x_train.iloc[val_index]
    val_labels = x_train_label.iloc[val_index]

    train_dataset = CustomDataset(train_ProtT5,train_labels)
    val_dataset = CustomDataset(val_ProtT5, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)


    cnn_model = Model()
    cnn_model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.00001)

    cnn_model.train()
    x_list = []
    transformer_data_list = []
    cnn_data_list = []
    final_output_list = []
    for epoch in range(num_epochs):
        for data2, labels in train_loader:
            data2 = data2.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            final_output = cnn_model(data2.unsqueeze(1))
            scores = final_output[:, 0]
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

    all_predictions = []
    all_labels = []
    all_auc = []
    for data2, labels in val_loader:
        data2 = data2.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()
        final_output = cnn_model(data2.unsqueeze(1))
        scores = final_output[:, 0].tolist()
        all_auc.extend(scores)
        final_output = (final_output.data > 0.5).int()
        # final_output = final_output.cpu().detach().numpy()
        all_labels.extend(labels.tolist())
        all_predictions.extend(final_output.tolist())

    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions)
    val_roc = roc_auc_score(all_labels, all_auc)
    val_recall = recall_score(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions)
    precision, recall, _ = precision_recall_curve(all_labels, all_auc)
    fpr, tpr, _  = roc_curve(all_labels, all_auc)
    val_pr_auc = auc(recall, precision)
    num_samples = 100
    precision_sampled = np.linspace(0, 1, num_samples)
    recall_sampled = np.interp(precision_sampled, precision, recall)
    fpr_sampled = np.linspace(0, 1, num_samples)
    tpr_sampled = np.interp(fpr_sampled, fpr, tpr)



    fprs.append(fpr_sampled)
    tprs.append(tpr_sampled)
    precisions.append(precision_sampled)
    recalls.append(recall_sampled)

    # 最佳模型选择
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        best_acc_model = cnn_model.state_dict().copy()


    print(f"{val_accuracy:.4f} {val_precision:.4f} {val_recall:.4f} {val_f1:.4f} {val_roc:.4f} {val_pr_auc:.4f}")

mean_precision = np.mean(precisions, axis=0)
mean_recall = np.mean(recalls, axis=0)
mean_fpr = np.mean(fprs, axis=0)
mean_tpr = np.mean(tprs, axis=0)


val_pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
val_pr_curve_data.to_csv('Draw/deeplearning/PR/classifier_5cv.csv', index=False)

val_roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
val_roc_curve_data.to_csv('Draw/deeplearning/ROC/classifier_5cv.csv', index=False)

torch.save(best_acc_model, "Draw/deeplearning/best_model/classifier.pt")
