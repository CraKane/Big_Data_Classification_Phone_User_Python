"""
@project = mnist_handwriting_pytorch
@file = train_model
@author = 10374
@create_time = 2020/02/03 13:45
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import pandas as pd
from torchvision.datasets import mnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils import data
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# define the parameters
train_batch_size = 32
test_batch_size = 128
learning_rate = 0.01
num_epoches = 10
momentum = 0.5


dir_name = 'F:/编程各种文件/part_time/20200205/data/'
X_train = pd.read_csv(dir_name + 'train_x.csv',encoding='utf-8', low_memory=False)
y_train = pd.read_csv(dir_name + 'train_label_x.csv',encoding='utf-8', low_memory=False)

print("read over!!!")
# 自定义torch数据集
class SDataset(data.Dataset):
	def __init__(self, data_, label_):
		self.Data = data_
		# print(self.Data)
		self.label = label_

	def __getitem__(self, item):
		tdata = th.from_numpy(self.Data[item])
		label = th.tensor(self.label[item])
		return tdata, label

	def __len__(self):
		return len(self.Data)


# definition of network
class Net(nn.Module):
	"""
	使用sequential构建网络，该函数的功能是将网络的层组合到一起
	"""
	def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
		super(Net, self).__init__()
		self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1))
		self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2))
		self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = self.layer3(x)
		return x

## 第二级的多分类
# TODO: complete the network
idx = np.where(y_train != 2)[0]
X_train_2 = X_train.iloc[idx, :]
y_train_2 = y_train.iloc[idx]


X_train_2_tmp = np.array(X_train_2)
print(X_train_2_tmp)
y_train_2_tmp = np.array(y_train_2)

for i in range(len(y_train_2_tmp)):
	y_train_2_tmp[i] -= 1
	if y_train_2_tmp[i] == -1:
		y_train_2_tmp[i] = 1
# 准备训练和测试数据，适应网络的

print(y_train_2_tmp)
X_key = X_train_2.keys()
x_train = SDataset(X_train_2_tmp, y_train_2_tmp)
x_loader = data.DataLoader(x_train, batch_size=train_batch_size)
x_train_loader = data.DataLoader(x_train, batch_size=len(x_train))
feature_num = X_train_2.shape[1]

# 实例化网络
# define the processing device
device = th.device("cpu")

model = Net(feature_num, 300, 100, 4)

# 将网络导入到cpu内存
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# X_train_2 = pd.DataFrame(X_train_2,index=[i for i in range(len(X_train_2))])
# y_train_2 = pd.DataFrame(y_train_2,index=[i for i in range(len(y_train_2))])
print("training start!!")
starttime = datetime.now()
for epoch in range(num_epoches):
	print('train_round {}'.format(epoch))
	model.train()
	if epoch % 5 == 0:
		optimizer.param_groups[0]['lr'] *= 0.9

	for dad, label in x_loader:
		dad = dad.to(device)
		label = label.to(device)

		# forward
		dad = th.tensor(dad, dtype=th.float32)
		out = model(dad).squeeze()
		label = label.squeeze()
		loss = criterion(out, label)
		print(loss)

		# backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# calculate the accuracy in classification
		_, pred = out.max(1)

endtime = datetime.now()
print("耗时：%d" % (endtime - starttime).seconds + "s")
# test
print('test here!!!\n')

model.eval()
y_pred_2 = []
for dad, label in x_train_loader:
	dad = dad.to(device)
	label = label.to(device)

	# forward
	dad = th.tensor(dad, dtype=th.float32)
	out = model(dad).squeeze()
	label = label.squeeze()
	loss = criterion(out, label)

	# calculate the accuracy in classification
	_, pred = out.max(1)
	print(pred)

y_pred_2 = pred
print(y_pred_2)
# 计算正确率
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def print_metrics(y_true, y_pred):
	f1 = f1_score(y_true, y_pred, average='macro')
	p = precision_score(y_true, y_pred, average='macro')
	r = recall_score(y_true, y_pred, average='macro')
	print('f1',f1)
	print('p',p)
	print('r',r)

	from sklearn.metrics import confusion_matrix

	cm = confusion_matrix(y_true, y_pred)
	print('confusion matrix', cm)
	cm = cm / cm.sum(axis=0, keepdims=True)
	print('confusion matrix', cm)
# 对训练集进行预测
print_metrics(y_train_2, y_pred_2)
