from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time
from sklearn import metrics

time0=time.time()
file = open("F:/dataset1.txt", "r")
text = file.readlines() #返回列表，每个元素都是txt文件中的一行字符串
n_sample = len(text)# 样本数，也就是text列表的元素个数
for i in range(n_sample):
    text[i] = text[i].strip()#去掉每个元素首尾的空格，\n \t \r字符
    text[i] = text[i].split(",")#现在text变成一个二维列表，每一行有两个元素，分别是SMILE字符串和标签字符串
dataset = np.array(text)#二维数组
file.close()

feature = [] #列表，每个元素是一个分子样本的二进制表示，string型。
label = [] #列表，每个元素是一个分子样本的标签,int型。
n_feature=0 #特征数，也是二进制表示的最长长度
for i in range(n_sample):
    smi = dataset[i][0] #第i行第0列，是第i个样本的SMILE字符串
    m = Chem.MolFromSmiles(smi)
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    bits = fp.ToBitString() #第i个样本的SMILE字符串的二进制表示。
    n_feature=max(n_feature,len(bits))
    feature.append(bits)
    label.append(int(dataset[i][1]))

print("n_feature=",n_feature)
print("n_sample=",n_sample)
assert(len(feature)==len(label)==n_sample)#都是样本数量


#构造x_train数组，大小为n_sample*n_feature
#y_train数组,大小为n_sample*2，若标签为0，则这一行为（1,0）；若标签为1，则这一行为（0,1）
x_train=-1*torch.ones((n_sample,n_feature),dtype=np.int)
y_train=Variable(torch.tensor(label,dtype=torch.long))
print("未开始")
for i in range(n_sample):#每个样本
    for j in range(len(feature[i])):
        x_train[i][j]=float(feature[i][j])
x_train=Variable(x_train).float()
time1=time.time()

#超参数
N_HIDDEN=10
LEARNING_RATE=0.2

#定义网络
#输入n_sample*n_feature大小的x，经过hidden的全连接层、relu和output的全连接层，输出n_sample*n_output大小的x
class Model(nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Model,self).__init__()
        self.hidden=nn.Linear(n_feature,n_hidden)
        self.output=nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=nn.functional.relu(self.hidden(x))
        x=self.output(x)
        return x
model=Model(n_feature,N_HIDDEN,2) #输出大小为n_sample*2

#训练网络
optimizer=torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
loss_fun=nn.CrossEntropyLoss()
print("准备开始")

for i in range(201):
    print("开始")
    out=model(x_train)
    assert (out.shape==(n_sample,2))
    loss=loss_fun(out,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(i%2==0):
        # nn.functional.softmax(out)把每一行缩放成和为1且在0~1区间内的两个数字
        # torch.max(*,1)[1]返回的是偏大的那个数字的索引，第一个数字偏大为0，第二个数字偏大为1
        prediction=torch.max(nn.functional.softmax(out,dim=1),1)[1]
        y_pred=prediction.data.numpy().squeeze()
        y_target=y_train.data.numpy().squeeze()
        assert(y_pred.shape==y_target.shape)
        #print("y_pred=",y_pred,",y_target=",y_target)
        accuracy=sum(y_pred==y_target)/n_sample
        print("当前迭代第",i,"次,accuracy=",accuracy,"loss=",loss.data.numpy())

        fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_target, pos_label=1)
        print("auc=",metrics.auc(fpr, tpr))




time2=time.time()
print("数据导入、处理耗时：",time1-time0,",训练网络耗时：",time2-time1)




