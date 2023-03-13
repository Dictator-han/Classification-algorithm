import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from sklearn.model_selection import train_test_split
from transformers import Trainer,TrainingArguments,BertTokenizer,BertModel
from torch.utils.data import Dataset,DataLoader
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# print(train['text_len'].describe())
# print(train['text'][0])

# def fill_padding(data,max_len):
#     if len(data) < max_len:
#         pad_len = max_len - len(data)
#         padding = [0 for _ in range(pad_len)]
#         data = torch.tensor(data+padding)
#     else:
#         data = torch.tensor(data[:max_len])
#     return data

transform_dict = {
0:'医疗服务',
1:'整形美容',
2:'医疗器械',
3:'保健品/药品',
4:'机械设备',
5:'商务服务',
6:'生活服务',
7:'文娱传媒',
8:'文体器材',
9:'交通出行',
10:'物流业',
11:'日用消费品',
12:'食品饮料',
13:'母婴用品',
14:'金融服务',
15:'IT/消费电子',
16:'软件',
17:'游戏',
18:'教育培训',
19:'旅游服务',
20:'箱包服饰',
21:'商品交易',
22:'房产家居',
23:'电子电工',
24:'通信',
25:'网络服务',
26:'农林牧渔',
27:'社会公共',
28:'化工及能源',
29:'招商加盟',
30:'其他'
}

def Label2num(dict,Label):
    for k,v in dict.items():
        if Label == v:
            return k

def wash(df):
    i = 0
    for each in df['text']:
        cut1 = re.sub('(（.*?）)', '', each).replace('其他', '').replace('经营', '').replace('其', '').replace('是', '').replace('项目', '').replace(
            '为', '').replace('一般', '').replace('经营范围', '').replace('一般项目', '').replace('服务', '').replace('许可项目',
                                                                                                         '').replace('及',
                                                                                                                     '').replace(
            '与', '').replace('业务', '').replace('和', '').replace('或', '').replace('的', '').replace('许可', '').replace('经营项目', '')
        cut2 = re.sub('(\(.*?\))', '', cut1)
        cut3 = re.sub('(【.*?】)', '', cut2)
        cut4 = re.sub(r'[^\u4e00-\u9fa5]', '', cut3).replace('业','')
        # cut_word = jieba.lcut(cut4, cut_all=False)[0:20]
        df['text'][i] = cut4
        i = i+1
    return (df)

def read_fl(path):# 调整label、scope参数
    df = pd.read_excel(path)

    df['text'] = df['ai_cust_trade'] + '、' + df['ai_cust_business_scope']
    df['label'] = df['meg_cust_trade_1']
    df = pd.concat([df['text'],df['label']],axis=1,join='inner')

    return df

data_path = 'D:\摘星\中文文本分类\\train.xlsx'

data = read_fl(data_path)
data = wash(data)
data = data.replace(['医疗服务',
'整形美容',
'医疗器械',
'保健品/药品',
'机械设备',
'商务服务',
'生活服务',
'文娱传媒',
'文体器材',
'交通出行',
'物流业',
'日用消费品',
'食品饮料',
'母婴用品',
'金融服务',
'IT/消费电子',
'软件',
'游戏',
'教育培训',
'旅游服务',
'箱包服饰',
'商品交易',
'房产家居',
'电子电工',
'通信',
'网络服务',
'农林牧渔',
'社会公共',
'化工及能源',
'招商加盟',
'其他',
],[0,
1,
2,
3,
4,
5,
6,
7,
8,
9,
10,
11,
12,
13,
14,
15,
16,
17,
18,
19,
20,
21,
22,
23,
24,
25,
26,
27,
28,
29,
30,
])
# print(data)
class BertClassfication(nn.Module):
    def __init__(self):
        super(BertClassfication, self).__init__()
        self.model_name = 'hfl/chinese-bert-wwm'
        self.model = BertModel.from_pretrained(self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.fc = nn.Linear(768, 31)  # 768取决于BERT结构，2-layer, 768-hidden, 12-heads, 110M parameters

    def forward(self, x):  # 这里的输入是一个list
        batch_tokenized = self.tokenizer.batch_encode_plus(x, add_special_tokens=True,
                                                           max_length=148,
                                                           truncation=True,
                                                           pad_to_max_length=True)  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids'])
        attention_mask = torch.tensor(batch_tokenized['attention_mask'])
        hiden_outputs = self.model(input_ids, attention_mask=attention_mask)
        outputs = hiden_outputs[0][:, 0, :]  # [0]表示输出结果部分，[:,0,:]表示[CLS]对应的结果
        output = self.fc(outputs)
        return output
model = BertClassfication()

label = data['label'].values
text = data['text'].values
print(label)


train_text,test_text,train_label,test_label = train_test_split(text,label)
print(train_text)
batch_size = 64
batch_count = int(len(train_text) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    batch_train_inputs.append(train_text[i*batch_size : (i+1)*batch_size])
    batch_train_targets.append(train_label[i*batch_size : (i+1)*batch_size])

bertclassfication = BertClassfication()
lossfuction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bertclassfication.parameters(),lr=2e-5)
epoch = 4
batch_count = batch_count
print_every_batch = 4
for _ in range(epoch):
    print_avg_loss = 0
    for i in range(batch_count):
        inputs = batch_train_inputs[i]
        targets = torch.tensor(batch_train_targets[i])
        optimizer.zero_grad()
        outputs = bertclassfication(inputs)
        loss = lossfuction(outputs,targets)
        loss.backward()
        optimizer.step()

        print_avg_loss += loss.item()
        if i % print_every_batch == (print_every_batch-1):
            print("Batch: %d, Loss: %.4f" % ((i+1), print_avg_loss/print_every_batch))
            print_avg_loss = 0

hit = 0
total = len(test_text)
for i in range(total):
    outputs = model([test_text[i]])
    _,predict = torch.max(outputs,1)
    if predict==test_label[i]:
        hit+=1
print('准确率为%.4f'%(hit/len(test_label)))


result = model()
_,result = torch.max(result,1)
result = int(result)
print(transform_dict[result])

# 使用mutinomialNB
# for each in test_data:
#     print(str(each))
#     tf = TfidfVectorizer()
#     train_text = tf.fit_transform(train_data['text'])
#     train_label = tf.fit_transform(train_data['label'])
#     test_text = tf.transform(test_data['text'])
#     test_label = tf.transform(test_data['label'])
#     print(test_label)
# mlb = MultinomialNB()
# mlb.fit(train_text,train_label)
# y_predict = mlb.predict(test_label)
# rate = mlb.score(test_text,test_label)
# print(test_label)
# print(y_predict)
# for each in data:
#     str = ''
#     for string in each:
#         str = str+string
# print(str)
    # str = " ".join(each for x in data)
# print(train['label'].head(50))
# max_f = 10
# for each in data.iterrows():
#         print(each)
        # tfidf = TfidfVectorizer() # 继承类方法
        # retfidf = tfidf.fit_transform(i) # 拟合，将corpus转化为数值向量
        # input_data_matrix = retfidf.toarray() # 得到的矩阵
        # print(input_data_matrix)