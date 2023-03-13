# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import jieba

import gensim
import os
import pandas as pd
import numpy as np
import pymysql
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import xlrd
import xlwt

# multinomialNB = MultinomialNB()
# multinomialNB.fit()

#分词+计算词频
# text = ''
# data2 = '一般项目：电力电子元器件销售；汽车零配件零售；第一类医疗器械销售；再生资源销售；半导体器件专用设备销售；机械设备销售；建筑材料销售；新材料技术研发；新兴能源技术研发；技术服务、技术开发、技术咨询、技术交流、技术转让、技术推广；会议及展览服务；创业投资（限投资未上市企业）；电气设备销售；办公设备销售；智能输配电及控制设备销售；通信设备销售；配电开关控制设备销售；电子元器件与机电组件设备销售；机械电气设备销售；计算器设备销售；制冷、空调设备销售；照明器具生产专用设备销售；新能源原动设备销售；家用电器销售；气体'
# data2 = data2.replace('依法须经批准的项目，经相关部门批准后方可开展经营活动','').replace('一般项目','').replace('经营范围','').replace('按《食品经营许可证》核定项目经营','').replace('限投资未上市企业','')
# wl = jieba.lcut(data2,)
# stopword = {}.fromkeys([';','；',',','：','，','。','（','）','、','及','与'])
# counts = {}
# for word in wl:
#     if word not in stopword:
#         text += word + " "
#         counts[word] = counts.get(word,0)+1
#     if len(word) == 1:
#         continue
# print(text)
# print(counts)

#影评实例
def get_dataset():
    data = []
    connection = pymysql.connect(host='172.16.206.101',
                                 port=8836,
                                 user='suzhou02',
                                 password='6e48a702',
                                 cursorclass=pymysql.cursors.DictCursor)
    with connection:
        with connection.cursor() as cursor:
            sql_train = (
                    "SELECT\n" +
                    "id,\n" +
                    "meg_cust_trade_1,\n" +
                    "meg_cust_trade_2,\n" +
                    "ai_cust_trade,\n" +
                    "ai_cust_business_scope\n" +
                    "FROM\n" +
                    "suzhou.bayes_train_test\n" +
                    "WHERE\n" +
                    "ai_cust_trade is not NULL\n" +
                    "AND\n" +
                    "ai_cust_business_scope != '-'\n" +
                    "limit\n" +
                    "45000"
            )
            global data_train,data_train
            data_train = pd.read_sql(sql_train,connection)
            # data_test = pd.read_sql(sql_test,connection)
            global train_x,train_y,test_x,test_y
            train_x = pd.DataFrame(data_train)['ai_cust_business_scope']
            train_y = pd.DataFrame(data_train)['meg_cust_trade_1']
            # test_x = pd.DataFrame(data_test)['ai_cust_business_scope']
            # test_y = pd.DataFrame(data_test)['meg_cust_trade_1']
    excelfile = xlrd.open_workbook(r'C:\Users\zhouxihan_suz\Downloads\批量查询数据导出-【爱企查】-D71218798971635299855.xls')
    test_x = excelfile.sheet_by_index(0).col_values(3)
    test_y = excelfile.sheet_by_index(0).col_values(0)

get_dataset()

stopwords = ['、','；']
# train_x_word = jieba.lcut(train_x.to_list())
# train_x_word = [word for word in train_x_word if len(word)>1]
# train_x_word = [word for word in train_x_word if word not in stopwords]
# print(train_x_word)

tf = TfidfVectorizer()
train_x = tf.fit_transform(train_x)
test_x = tf.transform(test_x)

mlb = MultinomialNB()
mlb.fit(train_x,train_y)
y_predict = mlb.predict(test_x)
rate = mlb.score(test_x,test_y)

print(test_y)
print(y_predict)

for i in y_predict:
    print(i)
for j in test_y:
    print(j)
# 写入excel
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')

print('预测准确率为',rate)