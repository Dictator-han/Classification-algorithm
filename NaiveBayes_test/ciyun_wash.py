#-*- coding:utf-8 -*-

from sqlalchemy import create_engine
import pandas as pd
import pymysql
import os
pymysql.install_as_MySQLdb()
import  re
import jieba
from jieba import analyse

# 1读取数据库sql
# engine = create_engine("mysql://suzhou02:6e48a702@172.16.206.101:8836/suzhou",echo=True)
#
#
# connection = pymysql.connect(host='172.16.206.101',
#                              port=8836,
#                              user='suzhou02',
#                              password='6e48a702',
#                              cursorclass=pymysql.cursors.DictCursor)
# with connection:
#     with connection.cursor() as cursor:
#         sql = (
#             "SELECT 企业名称,经营范围,词云 FROM suzhou.`company_tt` limit 50000"
#         )
#         data = pd.read_sql(sql,connection)
#         df =pd.DataFrame(data)
#         print('yes')

# 1读取excel文件进行wash
df = pd.read_excel('D:\\摘星\\中文文本分类\\train_meg_1_wash.xlsx')

#将大df切分成10个小的df分别进行wash
# df1 = df[0:10]
# print(df1[['企业名称','词云']])
# df2 = df[10:20000]
# df3 = df[20000:30000]
# df4 = df[30000:40000]
# df5 = df[40000:50000]
# df6 = df[50000:60000]
# df7 = df[60000:70000]
# df8 = df[70000:80000]
# df9 = df[80000:90000]
# df10 = df[90000:]

#jieba自定义词典
def wash(df):
    # jieba.add_word('非创伤性').add_word('乳制品（含婴幼儿配方乳粉）')
    i = 0
    for each in df['content']:
        cut1 = re.sub('(（.*?）)', '', each)
        cut2 = re.sub('(\[.*?])', '', cut1)
        cut3 = re.sub('(【.*?】)', '', cut2)
        # cut4 = re.sub('(\(.*?\))', '', cut3)\
        cut4 = cut3\
            .replace('项目', '').replace('一般', '').replace('经营', '').replace('范围', '').replace('许可', '')\
            .replace('技术开发', '').replace('技术研发', '').replace('技术转让', '').replace('技术咨询', '').replace('技术培训', '').replace('技术推广', '').replace('技术', '')\
            .replace('、', '').replace('购销', '').replace('其', '').replace('是', '').replace('为', '').replace('服务', '').replace('加工', '').replace('制造', '').replace('生产', '').replace('零售', '')\
            .replace('及','').replace('与', '').replace('业务', '').replace('和', '').replace('或', '').replace('的', '').replace('许可', '').replace('经营项目', '')\
            .replace('从事', '').replace('产品', '').replace('销售', '')\
            .replace('相关', '').replace('自产', '').replace('有关', '').replace('提供', '').replace('相应', '').replace('同类','').replace('并', '').replace('等', '').replace('用', '')
        cut5 = re.sub(r'[^\u4e00-\u9fa5]', '', cut4)\
        # cut_word = jieba.lcut(cut4, cut_all=False)
        cut_word = jieba.analyse.extract_tags(cut4) #topK=6
        df['content'][i] = cut5
        # print(df['经营范围2'][i])
        # str = ''
        # for each in df['经营范围2'][i]:
        #     str = str + each + "，"
        #     df['经营范围2'][i] = str
        i = i+1
    print(df['content'])
    df.to_excel('D:\\摘星\\中文文本分类\\train_meg_1_wash_done.xlsx')
    # print(df[['企业名称','词云']])#.to_sql("suzhou.company_ciyun",con=engine,index=False,if_exists='append')
    # df[['企业名称','词云']].to_sql("suzhou.company_ciyun",con=engine,index=False,if_exists='append')

wash(df)