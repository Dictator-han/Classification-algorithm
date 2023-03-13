from sqlalchemy import create_engine
import pandas as pd
import pymysql
import os
pymysql.install_as_MySQLdb()

# create_engine连接，daraframe整体导入到mysql
# engine = create_engine("mysql://suzhou02:6e48a702@172.16.206.101:8836/suzhou",echo=True)
## read_excel 读取 excel，生成dataframe
##！ read_excel结果直接是dataframe数据结构，xlrd读取后还要循环遍历重组成dataframe数据结构
# for root,ds,fs in os.walk('D:\摘星\数据导入1021',):
#     for f in fs:
#         file = 'D:\摘星\数据导入1021\%s'%f
#         df = pd.read_excel(file, sheet_name='批量查询导出数据',usecols=[0,14,18], skiprows=[0, 1])
#         # print(df)
#         df.to_sql("bayes_middle_test",con=engine,index=False,if_exists='append')
#         print('success')

# 导入down公司名
