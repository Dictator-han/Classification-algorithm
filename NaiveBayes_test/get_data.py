# 方法 1 pymsql.connect建立连接，遍历每条insert into
# # 每行读取写入 Mysql
# ## 读取excel文件数据
# book = xlrd.open_workbook("D:\摘星\数据导入1021\批量查询数据导出-【爱企查】-D71218798971632708208.xls")
# sheet = book.sheet_by_name('批量查询导出数据')
# print("数据行数：",sheet.nrows,"---","数据列数：",sheet.ncols)
# std = []
# for i in range(1,sheet.nrows):
#     temp1 = sheet.row_values(i)[0]
#     temp2 = sheet.row_values(i)[14]
#     temp3 = sheet.row_values(i)[18]
#     # print(temp1,temp2,temp3)
# ## 将数据导入到 mysql
#     connection = pymysql.connect(host = '172.16.206.101',
#                                  port = 8836,
#                                  user = 'suzhou02',
#                                  password = '6e48a702',
#                                  cursorclass=pymysql.cursors.DictCursor)
#     with connection:
#         with connection.cursor() as cursor:
#             sql = (
#                     "INSERT ignore into\n" +
#                     "suzhou.bayes_middle_test\n" +
#                     "(\n" +
#                     "'{0}','{1}','{2}'\n" .format(temp1,temp2,temp3) +
#                     ")"
#             )


# 方法 2 读取excel将dataframe整体导入到 Mysql
# ————————————————————————————————————————————————————————————————————————————————————————————————————
# down的数据整理入库
from sqlalchemy import create_engine
import pandas as pd
import pymysql
import os
pymysql.install_as_MySQLdb()


# create_engine连接，daraframe整体导入到mysql
engine = create_engine("mysql://suzhou02:6e48a702@172.16.206.101:8836/suzhou",echo=True)
## read_excel 读取 excel，生成dataframe
##！ read_excel结果直接是dataframe数据结构，xlrd读取后还要循环遍历重组成dataframe数据结构
# for root,ds,fs in os.walk('D:\\2\\男妇科'):
#     for f in fs:
# 紫荆.xlsx
file = 'D:\\2\\男妇科\\紫荆.xlsx'
df = pd.read_excel(file,skiprows=0)
df1 = df[0:100000]
df2 = df[100000:]
try:
    df1.to_sql("sz_yl",con=engine,index=False,if_exists='append')
    print('success')
    df2.to_sql("sz_yl",con=engine,index=False,if_exists='append')
    print('success')
except:
    print(1)

# 导入down公司名
# engine = create_engine("mysql://suzhou02:6e48a702@172.16.206.101:8836/suzhou",echo=True)
# for root,ds,fs in os.walk('D:\\1\\晓明\\男妇科'):
#     for f in fs:
#         try:
#             df = pd.read_excel(f)
#             df1 = df.iloc[0:50000,:]
#             df2 = df.iloc[50000:,:]
#             df1.to_sql("sz_yl",con=engine,index=False,if_exists='append')
#             df2.to_sql("sz_yl",con=engine,index=False,if_exists='append')
#             print('done')
#         except:
#             print(f)
# engine.dispose()

# ——————————————————————————————————————————————————————————————————————————————————————————
#导入有品牌的公司-品牌页面链接
# import os
# import openpyxl
# import xlrd
#
# workbook = openpyxl.load_workbook('D:\\摘星\\有商标企业名单\\品牌汇总.xlsx')
# sheet = workbook['Sheet1']
# # xlrd只能操作xls文件，并且不能通过修改文件名把xlsx文件改成xls，要打开文件点击另存，选择xls格式方可
# # for root,ds,fs in os.walk('D:\\摘星\\有商标企业名单'):
# #     for f in fs:
# # file = 'D:\摘星\有商标企业名单\%s'%f
# file = 'D:\\摘星\\有商标企业名单\\有商标-管理有限公司.xls'
# print(file)
# main_book = xlrd.open_workbook(file)
# main_sheet = main_book.sheet_by_index(0)
# row = main_sheet.nrows  # 总行数
# for row in range(row):
#     rowvalues = main_sheet.row_values(row,start_colx=0,end_colx=1)
#     # print(rowvalues)
#     name = rowvalues[0]
#     url0 = main_sheet.hyperlink_map.get((row, 0))
#     # print(url0)
#     url = '' if url0 is None else url0.url_or_path
#     url_gai = url.replace('fr=excel','tab=certRecord')
#     print(url_gai)
#     sheet.append([name,url_gai])
# workbook.save('D:\\摘星\\有商标企业名单\\品牌汇总.xlsx')