# -*- coding:utf-8 -*-

import pandas
import pandas as pd
import os
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer # 向量化的类方法
from sklearn.decomposition import PCA # 主成分方法
from sklearn.model_selection import train_test_split # 切割数据---train + test
from sklearn import preprocessing # 结果评估

# file = 'D:\\摘星\\训练集\\bayes_train_final.xlsx'
# df = pd.read_excel(file, sheet_name='bayes_train_final')
# x = df.values[0:5000,2:3]
# y = df.values[0:5000:,1]
# # 编码 label
# encoder = LabelEncoder()
# encoder = encoder.fit([
# '普通运输',
# '妇产科',
# '男科',
# '内科',
# '皮肤科',
# '外科',
# '中医科',
# '耳鼻咽喉科',
# '眼科',
# '口腔科',
# '儿科',
# '体检科',
# '精神科',
# '生殖医学科',
# '戒毒科',
# '康复科',
# '综合医院',
# '医疗周边服务',
# '整形美容综合',
# '整形修复外科',
# '皮肤美容',
# '面部整形',
# '美体塑形',
# '纹身',
# '植发',
# '口腔美容',
# '眼科美容',
# '医美服务平台',
# '整形美容-其他',
# '一类医疗器械',
# '二类、三类医疗器械',
# '医疗器械-其他',
# '药品',
# '药房',
# '医药电商平台',
# '兽药',
# '保健品',
# '通用机械设备',
# '建筑工程机械',
# '清洁及通风设备',
# '机床机械',
# '物流设备',
# '食品机械',
# '机械设备-其他',
# '五金配件',
# '机械设备信息平台',
# '咨询调查',
# '商业检修',
# '检测认证',
# '法律服务',
# '技术服务',
# '拍卖',
# '代理代办',
# '出国移民',
# '签证服务',
# '招聘/人才中介',
# '包装印刷',
# '广告服务',
# '安全安保',
# '商务服务-综合平台',
# '团建拓展',
# '商务服务-其他',
# '开锁配钥',
# '刻章办证',
# '便民回收',
# '便民充值',
# '家政服务',
# '月子中心（非医疗机构）',
# '居民维修',
# '摄影婚庆',
# '餐饮',
# '宠物医院',
# '宠物周边',
# '婚恋相亲',
# '室内娱乐',
# '户外娱乐',
# '体育演出场馆',
# '生活美容',
# '养生保健',
# '文玩收藏',
# '心理援助',
# '彩票',
# '生活服务-综合平台',
# '生活服务-其他',
# '影音动漫',
# '传统媒体',
# '文书期刊',
# '资讯平台',
# '视频平台',
# '直播平台',
# '小说阅读',
# '自媒体',
# '文娱票务',
# '活动演出',
# '文娱传媒-其他',
# '办公设备及器械',
# '文教具',
# '办公文教-其他',
# '体育器械',
# '音乐器械',
# '玩具模型',
# '娱乐器械',
# '户外装备',
# '摩托车',
# '电动车',
# '航铁船',
# '非机动车',
# '便民出行',
# '车辆平台',
# '交通出行-其他',
# '汽车厂商',
# '汽车经销商',
# '汽配及服务',
# '物流运输',
# '特殊运输',
# '快递运输',
# '物流业-其他',
# '礼品',
# '日化用品',
# '一般化妆品',
# '特殊用途化妆品',
# '成人用品',
# '日用消费品-其他',
# '粮油米面',
# '生鲜',
# '速食',
# '烟酒',
# '乳制品及乳制品饮料',
# '休闲零食',
# '调味品',
# '饮料冲调',
# '营养品',
# '特殊医学用途配方食品',
# '食品饮料综合',
# '食品饮料-其他',
# '奶粉',
# '辅食',
# '婴儿用品',
# '孕妇用品',
# '母婴服饰',
# '母婴用品综合',
# '母婴用品-其他',
# '银行业',
# '证券业',
# '基金业',
# '保险业',
# '期货业',
# '信托业',
# '金融征信',
# '外汇类',
# '小额贷款',
# '典当',
# '担保及保理',
# '租赁业',
# '平台中介',
# '第三方支付',
# '网络借贷服务',
# '金融门户网站',
# '金融服务-其他',
# '手机数码',
# '电脑办公',
# '电器',
# 'IT/消费电子交易',
# 'IT/消费电子-其他',
# '多媒体处理',
# '社交通讯',
# '商用软件',
# '实用工具',
# '软件平台',
# '页游端游',
# '手机游戏',
# '休闲益智',
# '桌游卡牌',
# '游戏平台',
# '游戏周边',
# '开发服务',
# '游戏-其他',
# '早教',
# 'K12教育',
# '学历教育',
# '职业培训',
# '语言培训',
# '出国留学',
# '兴趣培训',
# '特殊教育',
# '教育培训综合',
# '教育培训-其他',
# '旅游局',
# '景点',
# '酒店',
# '旅行社',
# '商旅票务',
# '航空公司',
# '在线旅游',
# '旅游服务-其他',
# '服装鞋帽',
# '珠宝饰品',
# '眼镜',
# '钟表',
# '箱包皮具',
# '奢侈品',
# '电商B2B',
# '电商B2C',
# '二类电商',
# '实体零售',
# '商品信息平台',
# '房地产开发商',
# '房产中介',
# '物业管理',
# '装修建材',
# '装潢装修',
# '家具家居',
# '装修服务平台',
# '房产家居-其他',
# '电子器件',
# '仪器仪表',
# '智能制造',
# '电工电气',
# '电信运营商',
# '虚拟运营商',
# '通信及网络设备',
# '通信-其他',
# '网站建设',
# '域名空间',
# '云服务',
# '系统集成',
# '网络营销',
# '网络服务-其他',
# '农业',
# '林业',
# '渔业',
# '畜牧业',
# '化肥及农药',
# '农林牧渔-其他',
# '政府政务',
# '社会组织',
# '市政建设',
# '宗教',
# '化工原料',
# '矿产资源',
# '矿产及化工制品',
# '消毒产品',
# '危险化学品',
# '能源',
# '污染处理',
# '废旧回收',
# '节能',
# '化工及能源-其他',
# '招商-餐饮酒店',
# '招商-教育培训',
# '招商-服务类',
# '招商-家居建材',
# '招商-服装鞋帽',
# '招商-礼品饰品',
# '招商-美容化妆',
# '招商-休闲娱乐',
# '招商-生活用品',
# '招商加盟联展平台',
# '招商-其他',
# '其他'])
# y = encoder.transform(y)
# print(encoder.inverse_transform([24,27,200])) # 反编码

# 文本分词、清洗
# 将所有excel数据存放在列表中
class InitialProcess(object):

    # df_test = pd.read_excel()
    # 分词
    def fenci(self):
        ls_train_label = []
        ls_train_trade = []
        df_train = pd.read_excel('D:\\摘星\\训练集\\bayes_train_final.xlsx')
        df_train_label = df_train['meg_cust_trade_2']
        df_train_trade = df_train['ai_cust_business_scope']
        for each in df_train_label[0:50]:
            ls_train_label.append(each)
        print(ls_train_label)
        for each in df_train_trade[0:50]:
            cut1 = re.sub('(（.*?）)', '', each).replace('经营', '').replace('其', '').replace('是', '').replace('项目', '').replace('为', '').replace('一般', '').replace('经营范围', '').replace('一般项目', '').replace('服务', '').replace('许可项目', '').replace('及', '').replace('与', '').replace('和', '').replace('或', '').replace('的', '').replace('许可', '').replace('经营项目', '')
            cut2 = re.sub('(\(.*?\))', '', cut1)
            cut3 = re.sub('(【.*?】)', '', cut2)
            cut4 = re.sub(r'[^\u4e00-\u9fa5]','',cut3)
            cut_word = jieba.lcut(cut4,cut_all = False)[0:20]
            ls_train_trade = ls_train_trade + cut_word
        print(ls_train_trade)
            # ls_train_trade = ls_train_trade.append(cut_word[:20]) # 取前二十关键字
            # print(ls_train_trade)
        # max_f = 300
        # tfidf = TfidfVectorizer()
        # retfidf = tfidf.fit_transform(ls_train_trade)
        # input_data_matrix = retfidf.toarray()
        # x_train, x_test, y_train, y_test = train_test_split(input_data_matrix, test_size=0.2,random_state=400)
            # 将上面的数据标准化处理
            # -----------------------
            # stander=preprocessing.StandardScaler()
            # x_train = stander.fit_transform(x_train)
            # x_test = stander.transform(x_test)
            # max_min = preprocessing.MinMaxScaler()
            # x_train = max_min.fit_transform(x_train)
            # x_test = max_min.fit_transform(x_test)
            # # ---------------------
            # # 主成分降维
            # pca = PCA(n_components=max_f)
            # x_train, x_test = pca.fit_transform(x_train), pca.transform(x_test)
            # return x_train, x_test, y_train, y_test


        # return ls_train_trade

IniPro = InitialProcess()
IniPro.fenci()



# 读取excel文件,经营范围分词
# stop_word = {}.fromkeys(['许可','项目','的','及','与',':','：',';','；','、',' ',',',':'])
# for root,ds,fs in os.walk('D:\摘星\训练集',):
#     for f in fs:
#         file = 'D:\摘星\训练集\%s'%f
#         df = pd.read_excel(file, sheet_name='bayes_train_final',usecols=[1,3])
#         df_trade = df['ai_cust_business_scope']
#         for each in df:
            # print(each)
            #中文标点符号 ('[\u4e00-\u9fa5]', '', cut1)
            # 文本清洗
            # cut1 = re.sub('(（.*?）)', '', each)
            # cut2 = re.sub('(\(.*?\))', '', cut1).replace('为', '').replace('一般', '').replace(':', '').replace('经营范围', '').replace('一般项目', '').replace('服务', '').replace('许可项目', '').replace(' ', '').replace('及', '').replace('。', '').replace('；', '').replace('：', '').replace('，', '').replace('、', '').replace('与', '').replace('和', '').replace('或', '').replace('的', '').replace('许可', '').replace('经营项目', '')
            # cut_word = jieba.lcut(cut2,cut_all = False)
            # print(cut_word)
            # df_train = pandas.DataFrame()
            # df_train['label'] = df['meg_cust_trade_2']
            # df_train['des'] = cut_word

