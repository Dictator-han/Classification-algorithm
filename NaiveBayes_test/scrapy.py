import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import svm

# s = '生产、销售：保险柜，电子元件，电子器件，照明器具，音箱，电子门锁、五金锁具；经营和代理各类商品及技术的进出口业务（国家限定经营或禁止进出口的商品及技术除外，涉及技术许可证的必须凭有效许可证经营）；广告设计、发布。(依法须经批准的项目，经相关部门批准后方可开展经营活动)'
# cut = jieba.lcut(s,cut_all = False)
# temp = [v for v in cut if v is not '']
# print(temp)

s = ['生产','销售','保险柜','电子元件','电子器件','照明器具','音箱','电子门锁','五金锁具','经营和代理各类商品及技术的进出口业务']
x_test= ['软木制品','研发','生产','销售','栓皮','收购','自营和代理各类商品的进出口业务']
l = ['电子电工']
y_test = ['房产家居']
# print(temp)
max_f = 10
tfidf = TfidfVectorizer() # 继承类方法
retfidf = tfidf.fit_transform(s) # 拟合，将corpus转化为数值向量
retfidf_x = tfidf.fit_transform(l)
input_data_matrix = retfidf.toarray() # 得到的矩阵
input_data_matrix_y = retfidf_x.toarray() # 得到的矩阵
print(input_data_matrix)
print(input_data_matrix_y)

pca = PCA(n_components=max_f)
# pca2 = PCA(n_components=1)
x_train = pca.fit_transform(input_data_matrix)
# input_data_matrix_y = pca2.fit_transform(input_data_matrix_x)
print(x_train)



# 继承类，传入不同的核函数
clf_1 = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr') # 使用rbf径向基函数来讲低维数据转化为高维数据，使其可分
clf_2 = svm.SVC(C=1, kernel='linear', gamma=20, decision_function_shape='ovr')
clf_3 = svm.SVC(C=1, kernel='poly', gamma=20, decision_function_shape='ovr')
clf_4 = svm.SVC(C=1, kernel='sigmoid', gamma=20, decision_function_shape='ovr')

# 对应不同的核函数， 拟合
clf_1.fit(x_train, input_data_matrix_y)
clf_2.fit(x_train, input_data_matrix_y)
clf_3.fit(x_train, input_data_matrix_y)
clf_4.fit(x_train, input_data_matrix_y)

# 预测
y_pred_1 = clf_1.predict(x_test)
y_pred_2 = clf_2.predict(x_test)
y_pred_3 = clf_3.predict(x_test)
y_pred_4 = clf_4.predict(x_test)
print(y_pred_1)
print(y_pred_2)
print(y_pred_3)
print(y_pred_4)

# 分类评估
print(metrics.classification_report(y_test,y_pred_1))
print(metrics.classification_report(y_test,y_pred_2))
print(metrics.classification_report(y_test,y_pred_3))
print(metrics.classification_report(y_test,y_pred_4))

# 混淆矩阵
print(metrics.confusion_matrix(y_test,y_pred_1))
print(metrics.confusion_matrix(y_test,y_pred_2))
print(metrics.confusion_matrix(y_test,y_pred_3))
print(metrics.confusion_matrix(y_test,y_pred_4))