import pandas as pd # 用pandas包读取数据集
import matplotlib.pylab as plt # 画图
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, learning_curve, KFold, \
    validation_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm, __all__
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score

# 读取文件
df = pd.read_csv('./Data.csv',sep=',',header=0) # 用dataframe数据结构（表格行列式）存储数据 df是使用dataframe结构的对象 数据路径 分隔符号 表头是第0行
# 数据预处理
df = df.drop(columns = ["Patient index"]) # 删除无意义的ID列
df = df[df['Label'] != 2] # 删除噪音‘2’
result = df.iloc[:,-1].values # 保留真实label之后做模型准确率比对

data = df.values
corr_matrix = df.corr()
print(corr_matrix["Label"].sort_values(ascending=False))


data = df.iloc[:, 0:15].values # 提取表格值 以ndarray数组形式存储在data里 所有行，0-14列
print(df.iloc[:, 0:15].corr(method='spearman')) # 非正态分布用spearman相关分析 并生成热力图
figure, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(df.iloc[:, 0:15].corr(), square=True,linecolor='white', annot=True) # 小方格展示 方格内展示数字
plt.title('Correlation Heatmap')
plt.show()

# 数据降维
newdata = df.loc[:,['F14','F4','F10','F9','F12','F15']].values
pca = PCA(n_components=13) # 实例化PCA 自定义维数
sdata = pca.fit_transform(data)
var = pca.explained_variance_
var_r = pca.explained_variance_ratio_
t_var_r = np.cumsum(var_r)
plt.figure(figsize=(10, 10), dpi=75)
plt.title("Variance")
plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], var_r)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], ['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8','PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15'],fontsize=12)
plt.xlabel('Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

score = pca.score_samples(data)
plt.figure()
plt.title('Mahalanobis Distance')
plt.scatter(range(len(score)), score,edgecolors='white',linewidths=0.2)
plt.show()
plt.figure()
plt.title('PCA-bias removal')
plt.scatter(sdata[:, 0], sdata[:, 1], marker='o', c=score < -15)
plt.show()

plt.plot(range(1,len(var_r)+1),t_var_r,marker = 'o',linestyle = '--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Cumulative Sum of Explained Variance Ratio')
plt.show()
my_pal = sns.color_palette(['blue','red'])
sns.scatterplot(data=pd.DataFrame(sdata,columns=['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8','PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15']),x='PC1',y='PC2',hue=df['Label'],palette=my_pal)
plt.title('PCA')
plt.show()

# SVM分类器
y = result
x_train, x_test, y_train, y_test = train_test_split(sdata, y, test_size=0.3, random_state=0) # 固定r_s的值 每次可以分割得到同样的训练集和测试集 重复展现相同的结果
classifier = svm.SVC() # 非线性数据用rbf 惩罚系数过高会过拟合 过低不准确
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)
kfold = KFold(n_splits=10,shuffle=False)
scores = cross_val_score(classifier,x_train,y_train,cv=kfold) # 评估分数可以反应准确率 用F1得分作为评估指标
print(scores.mean())
scores = cross_val_score(classifier,x_test,y_test,cv=kfold)
print(scores.mean())
def plot_svc_decision_function(model, ax=None):
    if ax == None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    axisx = np.linspace(xlim[0], xlim[1], 30)
    axisy = np.linspace(ylim[0], ylim[1], 30)

    axisx, axisy = np.meshgrid(axisx, axisy)
    xy = np.vstack([axisx.ravel(), axisy.ravel()]).T

    Z = model.decision_function(xy).reshape(axisx.shape)

    ax.contour(axisx, axisy, Z
               , colors='k'
               , levels=[-1, 0, 1]
               , linestyles=['--', '-', '--']
               )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.figure(figsize = (20,6))
plt.subplot(1,2,1)
plt.scatter(sdata[:,0],sdata[:,1],c=y,cmap='rainbow',s=5)
plot_svc_decision_function(classifier)
plt.scatter(classifier.support_vectors_[:,0],classifier.support_vectors_[:,1],facecolor = 'none',edgecolor = 'k')
plt.show()
# 学习曲线
pipe_lr = Pipeline([('scl',StandardScaler()),('clf',SVC())])
train_sizes,train_scores,test_scores=learning_curve(estimator= pipe_lr ,X=x_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=5,n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.ylim([0.0,1.0])
plt.show()
# 验证曲线
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
# 10折，验证正则化参数C
train_scores,test_scores = validation_curve(estimator=pipe_lr,X=x_train,y=y_train,param_name='clf__C',param_range = param_range,cv=10)
# 统计结果
train_mean= np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean =np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color='c',marker='o',markersize=5,label='training accuracy')
plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='c')
plt.plot(param_range,test_mean,color='red',linestyle='--',marker='s',markersize=5,label='test accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='red')
plt.grid()
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Validation Curve')
plt.ylim([0.0,1.0])
plt.show()

print(accuracy_score(y_test,predictions))
print("Accuracy of train set in SVM:", classifier.score(x_train, y_train))
print("Accuracy of test set in SVM:", classifier.score(x_test, y_test))
plt.figure(0).clf()
fpr,tpr,threshold = roc_curve(y_test,predictions)
roc_auc = auc(fpr,tpr)
lw = 2
plt.plot(fpr,tpr,label='SVM, AUC= %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label = 'Random, AUC = 0.50')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
# 建立图像坐标
axis = plt.gca()
xlim = axis.get_xlim()
ylim = axis.get_ylim()
# 生成两个等差数列
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# print("xx:", xx)
# print("yy:", yy)
X, Y = np.meshgrid(xx, yy)
# print("X:", X)
# print("Y:", Y)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = classifier.decision_function(xy).reshape(X.shape)
# 画出分界线
axis.contour(X, Y, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
axis.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, linewidths=1, facecolors='none')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('SVM')
plt.show()

# 决策树
clf = tree.DecisionTreeClassifier()
clf.fit(x_train,y_train)
kfold = KFold(n_splits=10,shuffle=False)
scores = cross_val_score(clf,x_train,y_train,cv=kfold)
print(scores.mean())
scores = cross_val_score(clf,x_test,y_test,cv=kfold)
print(scores.mean())
print("Accuracy of train set in Decision Tree:", clf.score(x_train, y_train))
print("Accuracy of test set in Decision Tree:", clf.score(x_test, y_test))
predictions = clf.predict(x_test)
fpr,tpr,threshold = roc_curve(y_test,predictions)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='Decision Tree, AUC= %0.2f' % roc_auc)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf)
plt.show()

# 贝叶斯
clf = GaussianNB()
clf.fit(x_train,y_train)
kfold = KFold(n_splits=10,shuffle=False)
scores = cross_val_score(clf,x_train,y_train,cv=kfold)
print(scores.mean())
scores = cross_val_score(clf,x_test,y_test,cv=kfold)
print(scores.mean())
print("Accuracy of train set in GNB:", clf.score(x_train, y_train))
print("Accuracy of test set in GNB:", clf.score(x_test, y_test))
predictions = clf.predict(x_test)
fpr,tpr,threshold = roc_curve(y_test,predictions)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='Bayes, AUC= %0.2f' % roc_auc)
cm = confusion_matrix(y_test,predictions)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix of Bayes')
plt.show()

# 神经网络
mlp = MLPClassifier() # 隐藏层神经元是输入输出层和的2/3 数据集较大时用adam梯度下降进行优化
mlp.fit(x_train,y_train)
kfold = KFold(n_splits=10,shuffle=False)
scores = cross_val_score(mlp,x_train,y_train,cv=kfold)
print(scores.mean())
scores = cross_val_score(mlp,x_test,y_test,cv=kfold)
print(scores.mean())
print("Accuracy of train set in BP:", mlp.score(x_train, y_train))
print("Accuracy of test set in BP:", mlp.score(x_test, y_test))
predictions = mlp.predict(x_test)
fpr,tpr,threshold = roc_curve(y_test,predictions)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='DNN, AUC= %0.2f' % roc_auc)

# 逻辑回归
lr = LogisticRegression()
lr.fit(x_train,y_train)
kfold = KFold(n_splits=10,shuffle=False)
scores = cross_val_score(lr,x_train,y_train,cv=kfold)
print(scores.mean())
scores = cross_val_score(lr,x_test,y_test,cv=kfold)
print(scores.mean())
print("Accuracy of train set in LR:", lr.score(x_train, y_train))
print("Accuracy of test set in LR:", lr.score(x_test, y_test))
predictions = lr.predict(x_test)
fpr,tpr,threshold = roc_curve(y_test,predictions)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='Logistic Regression, AUC= %0.2f' % roc_auc)
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# 无监督聚类 k=3时轮廓系数最大
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(sdata)
k = 3
label_p = k_means.predict(sdata)
plt.scatter(sdata[:,0],sdata[:,1],c=label_p,edgecolors='white',linewidths=0.5)
plt.title('K-Means')
plt.text(.99, .01, ('k=%d' % k), transform=plt.gca().transAxes, size=10, horizontalalignment='right')
plt.show()
# 不同k的图像
for index, k in enumerate((3,4,5,6)):
    plt.subplot(2,2,index+1)
    y_pre = KMeans(n_clusters=k, random_state=0).fit_predict(sdata)
    plt.scatter(sdata[:, 0], sdata[:, 1], c=y_pre,edgecolors='white',linewidths=0.2)
    plt.text(.99, .01, ('k=%d' % k), transform=plt.gca().transAxes, size=10, horizontalalignment='right')
plt.show()
# 轮廓系数曲线
s_scores = []
for k in range(3,15):
    labels = KMeans(n_clusters=k).fit(sdata).labels_
    score = metrics.silhouette_score(sdata,labels)
    s_scores.append(score)
# 通过画图找出最合适的K值
plt.plot(list(range(3,15)),s_scores,marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score of Different K')
plt.show()
