# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#################### train_df
df = pd.read_csv('train.csv')


'''
Title: [Mr Miss Mrs other] => [1 3 4 2]
'''
a=df.Name
l=[]
for a in df.Name:
  l.append(a.split(',')[1].split('.')[0].strip())
df["Title"]=pd.Series(l)

title_dict=\
{'Capt': 2,
 'Col': 2,
 'Don': 2,
 'Dr': 2,
 'Jonkheer': 2,
 'Lady': 2,
 'Major': 2,
 'Master': 2,
 'Miss': 3,
 'Mlle': 2,
 'Mme': 2,
 'Mr': 1,
 'Mrs': 4,
 'Ms': 2,
 'Rev': 2,
 'Sir': 2,
 'the Countess': 2}

df.Title = df.Title.map(title_dict).astype(int) 
#sns.factorplot('Title','Survived', data = df)#按生存率排序

'''
Sex: [female male] => [1  2]
'''
df.Sex=df.Sex.map({'female':1,'male':2}).astype(int)


'''
Embarked: [S  C  Q] => [1  2  3]  Pclass & Fare => predict_value => Nan
'''
df.Embarked=df.Embarked.map({'S':1,'C':2,'Q':3,np.nan:0}).astype(int)
df.loc[df.PassengerId==830,'Embarked']=1 #get from another correct dataset
df.loc[df.PassengerId==62,'Embarked']=2


'''
Family: ["SibSp","Parch"] => [alone small large]
'''
df["Family"]=df.SibSp+df.Parch

def Family_category(x):
    if x==0:
        return 1
    elif x==1:
        return 2
    elif x==2:
        return 2
    else:
        return 3
        
df.Family=df.Family.map(lambda x:Family_category(x))


'''
Age: [LogisticRegression] => [nan] 
'''
x=df[["Pclass","Sex","SibSp","Parch","Fare","Embarked","Title"]][df.Age.notnull()].values
y=df.Age[df.Age.notnull()].values
z=df[["Pclass","Sex","SibSp","Parch","Fare","Embarked","Title"]][df.Age.isnull()].values
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1,penalty='l2',tol=0.0001)
output = clf.fit(x,y).predict(z).astype(int)
df.Age.loc[df.Age.isnull()]=output


'''
train data processing
'''
#df=df.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin"],axis=1)
#df.Survived=df.Survived.map({1:1,0:-1})
train_label =df["Survived"].values.astype(int)
#train_data  =df[["Pclass","Sex","Age","Fare","Embarked","Title","Family"]].values.astype(int)
train_data  =df[["Pclass","Sex","Age","Fare","Embarked","Title","Family"]].values.astype(float)



#################### test_df
df = pd.read_csv('test.csv', header=0)


'''
Title: [Mr Miss Mrs other] => [1 2 3 4]
Title: [Mr Miss Mrs other] => [1 3 4 2]
'''
a=df.Name
l=[]
for a in df.Name:
  l.append(a.split(',')[1].split('.')[0].strip())
df["Title"]=pd.Series(l)

title_dict=\
{
 'Col': 2,
 'Dona': 2,
 'Dr': 2,
 'Master': 2,
 'Miss': 3,
 'Mr': 1,
 'Mrs': 4,
 'Ms': 2,
 'Rev': 2,
 'ha':5,
 }

df.Title = df.Title.map(title_dict).astype(int) 


'''
Fare: Pclass & Embarked => predict_value => Nan
'''
df.Fare.loc[df.Fare.isnull()]=df.Fare[df.Pclass==3][df.Embarked=='S'][df.Fare<35].median()


'''
Sex: [female male] => [1  2]
'''
df.Sex=df.Sex.map({'female':1,'male':2}).astype(int)


'''
Embarked: [S  C  Q] => [1  2  3]  Pclass & Fare => predict_value => Nan
'''
df.Embarked=df.Embarked.map({'S':1,'C':2,'Q':3,np.nan:0}).astype(int)
df.loc[df.PassengerId==830,'Embarked']=1 #get from another correct dataset
df.loc[df.PassengerId==62,'Embarked']=2


'''
Family: ["SibSp","Parch"] => [alone small large]
'''
df["Family"]=df.SibSp+df.Parch        
df.Family=df.Family.map(lambda x:Family_category(x))
df.Family


'''
Age: [LogisticRegression] => [nan] 
'''
z=df[["Pclass","Sex","SibSp","Parch","Fare","Embarked","Title"]][df.Age.isnull()].values
output = clf.predict(z).astype(int)
df.Age.loc[df.Age.isnull()]=output


'''
test data processing
'''
test_data =df[["Pclass","Sex","Age","Fare","Embarked","Title","Family"]].values.astype(float)




from sklearn import preprocessing
train_data=preprocessing.scale(train_data)
test_data=preprocessing.scale(test_data)
#===============数据准备好====================







#===============特征选择======================
print '单特征响应线性度比较(f值和p值):'
from scipy.stats import pearsonr
for i in range(train_data.shape[1]):
  print pearsonr(train_data[:,i], train_label)
'''
from sklearn.linear_model import RandomizedLogisticRegression
clf = RandomizedLogisticRegression(selection_threshold=0.25)
clf.fit(train_data, train_label)
print '有效特征选择:',clf.get_support()
train_data=train_data[:,clf.get_support()]
test_data=test_data[:,clf.get_support()]
'''


















'''
#===============正则化？？？======================
from sklearn.linear_model import Ridge
clf = Ridge(alpha=0.1)
clf.fit(train_data, train_label)
clf.score(train_data, train_label)
print 'ridge正则化系数:',clf.coef_

from sklearn.linear_model import Lasso
clf = Lasso(alpha=.3)
clf.fit(train_data, train_label)
print 'lasso正则化系数:',clf.coef_
#print "Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True)
#Y_predict=lasso.predict(train_data)
'''








#===============建立模型====================
print '**************************************************\n\
                    Training... \n\
**************************************************'
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=0.5,penalty='l2',tol=0.0001)#0.8137

#from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
#clf=GaussianNB() #0.7744  

#from sklearn.neighbors import KNeighborsClassifier  
#clf=KNeighborsClassifier(n_neighbors=8)#0.8081

#from sklearn import svm   
#clf = svm.SVC(C=10,gamma=0.1)#0.8238
#clf=GridSearchCV(svm.SVC(), param_grid={"C":np.logspace(-2, 10, 13),"gamma":np.logspace(-9, 3, 13)})
#output = clf.fit( train_data[0::,1::], train_data[0::,0] ).predict(test_data).astype(int)
#print("The best parameters are %s with a score of %0.2f"% (knnClf.best_params_, knnClf.best_score_))

#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=1)#0.778

#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(min_samples_split = 16, n_estimators = 300)#0.8350

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05,max_depth=2)#0.8395

#from sklearn.ensemble import ExtraTreesClassifier
#clf=ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=1, random_state=0)#0.7968

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=1000)#0.8115






#===============模型评价====================
output = clf.fit(train_data, train_label).predict(train_data).astype(int) #用train_data检验分类效果

#print u'线性拟合参数(线性相关度):W0{}\nW1-Wn{}'.format(clf.intercept_,clf.coef_) 
#print clf.feature_importances_ #随机森林
from sklearn.metrics import confusion_matrix
print '分类效果检验:[TN FP]/[FN TP]\n',confusion_matrix(train_label,output)
print '正确率:',clf.score(train_data,train_label)
from sklearn.cross_validation import cross_val_score
print u'精确率P:',cross_val_score(clf, train_data, train_label,scoring='precision').mean()
print u'召回率R:',cross_val_score(clf, train_data, train_label,scoring='recall').mean()
print u'F1-measure:',cross_val_score(clf, train_data, train_label, cv=5).mean()











#===============预测输出====================
output = clf.predict(test_data).astype(int) #用test_data真正预测
pd.DataFrame({"PassengerId": df["PassengerId"],"Survived": output}).to_csv("rf.csv", index=False)


