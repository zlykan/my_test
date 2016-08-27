""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more
""" 
import pandas as pd
import numpy as np
import csv as csv

from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
# Data cleanup
####################### TRAIN DATA #########################
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


######################## TEST DATA ###########################
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print train_df.columns
print 'Training...'
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=100)#0.79

#from sklearn.neighbors import KNeighborsClassifier  
#clf=KNeighborsClassifier(n_neighbors=7)#0.7037

#from sklearn import svm   
#clf=svm.SVC(C=10,gamma=0.0029)#0.7878
#clf=GridSearchCV(svm.SVC(), param_grid={"C":np.logspace(-2, 10, 13),"gamma":np.logspace(-9, 3, 13)})
#output = clf.fit( train_data[0::,1::], train_data[0::,0] ).predict(test_data).astype(int)
#print("The best parameters are %s with a score of %0.2f"% (knnClf.best_params_, knnClf.best_score_))

#from sklearn.naive_bayes import GaussianNB      #nb for 高斯分布的数据
#clf=GaussianNB() #0.7867       

#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(C=1,penalty='l2',tol=0.0001)#0.7946

#from sklearn.ensemble import GradientBoostingClassifier
#clf=GradientBoostingClassifier(n_estimators=13)#0.8181

#from sklearn.ensemble import ExtraTreesClassifier
#clf=ExtraTreesClassifier(n_estimators=14, max_depth=None,min_samples_split=1, random_state=0)#0.7957

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators=100)#0.8013

#from sklearn.ensemble import GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)#0.8047

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=1)#0.7676


print cross_val_score(clf, train_data[:,1:], train_data[:,0]).mean()


output = clf.fit( train_data[0::,1::], train_data[0::,0] ).predict(test_data).astype(int)
print clf.feature_importances_



'''
predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
'''