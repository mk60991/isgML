# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 22:25:18 2019

@author: hp
"""

import numpy as np
import pandas as pd

df=pd.read_csv("isg.csv")

features=df.iloc[:,1:].values
labels=df.iloc[:,0].values

df1=pd.DataFrame(features)
df2=pd.DataFrame(labels)

print(df.head())

# Split in training and testing
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25)

"""
# Scaleing the features to overcome the bigger differences
from sklearn.preprocessing import StandardScaler
features_scale = StandardScaler()
features_train = features_scale.fit_transform(features_train)
features_test = features_scale.transform(features_test)
"""

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy', random_state=0)
clf.fit(features_train, labels_train)


labels_pred = clf.predict(features_test)
print(labels_pred)
#or

#print(labels_pred)

#to calulate confusion matrix
#1st diagonal represents "ACCURACY" , diagonal values increses ACCURACY increases
#2nd diagonal reprsents "MISSCLASSIFIED" Its values have to decrese
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)
print(cm)

#score
print(clf.score(features_test, labels_test))




#save model and deploy it anywhere using flask
#method1
import pickle
#serializing our model to a file called model.pkl
pickle.dump(clf, open("isg.pkl","wb"))

#loading a model from a file called model.pkl
with open('isg.pkl', 'rb') as handle:
    clf= pickle.load(handle)   
    
    
    
pred_features=np.array([[0.2,0.1],[0.9,0.7]])
pred_result=clf.predict(pred_features)