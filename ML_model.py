#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
datin=pd.read_csv("/root/Automation/train.csv")
data16 = datin[['Pclass', 'Sex', 'Age', 'Parch','SibSp','Fare','Embarked']]
y=datin['Survived']
def MissingValues(pclass, age):
    if pclass==1:
        return 38
    elif pclass==2:
        return 30
    else:
        return 25 
i=0
while (i!=891):
    a= datin['Pclass'][i]
    b= datin['Age'][i]
    if math.isnan(b): 
        data16['Age'][i]= MissingValues(a,b)
        i+=1
    else:
        i+=1
        continue
X= data16[['Age','Fare']]
pclass = pd.get_dummies(data16['Pclass'], drop_first=True)
parch = pd.get_dummies(data16['Parch'], drop_first=True)
sib = pd.get_dummies(data16['SibSp'], drop_first=True) 
sex = pd.get_dummies(data16['Sex'], drop_first=True) 
embar = pd.get_dummies(data16['Embarked'], drop_first=True)
X_new= pd.concat([X, pclass, sib, parch, sex, embar], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 42)
model= LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report=classification_report(y_test,y_pred)
accuracy=(cm.diagonal().sum()/cm.sum())*100
print(report)
print(accuracy)
with open('/root/Automation/accuracy_ml.txt','w') as f:
    f.write(str(accuracy))