#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 01:35:13 2019

@author: nishtha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Churn_Modelling.csv")

X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
X[:,1]=labelencoder_x1.fit_transform(X[:,1])
labelencoder_x2=LabelEncoder()
X[:,2]=labelencoder_x2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]  # REMOVING FIRST COLOUMN


#split into test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#FeatureScaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)




###ANNN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initialising ANN
classifier=Sequential()

#input llayer and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))#start with 0.1 and then increment by 0.1 till 0.4(remove some neurons) to remove overfitting for first layer
#second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))#start with 0.1 and then increment by 0.1 till 0.4(remove some neurons) to remove overfitting for second layer
#output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compiling ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting ann
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)


y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5) #if greater show true else false

"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

#for predicting a new value
new_pred=classifier.predict(sc_x.transform(np.array([[0,0,600,1,40,3,6000,2,1,1,5000]]))) #by seeing the values from dummy variables
new_pred=(new_pred>0.5)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred) #ismai true false mai answer le rhe hai


#evaluating ann(better way)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()





#Tuning ANN(IMP)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(X_train,Y_train)
best_params=grid_search.best_params_
best_acc=grid_search.best_score_
