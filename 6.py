import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,precision_score,f1_score,accuracy_score,recall_score

dataset='diabetes.csv'
names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age','Outcome']
dataset=pd.read_csv(dataset,names = names)

Y=dataset.loc[:,"Outcome"] #Only outcome

print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

df=dataset.loc[:,['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']]
# HERE I ONLY KEEP THE COLUMNS I WANT TO ANALYZE. COLUMNS 0 AND 8 HAVE NO SENSE SO THEY WILL NOT BE ANALYZED

mean_of_values=df.mean(skipna=True) 
print(mean_of_values)

#I DO ZERO NaNS TO MAKE A REPLACEMENT

df=df.mask(df==0)
print(df)

#Change NaN to mean value

X=df.fillna(df.mean())
print(X)

X.to_csv('diabetes_missing_values_replace.csv') #Generate file

#I DIVIDE THE CHARACTERISTICS IN X AND THE CLASS IN Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)

k=5
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,Y_train)

#predict_label
Y_pred=knn.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

#I CHOOSE MY METRICS

Yps=[]
Yf1=[]
Yrc=[]
Yac=[]

Yps.append(precision_score(Y_test,Y_pred,average='weighted'))
Yf1.append(f1_score(Y_test,Y_pred))
Yrc.append(recall_score(Y_test,Y_pred))
Yac.append(accuracy_score(Y_test,Y_pred))

#print_results(mean)

print("Mean of Precision = ",mean(Yps))
print("Mean of F1_Score = ",mean(Yf1))
print("Mean of Recall = ",mean(Yrc))
print("Mean of Accuracy = ",mean(Yac))

k=0
k_range=range(1,16)
accuraciesA={}
accuracyA_list=[]

for k in k_range:
    print('k=',k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)
    
    #predict_label_again
    Y_pred=knn.predict(X_test)
    accuraciesA[k]=accuracy_score(Y_test,Y_pred)
    accuracyA_list.append(accuracy_score(Y_test,Y_pred))
    
    #print_results
    print(confusion_matrix(Y_test,Y_pred))
    print(classification_report(Y_test,Y_pred))
    print("AccuracyA=",accuracyA_list)
    print("Precision=",precision_score)
    print("F1 Score=",f1_score)
    print("Recall=",recall_score)
    
#I READ THE DATASET I SAVED IN EXERCISE 4

dataset2='finaldataset.csv'
names2=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age','Outcome']
dataset2=pd.read_csv(dataset2,names = names2)
print(dataset2.shape)

Y2=dataset2.loc[:,"Outcome"] #I ISOLATE THE CLASS


print(dataset2.shape)
print(dataset2.head(20))
print(dataset2.describe())

X2=dataset2

#I DO NOT HAVE ZERO SO NO NEED FOR TREATMENT


#I DIVIDE THE CHARACTERISTICS IN X AND THE CLASS IN Y
X2_train,X2_test,Y2_train,Y2_test=train_test_split(X2,Y2,test_size=0.20)

k=5
knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X2_train,Y2_train)

#predict_label
Y2_pred=knn.predict(X2_test)
print(confusion_matrix(Y2_test,Y2_pred))
print(classification_report(Y2_test,Y2_pred))

Yps2=[]
Yf12=[]
Yrc2=[]
Yac2=[]

Yps2.append(precision_score(Y_test,Y_pred,average='weighted'))
Yf12.append(f1_score(Y_test,Y_pred))
Yrc2.append(recall_score(Y_test,Y_pred))
Yac2.append(accuracy_score(Y_test,Y_pred))

#print_results(mean)

print("Mean of Precision = ",mean(Yps2))
print("Mean of F1_Score = ",mean(Yf12))
print("Mean of Recall = ",mean(Yrc2))
print("Mean of Accuracy = ",mean(Yac2))

k=0
k_range=range(1,16)
accuraciesB={}
accuracyB_list=[]

for k in k_range:
    print('k=',k)
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X2_train,Y2_train)
    
    #predict_label_again
    Y2_pred=knn.predict(X2_test)
    accuraciesB[k]=accuracy_score(Y2_test,Y2_pred)
    accuracyB_list.append(accuracy_score(Y2_test,Y2_pred))
    
    #print_results
    print(confusion_matrix(Y2_test,Y2_pred))
    print(classification_report(Y2_test,Y2_pred))
    print("AccuracyB=",accuracyB_list)
    print("Precision=",precision_score)
    print("F1 Score=",f1_score)
    print("Recall=",recall_score)
    