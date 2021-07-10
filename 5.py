import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,precision_score

dataset='breast-cancer-wisconsin.csv'

# I USE THE CHARACTERISTCS FOUND THROUGH TXT

attributes=['Sample Code Number','Clumb Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

dataset=pd.read_csv(dataset,names = attributes) #Read Dataset
print(dataset)
print(dataset.shape)

target=dataset.loc[:,"Class"] #Class of cancer
Y=target.replace({4:1, 2:0}) #I CHANGE TO 0 AND 1 BENIGN AND MALIGNANT FROM 2 AND 4 ACCORDING TO TXT
print("-----")
print(Y)

#Delete id and characteristics

df=dataset.loc[:,['Clumb Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']]

df=df.replace('?',np.NaN) #Missing values to N/A
counts=dataset.isna().sum() #Count missing values
print(counts) #Print

X=df.fillna(df.median())
(X.head(30))
print(X[['Bare Nuclei']])

X.boxplot(figsize=(20,3))

print(X.dtypes)
""" 
BECAUSE THE BARE NUCLEI ACCORDING TO THE ABOVE COMMAND IS OBJECT, IT MUST BE CONVERTED INTO NUMERIC
TO GET INTO THE PLOT, PLOT MAKE IT TO DETECT POSSIBLE OUTLIERS. OUTLIERS ARE DIFFERENT DIFFERENT PRICES FROM THESE TOTAL
I WANT TO FIND THE OUTLIERS TO SEE ABOUT IF I WILL HAVE A GOOD RESULT BY PERFORMING THE ALGORITHM"""

X['Bare Nuclei']=pd.to_numeric(X['Bare Nuclei'])
print(X.dtypes)

X.boxplot(figsize=(20,3))

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20) #80% TRAIN 20% TEST

svclass=SVC(kernel='linear')
svclass.fit(X_train,Y_train)

Y_pred=svclass.predict(X_test)

#Confusion Matrix

print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
print(precision_score(Y_test, Y_pred, average='weighted'))
