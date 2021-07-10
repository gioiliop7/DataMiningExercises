import numpy as np
import pandas as pd
from sklearn import preprocessing

dataset='diabetes.csv'
names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age','Outcome']
dataset=pd.read_csv(dataset,names = names)
print(dataset)

print("----------------")
print(dataset.shape)
print("----------------")
print(dataset.head(20))
print("----------------")
print(dataset.describe())
print("----------------")


length=len(dataset)
print(length)
print("----------------")
#Choose the columns that i want to check

df=dataset.loc[:,['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']]
print(df.head(768)) #Check if ok
print("----------------")
print(df.where(df==0).count()) #Print all zeros
print("----------------")
print(df==0)
print("----------------")
df=df[~(df==0).any(axis=1)] #Drop Zeros
print(df) #Print the new table without zeros
print("----------------")
length2=len(df)
print(length2) #Print the new length
df.to_csv('finaldataset.csv')
new_dataset=[dataset,df]
new_dataset_result=pd.concat(new_dataset,sort='True') #Outcome and pregnancies
print(new_dataset_result) #Print the new dataset

df2=dataset.loc[:,['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']]
#Choose the columns that i want to check.
print(df2.head(768)) #Check if ok
print("----------------")
print("Zeros :")
print(df2.where(df2==0).count()) #Print all zeros

threshold=40 #Add threshold
df2=df2.dropna(how='all')

df2.drop(df2[df2['Glucose']>=threshold].index,inplace = True)
df2.drop(df2[df2['BloodPressure']>=threshold].index,inplace = True)
df2.drop(df2[df2['SkinThickness']>=threshold].index,inplace = True)
df2.drop(df2[df2['Insulin']>=threshold].index,inplace = True)
df2.drop(df2[df2['BMI']>=threshold].index,inplace = True)
df2.drop(df2[df2['DPF']>=threshold].index,inplace = True)
df2.drop(df2[df2['Age']>=threshold].index,inplace = True)

print(df2.head(768))
print(df2.shape)
print(df2)

#Combine Datasets

new_dataset2=[dataset,df2]
new_dataset_result_2=pd.concat(new_dataset2,sort='True')
print(new_dataset_result_2)

class_count1=new_dataset_result_2[new_dataset_result_2['Outcome']==1].count() #Print when outcome = 1
print(class_count1)

class_count2=new_dataset_result_2[new_dataset_result_2['Outcome']==0].count() #Print when outcome = 0
print(class_count2)

new_dataset_result_2.to_csv('final_dataset.csv') #Generate csv
new_dataset_result.to_csv('final_dataset1.csv')