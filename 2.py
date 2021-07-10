# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:19:34 2019

@author: George
"""


import numpy as np
import random


list1 = [random.randint(1,50) for i in range(200)]
print(list1)

counting=list1.count(10)
print("Number 10 counted" , counting ," times ")


list2=np.arange(1,51).tolist()
print(list2)

#2nd solution as a comment.
#list2=[]
#for j in range(1,51):
    #list2.append(j)

for i in list2:
   list1.append(i)
     
print(list1)
print("The length is", len(list1)) 

for i in range(99,130):
    del list1[i]
print("The new length is", len(list1)) 


the_max=max(list1)
print("The max is " , the_max)
the_min=min(list1)   
print("The min is" ,the_min)

average=sum(list1)/len(list1)
print("Average is " , average)