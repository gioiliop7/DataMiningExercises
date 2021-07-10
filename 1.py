# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:54:13 2019

@author: George
"""

import math

x=input("Give 1st name \n")
y=input("Give 2nd name \n")
print(x,",",y)

z=1
while z>0:    
            z=int(input("Give a positive number \n"))
            if z<=0:
                z=int(input("A POSITIVE NUMBER\n"))
            else:
                break
            Sqrt_of_number=(math.sqrt(z))
            print("Τhe sqrt of your number is = ",Sqrt_of_number)          
    
a=eval(input("Give a number \n"))
b=20-a

if a>20:
    print("Our number is bigger than 20 and its = ",a)
else:
    print("The difference between our number and 20 is = ",b)

def calculate_value(b,i):
    k=int(b+i)
    print("k = ",k*k)
    
    i=eval(input("Give a number \n"))
    n=calculate_value(b,i)
    
    for p in range(0,200,5):
        print(p)
    for p in range(20,0,-1):
        print(p)