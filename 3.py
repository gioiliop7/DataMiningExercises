import numpy as np 
import matplotlib.pyplot as plt

dataset = np.loadtxt('first_attempt_dataset.txt')
print(dataset)

class1=np.arange(len(dataset)).reshape(dataset)
class2=np.arange(len(dataset)).reshape(dataset)

for i in range (len(dataset)):
    if dataset[i][2]==1:
        class1.append([dataset[i][0],dataset[i][1]])
    elif dataset[i][2]==2:
        class2.append([dataset[i][0],dataset[i][1]])

print(class1)
   
print(class2)


plt.title("Jain Clustering Dataset")
plt.xlabel("x axis")
plt.ylabel("y axis")


for j in range(len(class1)):
    plt.plot(class1[j][0],class1[j][1],"o",color = "black")

for k in range(len(class2)):
    plt.plot(class2[k][0],class2[k][1],"o",color="red")   

plt.show()    