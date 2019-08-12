import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys

a = np.array([1,2,3,4])
print("a",a)
a = a + 1
print("a+1",a)
b = np.ones(4) + 1
print("b",b)
c = a*b
print("a*b",c)  
d=np.dot(a,b)
print("dot of a and b ",d)
print(np.transpose(b))
#e=np.cross(a,np.transpose(b))header=None
#print("cross of a and b",e)

train = pd.read_csv("Datasets-Question1/dataset1/Train1.csv",header=None,names=["x1","x2","y"])

print(train.head())
plt.scatter(train["x1"],train["x2"],color='green')
#plt.show()
#sys.exit(0)

data = np.array(train)
x=data[:,:2]
print(x.shape)

converge = False
i=0

new =np.random.randint(x.shape[0])
print(new)
w=data[new]
print(w[2])
print(w)
print("dot",np.dot(w,w))
print(w)
w=[1,1,1]
#print(w)
#print(np.dot(w,data[new][:2]))
#w=w+data[new][:2]
#
#print(data[new][:2])
#print(w)
while (not converge):
    #  np.random.
    new = np.random.randint(data.shape[0])
    if (data[new][2]==1 and np.dot(w,[1,data[new][:2]])<0):
        w=w+[1,data[new][:2]]
    if (data[new][2]==0 and np.dot(w,[1,data[new][:2]])>=0):
        w=w-[1,data[new][:2]]
    i=i+1
    if(i>1200):
        converge=True
    print("w",w)
    print("data",data[new][:2])
    print("dot",np.dot(w,data[new][:2]))
print(w)
plt.scatter(w[0]*data[0][0],w[1]*data[0][1])
plt.show()
    