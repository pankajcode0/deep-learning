import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys

color = ['green','red',"blue","yellow","pink"]

train = pd.read_csv("Datasets-Question1/dataset7/Train7.csv",header=None,names=["x1","x2","y"])
test = pd.read_csv("Datasets-Question1/dataset7/Test7.csv",header=None,names=["x1","x2","y"])
test_data = np.array(test)

data = np.array(train)
plt.subplot(2,2,1)
for i in data :
    cl=int(i[2])
    plt.scatter(i[0],i[1],color=color[cl])
#plt.show()

w=np.ones(data.shape[1])
x0=1
print(w)
print(data.shape[0])
print(np.ones((5,1)))

x=np.hstack((np.ones((data.shape[0],1)),data))
print(x[0])
converge = False
i=0
while (not converge):
    #  np.random.
    x_i = np.random.randint(x.shape[0])
    if (x[x_i][3]==1 and np.dot(w,x[x_i][:3])<0):
        w=w+x[x_i][:3]
    if (x[x_i][3]==0 and np.dot(w,x[x_i][:3])>=0):
        w=w-x[x_i][:3]
    i=i+1
    if(i>11000):
        converge=True
    print("w",w)
    print("x",x[x_i][:3])
    print("dot",np.dot(w,x[x_i][:3]))
print(w)


x1=np.amin(x[:,1])
x2=np.amax(x[:,2])
print(x1,x2)

y1= -(w[0]+w[1]*x1)/w[2]
y2= -(w[0]+w[1]*x2)/w[2]

print(y1,y2)
plt.plot([x1,x2], [y1,y2])

x1_test = np.amin(test_data[:,0])
x2_test = np.amax(test_data[:,1])

y1_test = -(w[0]+w[1]*x1_test)/w[2]
y2_test = -(w[0]+w[1]*x2_test)/w[2]

plt.subplot(2,2,2)

for i in test_data :
    cl=int(i[2])
    plt.scatter(i[0],i[1],color=color[cl+2])

plt.plot([x1_test,x2_test],[y1_test,y2_test])

plt.show()
