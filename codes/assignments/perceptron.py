import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys

color = ['green','red',"blue","yellow","pink"]
maxIteration = 1000000

class perceptron:
    def __init__(self,train,test):
        self.train = np.hstack((np.ones((train.shape[0],1)),train))
        self.test = np.array(test)
        self.w = self.perceptronAlgorithm(self.train)
    def perceptronAlgorithm(self,train):
        w = np.ones(train.shape[1]-1)
        l = train.shape[1]
        converge = False
        j=0
        while(not converge):
            i = np.random.randint(train.shape[0])
            x_i = train[i,:l-1]
            y = train[i][l-1]  
            if (y==1 and np.dot(w,x_i)<0):
                w=w+x_i 
            if (y==0 and np.dot(w,x_i)>0):
                w=w-x_i
            if(j>maxIteration ):
                converge=True
            j=j+1
        return w
def perceptronLine(data,w):
    x1=np.amin(data[:,0])
    x2=np.amax(data[:,1])
    y1= -(w[0]+w[1]*x1)/w[2]
    y2= -(w[0]+w[1]*x2)/w[2]
    line=[x1,x2,y1,y2]
    return line

def ploter(data,color,line):
    for i in data :
        cl=int(i[2])
        plt.scatter(i[0],i[1],color=color[cl])
    plt.plot([line[0],line[1]],[line[2],line[3]])
    plt.show()
def wploter(train,test,color,w):
    plt.subplot(1,2,1)
    for i in train :
        cl=int(i[2])
        plt.scatter(i[0],i[1],color=color[cl])
    line1=perceptronLine(train,w)
    plt.title("train")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot([line1[0],line1[1]],[line1[2],line1[3]])
    plt.subplot(1,2,2)
    plt.title("test")
    plt.xlabel("x1")
    plt.ylabel("x2")
    for i in test :
        cl=int(i[2])
        plt.scatter(i[0],i[1],color=color[cl])
    line2=perceptronLine(test,w)
    plt.plot([line2[0],line2[1]],[line2[2],line2[3]])
    plt.show()


train = pd.read_csv("Datasets-Question1/dataset6/Train6.csv",header=None,names=["x1","x2","y"])
test = pd.read_csv("Datasets-Question1/dataset1/Test1.csv",header=None,names=["x1","x2","y"])

new_perceptron = perceptron(train,test)

print(new_perceptron.train.shape)
print(new_perceptron.test.shape)
print(new_perceptron.w)
print(new_perceptron.perceptronAlgorithm(new_perceptron.train))
print(perceptronLine(new_perceptron.test,new_perceptron.w))
#ploter(np.array(train),color,perceptronLine(new_perceptron.test,new_perceptron.w))
#ploter(np.array(test),color,perceptronLine(new_perceptron.test,new_perceptron.w))
wploter(np.array(train),np.array(test),color,new_perceptron.w)