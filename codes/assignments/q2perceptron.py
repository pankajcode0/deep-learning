import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys

color = ['green','red',"blue","yellow","pink"]
maxIteration = 1000000
test_size = 0.2

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
class predictions:
    def lossfunction(self,train,w):
        train = np.hstack((np.ones((train.shape[0],1)),train))
        l = train.shape[1]
        prediction = []
        numMisclasification = 0
        truePositive = 0
        trueNegative = 0
        falsePositive = 0
        falseNegative = 0
        index = 0
        for i in train:
            x_i = i[:l-1]
            y = i[l-1]  
            prediction.append([ np.dot(x_i,w),int(y)])
            print(prediction[index][1],"\n")
            if (prediction[index][1]==0 and prediction[index][0]>=0):
                numMisclasification = numMisclasification+1
                falseNegative = falseNegative+1
            if (prediction[index][1]==0 and prediction[index][0]<=0):
                falsePositive = falsePositive + 1
            if (prediction[index][1]==1 and prediction[index][0]<0):
                numMisclasification = numMisclasification+1
                trueNegative = trueNegative + 1
            if (prediction[index][1]==1 and prediction[index][0]>=0):
                truePositive = truePositive + 1
            index = index+1
        loss = numMisclasification/train.shape[0]
        self.accuracy = (1-loss)*100
        self.predictList = prediction
        self.numMisclasification=numMisclasification
        self.truePositive=truePositive
        self.trueNegative = trueNegative
        self.falsePositive = falsePositive
        self.falseNegative = falseNegative
        
   
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





df = pd.read_csv("DatasetQuestion2.csv",header=None)
num_features = df.columns.shape[0] - 1
X_train, X_test = train_test_split(df,test_size=0.2)


new_perceptron = perceptron(X_train,X_test)

#print(new_perceptron.train.shape)
#print(new_perceptron.test.shape)
#y = np.array([new_perceptron.w]*train.shape[0])
#print(y)
print(new_perceptron.perceptronAlgorithm(new_perceptron.train))
#print(perceptronLine(new_perceptron.test,new_perceptron.w))
#ploter(np.array(train),color,perceptronLine(new_perceptron.test,new_perceptron.w))
#ploter(np.array(test),color,perceptronLine(new_perceptron.test,new_perceptron.w))#
#wploter(np.array(train),np.array(test),color,new_perceptron.w)
test = predictions()
test.lossfunction(np.array(X_train),new_perceptron.w)
print("accuracy",test.accuracy)
print(test.trueNegative)
print(test.truePositive)
print(test.falseNegative)
print(test.falsePositive) 