import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

def get_data():
    x=[]
    url='https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
    data = pd.read_csv(url)
    for i, row in data.iterrows():
        x.append([float(row[i]) for i in range (len(row)-1)])
    return x

def get_target():
    target = []
    url='https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
    data = pd.read_csv(url)
    for i, row in data.iterrows():
        if (row[4]=='setosa'):
            target.append([0,0])
        elif (row[4]=='versicolor'):
            target.append([0,1])
        elif (row[4]=='virginica'):
            target.append([1,0])
    return target

def signoid(x):
    return 1 / (1 + np.exp(-x))

def rand_numb(n=1):
    arr = []
    for i in range(n):
        arr.append(np.random.random_sample())
    return arr

def rand():
    return np.random.random_sample()

def train(data,target,epoch,l_rate):
    theta1=rand_numb(4)
    theta2=rand_numb(4)
    bias1=rand()
    bias2=rand()
    
    error1 = []
    error2 = []
    accuracy = []

    for m in range(epoch):
        errors1=0
        errors2=0
        print('Epoch %s' % (m+1))
        acc=0
        
        for i in range(len(data)):
            o1=np.dot(data[i],theta1)+bias1
            o2=np.dot(data[i],theta2)+bias2
            
            #activation function
            prediction1 = signoid(o1)
            prediction2 = signoid(o2)

            if predict(prediction1)==target[i][0] and predict(prediction2)==target[i][1]:
                acc+=1
        
            #update theta
            for j in range(4):
                theta1[j]-=l_rate*deltatheta(prediction1,target[i][0],data[i][j])
                theta2[j]-=l_rate*deltatheta(prediction2,target[i][1],data[i][j])
 
            #update bias
            bias1=bias1-deltabias(prediction1,target[i][0])
            bias2=bias2-deltabias(prediction2,target[i][1])
            
            errors1+=error(prediction1, target[i][0])
            errors2+=error(prediction2, target[i][1])

        error1.append(errors1/len(data))
        print("Error1 = %s" % (errors1/len(data)))

        error2.append(errors1/len(data))
        print("Error2 = %s" % (errors2/len(data)))

        accuracy.append((acc/len(data))*100)
        print("Accuracy = %s" % ((acc/len(data))*100))

        print()

    return error1, error2, accuracy

def predict(x):
    p=0
    if x >= 0.5:
        p = 1
    return p

def error(prediction, target):
    return (prediction-target)**2
    

def deltatheta(prediction, target, attr):
    return (2*(prediction-target)*(1-prediction)*prediction*attr)

def deltabias(prediction, target):
    return (2*(prediction-target)*(1-prediction)*prediction)

def main():
    data=get_data()
    target=get_target()
    l_rate=0.1
    epoch=10

    error1, error2, accuracy = train(data,target,epoch,l_rate)

    print(error1)
    print(error2)
    print(accuracy)


if __name__ == '__main__':
    main()
