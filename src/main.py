import numpy as np
import pandas as pd
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

def rand_numb(n):
    return np.random.random_sample((n))


def train(data,target,epoch,l_rate):
    theta1=rand_numb(4)
    theta2=rand_numb(4)
    bias1=rand_numb(1)
    bias2=rand_numb(1)

    for m in range(epoch):
        errors1=0.0
        errors2=0.0
        print('Epoch')
        print(m+1)
        
        for i in range(len(data)):
            o1=np.dot(data[i],theta1)+bias1
            o2=np.dot(data[i],theta2)+bias2

            #activation function
            prediction1 = signoid(o1)
            prediction2 = signoid(o2)

            #update theta
            for j in range(len(theta1)):
                theta1[j]=theta1[j]-l_rate*deltatheta(prediction1,target[i][0],data[i][j])
                theta2[j]=theta2[j]-l_rate*deltatheta(prediction2,target[i][1],data[i][j])

            #update bias
            bias1=bias1-deltabias(prediction1,target[i][0])
            bias2=bias2-deltabias(prediction2,target[i][1])
            
            errors1+=error(prediction1, target[i][0])
            errors2+=error(prediction2, target[i][1])

        print(errors1/len(data))
        print(errors2/len(data))
        print()

def predict(x):
    if x >= 0.5
        return 1
    else return 0

def error(prediction, target):
    return (prediction-target)**2

def accuracy(prediction, target):
    

def deltatheta(prediction, target, attr):
    return (2*(prediction-target)*(1-prediction)*prediction*attr)

def deltabias(prediction, target):
    return (2*(prediction-target)*(1-prediction)*prediction)

def main():
    data=get_data()
    target=get_target()
    l_rate=0.1
    epoch=100

    train(data,target,epoch,l_rate)

if __name__ == '__main__':
    main()
