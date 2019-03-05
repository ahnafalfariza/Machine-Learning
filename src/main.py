import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(url):
    x=[]
    data = pd.read_csv(url)
    for i, row in data.iterrows():
        x.append([float(row[i]) for i in range (len(row)-1)])
    return x

def get_target(url):
    target = []
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

    print("Learning Rate = %s" % (l_rate))

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
            bias1-=deltabias(prediction1,target[i][0])
            bias2-=deltabias(prediction2,target[i][1])
            
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

def error_chart(error, epoch):
    t = np.arange(epoch)
    plt.plot(t,error)
    
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error Chart')
    plt.grid(True)
    plt.show()

def acc_chart(acc, epoch):
    t = np.arange(epoch)
    plt.plot(t,acc)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Chart')
    plt.grid(True)
    plt.show()

def main():
    url='https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
    #url="iris.csv"
    data = get_data(url)
    target = get_target(url)
    l_rate = 0.8
    epoch = 100

    error1, error2, accuracy = train(data,target,epoch,0.8)
    
    error_chart(error1, epoch)
    error_chart(error2, epoch)
    acc_chart(accuracy, epoch)


if __name__ == '__main__':
    main()
