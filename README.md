# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries. 
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRAJAN P
RegisterNumber: 212223240121 
*/

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:
Array value of X:


![270424335-e5311ce6-a9ee-4086-b99f-f8c6b4491790](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/303dee8a-a0ea-4e0b-850c-051e1e8d5467)


Array value of y:

![270424349-462fb7c0-fff3-4b13-94d0-b6498c351f26](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/20ec4327-5ad8-4e72-b6cd-94ff2abaf911)

Exam 1-score graph:

![270424409-edc4acfc-30af-40ec-9c5e-eac35cb89e19](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/1f74fc32-c5a9-4f0d-ba91-4db1038aa966)

Sigmoid function graph:

![270424445-9bd4bfca-0274-4d02-97ea-88bc4274d31a](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/c77b286e-7a13-4274-9eef-1b4fb22e418a)

X_train_grad value:

![270424466-63a6de99-e789-4656-8200-b5f9cea9747b](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/711a9d41-a354-4dd0-92e3-5685edfe89b5)


Y_train_grad value:

![270424645-ced57c1b-be0d-48a9-8d21-504778656c5f](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/3247ef45-292c-4f37-b2a8-c44d753510b0)

Print res.x:

![270424696-98c18a7b-e6c0-46db-bc53-05752a2fefbd](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/5bb7a5bf-92a4-4586-b602-d5cf1c66a7f3)

Decision boundary-graph for exam score:

![270424710-d0a35897-b5a6-42a3-a856-00ea5086697f](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/23dec1f0-6f63-4620-8845-0e5c24b0d818)

Probability value:

![270424744-ccfd7a31-69c4-41ab-bac5-c019aa989b86](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/a537b630-e9b0-4efe-ac19-e71175ab27da)

Probability value of mean:

![270424764-1d6e9ca6-3ecf-4029-828b-1993ca653c66](https://github.com/PRAJAN-23013995/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/150313345/50d10d48-ed3b-4869-8262-894e78abb537)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

