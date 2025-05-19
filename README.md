# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

step 1.Import the data file and import numpy, matplotlib and scipy.

step 2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

step 3.Plot the decision boundary .

step 4.Calculate the y-prediction.


## Program And Output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('/content/Placement_Data.csv')
dataset.head()

dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y

theta=np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,x,Y):
      h=sigmoid(x.dot(theta))
      return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(x.dot(theta))
        gradient=x.T.dot(h-y)/m
        theta-=alpha * gradient
    return theta

theta=gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)

def predict(theta,x):
    h=sigmoid(x.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,x)

accuracy=np.mean(y_pred.flatten()==Y)
print("Accuracy:",accuracy)

print(y_pred)
print(y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## OUTPUT:


![image](https://github.com/user-attachments/assets/10d6ffc6-2460-40ec-a746-11ee4997f0b9)


![image](https://github.com/user-attachments/assets/74c40643-5548-4f8c-b712-9550807ba2d5)


![image](https://github.com/user-attachments/assets/d03a9e64-8aa5-47be-acfb-aadd038392cc)


![image](https://github.com/user-attachments/assets/2e19f9e6-327f-41a0-a418-9f358766da4c)

![image](https://github.com/user-attachments/assets/f8c5f201-99db-4755-ab66-b0ef72fb25cb)


![image](https://github.com/user-attachments/assets/2c293d75-eed7-4493-9786-4bc15ac0e3ce)


![image](https://github.com/user-attachments/assets/575efd3b-6e9a-4ee6-a370-f4bf9af2792f)


![image](https://github.com/user-attachments/assets/c0a2405f-ac40-408a-81e3-538195015298)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

