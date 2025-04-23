# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 0:start 

step 1.Import the data file and import numpy, matplotlib and scipy.

step 2.Visulaize the data and define the sigmoid function, cost function and gradient descent.

step 3.Plot the decision boundary .

step 4.Calculate the y-prediction.

step 5.stop

## Program And Output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('/content/Placement_Data.csv')
dataset.head()
```
![image](https://github.com/user-attachments/assets/10d6ffc6-2460-40ec-a746-11ee4997f0b9)

```
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
```
![image](https://github.com/user-attachments/assets/74c40643-5548-4f8c-b712-9550807ba2d5)

```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/d03a9e64-8aa5-47be-acfb-aadd038392cc)

```
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
```
![image](https://github.com/user-attachments/assets/2e19f9e6-327f-41a0-a418-9f358766da4c)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

