# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vishwa vasu. R
RegisterNumber: 212222040183
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#remove the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report = classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
![273833218-c4603361-a589-4a4d-9454-adc780f5317a](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/cd1f0e50-b796-4e68-b527-9951a3d5fe15)
![273833240-6a05087a-896f-4dda-9183-117d0959c138](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/96a94aec-1ba0-413d-9fc7-5b94a8e3ac73)
![273833281-f15df1d5-ad1b-4207-9fb1-1ef4bec98351](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/8b176b7e-33d5-4780-9e53-8902cbb0e105)
![273833483-ec6f4dec-33eb-4213-9db7-826f3adc64c8](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/e5fbef53-67cd-4fa4-b097-c7d83becc722)
![273833507-85defd42-a247-4168-812c-e665dfd40f30](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/919905c5-649c-4bab-8f10-b694b0c1a919)
![273833536-ce6b2dde-45b4-4095-9a07-e823550cb8df](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/1d2b18e4-cd41-424b-b416-77fce0c48f81)
![273833563-f1eef88f-1fe9-4e55-92ba-d291dfe05bf4](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/0a61d8b5-00b8-4b4d-83fa-3ef2f5efce91)
![273833588-4e9d873c-53db-4c48-aaf9-d17c864bcb46](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/d14ddbd9-2c6a-43ef-a873-b3060370b862)
![273833606-edc84ac3-fa66-4b6b-a67c-e71f61e8e456](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/4f55532f-3e34-4f6e-9f54-9830dfb82d88)
![273833626-dd5ee355-0cde-41f6-a5c2-3d5e3a9b867e](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/13a44bd5-9942-4602-88c1-8e736a49afc2)
![273833652-1707abcb-487f-4c61-a0cf-738bc416509f](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/72d8fd9f-f153-42db-adb7-0843553fdc02)
![273833726-bd376339-df10-489b-b05e-7005289d893e](https://github.com/vishwa2005vasu/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/135954202/82427768-4d88-4735-a77d-24472a188dca)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
