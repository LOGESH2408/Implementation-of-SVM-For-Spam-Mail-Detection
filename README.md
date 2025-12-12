# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:LOGESHWARAN S
RegisterNumber:25007255  
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
<img width="846" height="468" alt="Screenshot 2025-12-12 132738" src="https://github.com/user-attachments/assets/95bf08fa-8d2b-4617-8d4d-44f12dd586e6" />

<img width="156" height="45" alt="Screenshot 2025-12-12 132805" src="https://github.com/user-attachments/assets/86e4a005-c379-4a84-915f-b9cbd6dd34a6" />

<img width="1341" height="207" alt="Screenshot 2025-12-12 132910" src="https://github.com/user-attachments/assets/263ae53a-ce1f-43bc-96cd-3a9db16fb843" />

<img width="188" height="23" alt="Screenshot 2025-12-12 132921" src="https://github.com/user-attachments/assets/ba7d6d3b-6221-4f7e-a806-6c693eec4ade" />

<img width="770" height="38" alt="Screenshot 2025-12-12 132935" src="https://github.com/user-attachments/assets/ce5c1a06-ef6f-4d05-85d8-549d1c05dc7f" />

<img width="285" height="37" alt="Screenshot 2025-12-12 132945" src="https://github.com/user-attachments/assets/2412051f-d8bc-4d87-a1ab-992636f76996" />


<img width="127" height="64" alt="Screenshot 2025-12-12 132950" src="https://github.com/user-attachments/assets/a65136d5-64db-4342-9553-2d9618e6f03e" />

<img width="541" height="208" alt="Screenshot 2025-12-12 132959" src="https://github.com/user-attachments/assets/64f60322-f994-42e1-b8e2-a463dbf4d22f" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
