import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix ,classification_report

data=pd.read_csv("sna_march24.csv")
print(data)

print(data.isnull().sum())

print(data.duplicated().sum())

features =data.drop(["User ID" , "Purchased" ] , axis="columns")
target =data["Purchased"]

nfeatures =pd.get_dummies(features)
print(features)
print(nfeatures)

x_train ,x_test , y_train ,y_test = train_test_split(nfeatures.values ,target)

model =GaussianNB()
model.fit(x_train , y_train)

cm= confusion_matrix(y_test , model.predict(x_test))
print(cm)

input()

cr =classification_report(y_test , model.predict(x_test))
print(cr)

age =int(input("Enter age "))
salary= model.predict([[age]])
gender = int(input("1 for Female and 2 for male "))
if gender == 1:
                d=[[ age , salary ,1 ,0]]
else:
                d=[[age ,salary ,0 ,1 ]]
ans =model.predict(d)
print(ans)

res =model.predict_proba(d)
print(res)


