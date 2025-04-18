import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data =pd.read_csv("loan_march24.csv")
print(data)

features =data[["GENDER" , "OCCUPATION"]]
target =data["DEFAULT"]

cfeatures =pd.get_dummies(features)
print(features)
print(cfeatures)

model = RandomForestClassifier(n_estimators =5 )
mf= model.fit(cfeatures.values ,target)

ge =int(input("1 female and 2 male"))
if ge ==1:
          d1 =[1 ,0]
else:
          d1 =[0,1]

oc=int(input("1 business and 2 saalry"))
if oc ==1:
           d2 =[ 1, 0]
else:
           d2 =[0, 1]

d=[d1+d2]
res=model.predict(d)
print(res)
