import pandas as pd 
from sklearn.naive_bayes import BernoulliNB

data=pd.read_csv("weather_car_result_nov23.csv")
print(data)

features =data[["Weather" , "Car"]]
target =data["Result"]
nfeatures =pd.get_dummies(features)

print(features)
print(nfeatures)

model =BernoulliNB()
model.fit(nfeatures.values ,target)

we=int(input("1 rainy and 2 sunny"))
if we ==1 :
             d2 =[1, 0]
else:
             d2 =[0, 1]

ca=int(input("1broken and 2 working"))
if ca ==1:
             d2=[1 ,0 ]
else :
             d2 =[ 0, 1 ]

d =[d1+ d2]
res =model.predict_proba(d)
print(res)

ans=model.predict_proba(d)
print(ans)




