import pandas as pd
from sklearn.naive_bayes import BernoulliNB

data=pd.read_csv("email_nov23.csv")
print(data)

features =data.drop(["Mn" , "Result"] ,axis="columns")
target =data["Result"]

print(features)
print(target)

nfeatures =pd.get_dummies(features)
print(nfeatures)

model=BernoulliNB()
model.fit( nfeatures.values ,target)

dear =int(input("Dear : 1 for no and 2 for yes "))
if dear ==1 :
               d1 =[ 1, 0]
else:
               d1 =[0 ,1 ]

friend =int(input( "Friend : 1 for no and 2 for yes"))
if friend == 1:
               d2 =[1,0]
else :
               d2 =[0, 1]

lunch =int(input("Lunch ; 1 for no and 2 for yes"))
if lunch == 1 :
                d3=[1,0]
else :
                d3=[0,1]

money =int(input("Money : 1 for no and 2 for yes "))
if money ==1:
                 d4 =[1,0]
else:
                 d4=[0,1]

d= [ d1 + d2+ d3 + d4]
res= model.predict(d)
print(res)