import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import sys

data=pd.read_csv("weather_play_nov23.csv")
print(data)

feature =data[["Weather"]]
target=data["Play"]
nfeature =pd.get_dummies(feature)

print(feature)
print(nfeatures)

model=BernoulliNB()
model.fit(nfeatures.values ,target)

we =int(input("1 overcast ,2 rainy and 3 sunny"))
if we ==1 :
             d =[[ 1 ,0 , 0]]
elif we ==2 :
             d =[[0,1 ,0]]
elif we ==3:
             d =[[ 0, 0 ,1]]
else :
             print("invalid option")
             sys.exit(1)

ans =model.predict(d)
print(ans)

res =model.predict_proba(d)
print(res)