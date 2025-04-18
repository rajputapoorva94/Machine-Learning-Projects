import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("weight_height_data_march24.csv")
print(data)

features = data[["Weight" ,"Height"]] 
target= data["Class"]

mms=MinMaxScaler()
nfeatures =mms.fit_transform(features.values)
print(features)
print(nfeatures)

k =int(len(data)  ** 0.5)
if k%2 == 0:
              k =k+1 
print(k)

model =KNeighborsClassifier(n_neighbors=k , metric="euclidean")
model.fit(nfeatures , target)

wt=float(input("Enter weight"))
ht =float(input("Enter the height"))
d=[[ wt ,ht]]
nd = mms.transform(d)
print(d)
print(nd)
ans=model.predict(nd)
print(ans)

res= model.kneighbors(nd, n_neighbors=k)
print(res)