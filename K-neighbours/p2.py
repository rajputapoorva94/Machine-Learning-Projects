import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

data =pd.read_csv("tshirt_data_march24.csv")
print(data)

features =data.drop(["Person" , "T-Shirt Size" ] , axis="columns")
target =data["T-Shirt Size"]

mms= MinMaxScaler()
nfeatures =mms.fit_transform(features.values)
print(features)
print(nfeatures)

k =int(len(data) **0.5)
if k%2 == 0:
              k = k+1
print(k)

model=KNeighborsClassifier(n_neighbors=k ,metric="euclidean")
#Initializes a KNeighborsClassifier object called model with k neighbors and the Euclidean distance metric.
model.fit(nfeatures ,target)

ht = float(input("Enter height"))
wt =float(input("Enter weight"))
d =[[ht , wt]]
#creates a list containing ht and wt
nd =mms.transform(d)
# Scales the user-entered height and weight using the same MinMaxScaler mms used earlier.
ans =model.predict(nd)
#Predicts the T-shirt size based on the scaled height and weight entered by the user using the trained KNN model.
print(ans)

res =model.kneighbors(nd , n_neighbors=k) 
#Finds the k nearest neighbors of the input data nd using the trained KNN model.
print(res)
