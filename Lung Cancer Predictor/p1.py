import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , classification_report

data =pd.read_csv("lc_march24.csv")
print(data)

print(data.isnull().sum())

print(data.duplicated().sum())
data.drop_duplicates(keep="first" ,  inplace=True)
print(data.duplicated().sum())

features =data.drop("LUNG_CANCER" ,axis="columns")
target =data["LUNG_CANCER"]

cfeatures =pd.get_dummies(features)
print(features)
print(cfeatures)

mms=MinMaxScaler()
nfeatures=mms.fit_transform(cfeatures)
print(nfeatures)

x_train , x_test, y_train , y_test = train_test_split(nfeatures , target)

model =KNeighborsClassifier(n_neighbors=75 , metric="euclidean")
model.fit(x_train , y_train)

cm=confusion_matrix(y_test ,model.predict(x_test))
print(cm)

input()

cr=classification_report(y_test ,model.predict(x_test))
print(cr)

age=float(input("Enter age  "))
smoking=int(input ("1 for NO and 2 for YES  "))
yellow_fingers=int(input("Yellow Fingers: Enter 1 for NO and 2 for YES   "))
anxiety= int(input("Anxiety: Enter 1 for NO and 2 for YES   "))
peer_pressure=int(input("peer_pressure: Enter 1 for NO and 2 for YES  "))
chronic_disease=int(input("chronic disease: Enter 1 for NO and 2 for YES  "))
fatigue=int(input("Fatigue: Enter 1 for NO and 2 for YES   "))
allergy=int(input("Allergy: Enter 1 for NO and 2 for YES "))
wheezing=int(input("Wheezing: Enter 1 for NO and 2 for YES  "))
alcohol=int(input("Alcohol: Enter 1 for NO and 2 for YES  "))
coughing=int(input("coughing: Enter 1 for NO and 2 for YES "))
sob=int(input("Shortness of Breath: Enter 1 for NO and 2 for YES  "))
Chest_pain=int(input("Chest pain: Enter 1 for NO and 2 for YES   "))
Lung_cancer=int(input("Lung cancer: Enter 1 for NO and 2 for YES  "))

gender =int(input(" 1 for female and 2 for male"))

if gender ==1 :
           d=[[age ,smoking ,yellow_fingers ,anxiety ,peer_pressure ,chronic_disease,fatigue,allergy ,wheezing,alcohol, coughing,sob,Chest_pain , Lung_cancer ,1,0]]
else :
           d=[[age ,smoking ,yellow_fingers ,anxiety ,peer_pressure ,chronic_disease,fatigue,allergy ,wheezing,alcohol, coughing,sob,Chest_pain , Lung_cancer , 0,1]]
nd=mms.transform(d)
ans=model.predict(nd)
print(ans)












