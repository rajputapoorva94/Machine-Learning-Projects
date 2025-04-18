import pandas as pd
from sklearn.tree import DecisionTreeRegressor ,plot_tree
import matplotlib.pyplot as plt

data =pd.read_csv("salary_march24.csv")
print(data)

feature =data[["Level"]]
target=data["Salary"]

model =DecisionTreeRegressor()
mf =model.fit(feature.values ,target)

level =float(input("Enter level"))
salary =model.predict([[level]])
print(salary)

plt.figure(figsize=(12,5))
plot_tree(mf , filled=True , feature_names=["Level"])
plt.show()