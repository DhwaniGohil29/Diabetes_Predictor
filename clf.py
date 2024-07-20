import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from pickle import *

data = pd.read_csv("diabetes23.csv")
print(data)

print(data.isnull().sum())

features = data[["FS", "FU"]]
target = data["Diabetes"]

nfeatures = pd.get_dummies(features)
print(features)
print(nfeatures)

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target, random_state=0)

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)

cr = classification_report(y_test, model.predict(x_test))
print(cr)

f = None
try:
	f = open("diab.model", "wb")
	dump(model, f)
except Exception as e:
	print("issue", e)
finally:
	if f is not None:
			f.close()