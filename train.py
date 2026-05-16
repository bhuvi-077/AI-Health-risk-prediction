import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#Load dataset
df = pd.read_csv("dataset/heart.csv")
print(df.head())

#understand the data
print(df.shape)
print(df.info())
print(df.describe())

#check missing values
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)

#visualization

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

#separate features and target variable

#x = df.drop("target", axis = 1)
x = df[["age", "trestbps", "chol"]]
y = df["target"]

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


#scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#train ml models

#model1

lr = LogisticRegression()
lr.fit(x_train, y_train)

#model2

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

#prediction

lr_pred = lr.predict(x_test)
rf_pred = rf.predict(x_test)

#Evaluation

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Random Forest Accuracy:", accuracy_score(y_test,rf_pred))

print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test,rf_pred))

joblib.dump(rf, "health_model.pkl")
joblib.dump(scaler, "scaler.pkl")

model = joblib.load("health_model.pkl")
prediction = model.predict([[52,1,140,250,1,1,160,0,2.3,1,0,2,3]])
print(prediction)