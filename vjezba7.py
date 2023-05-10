import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import MinMaxScaler, OneHotEncoder
from sklearn . neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model


data_df = pd.read_csv("titanic.csv")
data_df = data_df.dropna()
data_df.drop_duplicates()
data_df=data_df.reset_index(drop=True)
data_df.loc[data_df.Sex=="male", "Sex"]=0
data_df.loc[data_df.Sex=="female", "Sex"]=1
data_df.loc[data_df.Embarked=="S", "Embarked"]=0
data_df.loc[data_df.Embarked=="C", "Embarked"]=1
data_df.loc[data_df.Embarked=="Q", "Embarked"]=2


X = data_df.drop(columns=["PassengerId", "Survived", "Name", "Age", "SibSp", "Parch", "Ticket", "Cabin"]).to_numpy()
y=data_df["Survived"].copy().to_numpy()

X = np.asarray(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

sc = StandardScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)
#a)
model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(units=16, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=4, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()
#b)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

#c)
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=100, validation_split=0.1)

#d)
model.save('Model/')

#e)
model = load_model('Model/')
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')

# f)
y_predictions = model.predict(X_test)
y_predictions = np.around(y_predictions).astype(np.int32)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

