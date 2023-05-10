import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model



data = np.loadtxt(("pima-indians-diabetes.csv"),delimiter=",", skiprows=9)

#a)
print(f"Mjerenja su izvrsena na {len(data)} osoba") 
#b)
data_df = pd.DataFrame(data)
print(f'Broj dupliciranih: {data_df.duplicated().sum()}')
print(f'Broj izostalih: {data_df.isnull().sum()}')
data_df = data_df.drop_duplicates()
data_df = data_df.dropna(axis=0) #trebalo je i izbacit sve 0 iz BMI
data = data[data[:,5]!=0.0] #ovo je falilo, izbacivanje sve s 0.0 BMI
data_df = pd.DataFrame(data) #kreiranje ponovno data_df ali ovaj put s očišćenim podacima bez redaka s BMI 0.0
print(f'Broj preostalih: {len(data_df)}') 

#c)
plt.scatter(x=data[:, 7], y=data[:, 5])
plt.title('Odnos dobi i BMI')
plt.xlabel('Age(years)')
plt.ylabel('BMI(weight in kg/(height in m)^2)')
plt.show()

#d)
print(f"Minimalni BMI: {data_df[5].min()}")
print(f"Maximalni BMI: {data_df[5].max()}")
print(f"Srednji BMI: {data_df[5].mean()}")

#e)
print(f'Minimalni BMI (dijabetes): {data_df[data_df[8]==1][5].min()}')
print(f'Maksimalni BMI (dijabetes): {data_df[data_df[8]==1][5].max()}')
print(f'Srednji BMI: (dijabetes) {data_df[data_df[8]==1][5].mean()}')

print(f'Minimalni BMI (dijabetes): {data_df[data_df[8]==0][5].min()}')
print(f'Maksimalni BMI (dijabetes): {data_df[data_df[8]==0][5].max()}')
print(f'Srednji BMI: (dijabetes) {data_df[data_df[8]==0][5].mean()}')

data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# a)
logReg_model = LogisticRegression(max_iter=300)
logReg_model.fit(X_train, y_train)

# b)
y_predictions = logReg_model.predict(X_test)

# c)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_predictions))
disp.plot()
plt.show()
# broj TN je 89, TP 36, FN 18 i FP 11, model često osobe koje imaju dijabetes proglasi da nemaju - greška, nedovoljno komentirano

# d)
print(f'Tocnost: {accuracy_score(y_test, y_predictions)}')
print(f'Preciznost: {precision_score(y_test, y_predictions)}')
print(f'Odziv: {recall_score(y_test, y_predictions)}')

# učitavanje podataka:
data_df = pd.DataFrame(data, columns=['num_pregnant', 'plasma', 'blood_pressure',
                       'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes']) #koriste se ocisceni podaci za dataframe
X = data_df.drop(columns=['diabetes']).to_numpy()
y = data_df['diabetes'].copy().to_numpy()

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)


# a)
model = keras.Sequential()
model.add(layers.Input(shape=(8,)))
model.add(layers.Dense(units=12, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()

# b)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy", ])

# c)
history = model.fit(X_train, y_train, batch_size=10,
                    epochs=150, validation_split=0.1)


# d)
model.save('Model/')

# e)
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
