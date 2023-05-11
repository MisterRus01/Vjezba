import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



data = pd.read_csv("titanic.csv")


# Izbacivanje null i missing vrijednosti

data.dropna(inplace=True)



# Kreiranje funkcije plot_decision_regions

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("titanic.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Pclass", "Sex", "Fare", "Embarked"]].to_numpy()
y = data["Survived"].to_numpy()

# podijeli podatke u omjeru 60-40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify=y, random_state = 10)

# skaliraj ulazne velicine  - osigurava da svi podaci budu na istoj skali
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)            # treniranje modela s podacima za trening

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))           # izračunavanje točnosti train skupa
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))              # izračunavanje točnosti test skupa

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)          # prikazivanje granica odluke modela
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()




#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Izradite algoritam KNN na skupu podataka za učenje (uz K=5). Vizualizirajte podatkovneprimjere i granicu odluke.
# Izračunajte točnost klasifikacije na skupu podataka za učenje i skupu podataka za testiranje.

# KNN model
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train, y_train)

# Evaluacija KNN modela
y_train_p = KNN_model.predict(X_train_n)
y_test_p = KNN_model.predict(X_test_n)

print("KNN: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# Granica odluke pomoću KNN
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Točnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritmaKNN.

knn_model = KNeighborsClassifier()

# GridSearch za pronalazak optimalnog parametra K
param_grid = {'n_neighbors': range(1, 100)}

grid = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_n, y_train)

print("Najbolji K parametar: ", grid.best_params_['n_neighbors'])


# 