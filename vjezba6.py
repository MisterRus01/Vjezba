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

#data = np.loadtxt("titanic.csv", skiprows=1, delimiter=",")
data_df = pd.read_csv("titanic.csv")
print(data_df)
print(len(data_df))
data_df = data_df.dropna()
print(data_df)
data_df.drop_duplicates()
data_df=data_df.reset_index(drop=True)
data_df.loc[data_df.Sex=="male", "Sex"]=0
data_df.loc[data_df.Sex=="female", "Sex"]=1
data_df.loc[data_df.Embarked=="S", "Embarked"]=0
data_df.loc[data_df.Embarked=="C", "Embarked"]=1
data_df.loc[data_df.Embarked=="Q", "Embarked"]=2
print(len(data_df))

X = data_df.drop(columns=["PassengerId", "Survived", "Name", "Age", "SibSp", "Parch", "Ticket", "Cabin"]).to_numpy()
y=data_df["Survived"].copy().to_numpy()
X = np.asarray(X).astype(np.float32)

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.4 , random_state = 1 )

sc = MinMaxScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

#A)
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)
y_test_p_KNN = KNN_model.predict(X_test_n)
y_train_p_KNN = KNN_model.predict(X_train_n)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost, KNN : " +"{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

#B)
print("Algoritam k najblizih susjeda: ")
print("Tocnost train: " +"{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " +"{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

#C)
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
knn_gscv = GridSearchCV(KNN_model, param_grid, cv=5, scoring='accuracy', n_jobs =-1)

knn_gscv.fit(X_train_n, y_train)

print("best params for train: ", knn_gscv.best_params_ )
print("best score for train: ", knn_gscv.best_score_ )
knn_gscv.fit(X_test_n, y_test)

print("best params for test: ", knn_gscv.best_params_ )
print("best score for test: ", knn_gscv.best_score_ ) 
