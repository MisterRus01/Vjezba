import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model

data = pd.read_csv(("titanic.csv"),delimiter=",")

#a)
print(f"Broj osoba: {len(data)}") #suma ukupno osoba koje su bile na brodu
#b)
print(f"Broj prezivjelih osoba: {sum(data.Survived == 1)}") #Suma prezivjelih muskaraca i zena
#c)Pomo ́cu stupˇcastog dijagrama prikažite postotke preživjelih muškaraca i žena. Dodajte
#nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka preživljavanja

data_df = pd.DataFrame(data)

#sumManSurvived = sum((data_df.Survived == 1) & (data_df.Sex == "male"))
sumMan = sum(data_df.Sex == "male")
#avargeManSurvived = sumManSurvived/sumMan
#sumWomanSurvived = sum((data_df.Survived == 1) & (data_df.Sex == "female"))
#sumWoman = sum(data_df.Sex == "female")
#avargeWomanSurvived = sumWomanSurvived/sumWoman
#print(avargeManSurvived)
#print(avargeWomanSurvived)

survived_counts = data.groupby(['Sex', 'Survived']).size().unstack()

survived_percentages = survived_counts.apply(lambda x: x/x.sum() * 100, axis=1)

ax = survived_percentages.plot(kind='bar', stacked=True, rot=0)
ax.set_title('Postotak preživljavanja po spolu')
ax.set_xlabel('Spol')
ax.set_ylabel('Postotak preživjelih')
plt.show()


#d)Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih muškaraca?
survived_count = data_df[data_df["Survived"]==1]
male_survivors = survived_count[survived_count["Sex"]=="male"]
avargeAgeManSurvivor = np.mean(male_survivors["Age"])
print(f"Prosjecne godine prezivjelih muskaraca {avargeAgeManSurvivor}")

female_survivors = survived_count[survived_count["Sex"]=="female"]
avargeAgeWomanSurvivor = np.mean(female_survivors["Age"])
print(f"Prosjecne godine prezivjelih muskaraca {avargeAgeWomanSurvivor}")

#e)
#koliko godina ima najmla  ̄di preživjeli muškarac u svakoj od klasa? Komentirajte

firstClass = male_survivors[male_survivors["Pclass"]==1] #paziti da se stavi samo ono sto se trazi, ne svi podatci!!!
youngestFirstClass = np.min(firstClass["Age"])
print(f"Najmladi u prvoj klasi je:{youngestFirstClass}")

