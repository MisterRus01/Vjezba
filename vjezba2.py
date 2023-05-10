import math
import random

f = open("SMSSpamCollection.txt", "r")
readFile = f.readlines()

countHam = 0
countSpam = 0
countUsklicnik = 0
countHamWords = 0
countSpamWords = 0
for line in readFile:
    part = line.strip().split("\t")
    if len(part) == 2:
        x = part[0]
        y = part[1]
        countWords = len(y.split())
        if x == "ham":
            countHam+=1
            countHamWords+=countWords
        else:
            countSpam+=1
            countSpamWords+=countWords
            if (y.endswith("!")):
                countUsklicnik+=1

print("Prosjecan broj rijeci u ham porukama je: ", countHamWords/countHam)
print("Prosjecan broj rijeci u spam porukama je: ", countSpamWords/countSpam)
print("Broj spam poruka koje zavrsavaju s !: ", countUsklicnik)

