import math
import random


def countUniqueWords(fileName):
    f = open(fileName, 'r')
    readFile = f.read().lower()
    wordsInFile = readFile.split()
    count_map = {} 
    for i in wordsInFile:
        if i in count_map:
            count_map[i]+=1
        else:
            count_map[i]=1
    
    count = 0
    for i in count_map:
        if count_map[i] == 1:
            count+=1
    
    f.close()
    return count

count = countUniqueWords("song.txt")
print(count)


