import PIL
import pytesseract as pytesseract
from PIL import Image
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
from numpy import array
from collections import OrderedDict
import numpy

FILE_TXT = "/root/PycharmProjects/zad/szczypka/text"
FILE_BMP = '/root/PycharmProjects/zad/szczypka/Dokument.bmp'

def getDiff(source_file, dest_file):
    pass

def getText(source_file):
    with open(source_file) as f:
        str = f.read()
        characters = list(str)
    return "".join(OrderedDict.fromkeys(sorted(characters)))

#def BMPToFile(source_path):
#    output = pytesseract.image_to_string(PIL.Image.open(source_path).convert("RGB"), lang='eng')
#    output = list(output)
#    return output

def charAfterChar(source_path, list_of_chars):
    listOfLists = []
    for x in list_of_chars:
        with open(source_path, 'r') as f:
            str = f.read()
            listOfIndexes = [pos + 1 for pos, char in enumerate(str) if char == x]
            listOfChars = [str[x] for x in listOfIndexes if x < len(str)]
        chars = list(listOfChars)
        keys = list(OrderedDict.fromkeys(sorted(listOfChars)).keys())
        list_of_instances = [chars.count(z) for z in set(chars)]
        print(x, len(keys), keys, list_of_instances)
        listOfLists.append(keys)
    return listOfLists

def NewVectorMaker(source_path):
    dictionary = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z',
                  'x', 'c', 'v', 'b', 'n', 'm']
    with open(source_path) as f:
        string = f.read().split(" ")
        return NewVectorWordMaker(string, dictionary)

def NewVectorWordMaker(string, dictionary):
    #[[a, one_hot(x), one_hot(x)], ...][ascii(a), one_hot(x), one_hot(x)]

    index = 0
    temp = []
    for word in string:
        word = str.lower(word)
        for i in range(len(word)):
            if str.isalpha(word[i]):
                if len(word) == 1:
                    temp.append([word[i], one_hot('[', dictionary), one_hot(']', dictionary)])
                elif i == 0:
                    temp.append([word[i], one_hot('[', dictionary), one_hot(word[i + 1], dictionary)])
                elif i == len(word)-1:
                    temp.append([word[i], one_hot(word[i - 1], dictionary), one_hot('[', dictionary)])
                else:
                    temp.append([word[i], one_hot(word[i-1], dictionary), one_hot(word[i+1], dictionary)])
        if index%1000 == 0:
            print(index)
        index = index + 1
    return temp

def GetTableForTree(temp):
    arr1 = []
    arr2 = numpy.empty((2, 1))
    for i in temp:
        arr1.append(ord(i[0]))
        arr2 = numpy.append(arr2, i[1].append(i[2]))
    print(arr2)
    return arr1, arr2

def NewMakeTree(source_path):
    temp = GetTableForTree(NewVectorMaker(source_path))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(temp[0], temp[1])
    print("OK")

def VectorSingleMaker(vector_length, max_length_word, string, dictionary):
    for char in dictionary:
        letterVector = []
        size = 0
        for word in string:
            if size > vector_length:
                size = 1
                break
            for j in range(0, len(word)):
                if size > vector_length:
                    size = 1
                    break
                if word[j] == char:
                    for length in range(2, len(word) - j):
                        if length > max_length_word or size > vector_length:
                            break
                        if len([i for i in word[j:(length + j)] if not str.isalpha(i)]) == 0:
                            letterVector.append(word[j:(j + length)])
                            size = size + 1
        if size > 0:
            while size <= vector_length:
                letterVector.append("null")
                size = size + 1
            size = 1
        for word in string:
            if size > vector_length:
                break
            for j in range(0, len(word)):
                if len(word) > 0 and word[-1 - j] == char:
                    for length in range(len(word), 2, -1):
                        if size > vector_length or len(word) - length - j > max_length_word or len(word) - j - length > 1:
                            break
                        if len([i for i in word[(length - j):(len(word) - j)] if not str.isalpha(i)]) == 0:
                            w = word[(length - j):(len(word) - j)]
                            if len(w) > 1:
                                letterVector.append(w)
                                size = size + 1
        if size > 0:
            while size <= vector_length:
                letterVector.append("null")
                size = size + 1
        text = "[ "
        for i in letterVector:
            text = text + i + ", "
        text = text + " ]"
        text = char + text
        return text

def one_hot(dictionary, char):
    temp = [0 for i in range(len(dictionary))]
    for i in range(len(dictionary)):
        if temp[i] == char:
            temp[i] = 1
    return temp

def vectorMaker(source_path, vector_length, max_length_word):
    dictionary = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M']
    vector = open('vector.txt', 'w')
    with open(source_path) as f:
        string = f.read().split(" ")
        vector.write(vectorMaker(vector_length, max_length_word, string, dictionary))

    vector.close()

def makeTree():
    pass

print(getText(FILE_TXT))
#list_of_chars = BMPToFile(FILE_BMP)
#print(charAfterChar(FILE_TXT, list_of_chars))
print(NewMakeTree(FILE_TXT))
makeTree()
