from sklearn import tree
import myPredict as mp
import random
import statistics as stat

dictionary = []
testText = "./test.txt"
polishDict = "./text"
polishWords = "./testPL.txt"

with open(polishDict) as f:
    for i in f.read().split():
        for j in list(i):
            dictionary.append(j)
    f.close()
dictionary = list(set(dictionary))
dictionary.sort()


def nextCharPredict():
    temp = getListForTree(newVectorMaker(testText))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(temp[1], temp[0])
    finalList = list(set(temp[0]))
    finalList.sort()
    return clf, finalList


def getListForTree(temp):
    arr1 = []
    arr2 = []
    for i in temp:
        arr1.append(i[0])
        arr2.append(i[1])
    return arr1, arr2


def newVectorMaker(source_path):
    with open(source_path) as f:
        string = f.read().split()
        return newVectorWordMaker(string)


def newVectorWordMaker(string):
    index = 0
    temp = []
    for word in string:
        for i in range(len(word)):
            if str.isalpha(word[i]) or word[i] == '\'':
                if len(word) == 0:
                    continue
                elif i == 0:
                    temp.append([word[i], oneHotArray([' ', ' ', ' ', ' ', ' '])])
                elif i == 1:
                    temp.append([word[i], oneHotArray([word[i - 1], ' ', ' ', ' ', ' '])])
                elif i == 2:
                    temp.append([word[i], oneHotArray([word[i - 2], word[i - 1], ' ', ' ', ' '])])
                    temp.append([word[i - 1] + word[i], oneHotArray([word[i - 2], ' ', ' ', ' ', ' '])])
                elif i == 3:
                    temp.append([word[i], oneHotArray([word[i - 3], word[i - 2], word[i - 1], ' ', ' '])])
                    temp.append([word[i - 1] + word[i], oneHotArray([word[i - 3], word[i - 2], ' ', ' ', ' '])])
                    temp.append([word[i - 2] + word[i - 1] + word[i], oneHotArray([word[i - 3], ' ', ' ', ' ', ' '])])
                elif i == 4:
                    temp.append([word[i], oneHotArray([word[i - 4], word[i - 3], word[i - 2], word[i - 1], ' '])])
                    temp.append([word[i - 1] + word[i], oneHotArray([word[i - 4], word[i - 3], word[i - 2], ' ', ' '])])
                    temp.append(
                        [word[i - 2] + word[i - 1] + word[i], oneHotArray([word[i - 4], word[i - 3], ' ', ' ', ' '])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 4], ' ', ' ', ' ', ' '])])
                elif i == 5:
                    temp.append(
                        [word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], word[i - 1]])])
                    temp.append(
                        [word[i - 1] + word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], ' '])])
                    temp.append([word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 5], word[i - 4], word[i - 3], ' ', ' '])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 5], word[i - 4], ' ', ' ', ' '])])
                    temp.append([word[i - 4] + word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 5], ' ', ' ', ' ', ' '])])
                elif i == 6:
                    temp.append(
                        [word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], word[i - 1]])])
                    temp.append([word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], word[i - 3], word[i - 2]])])
                    temp.append([word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], word[i - 3], ' '])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], ' ', ' '])])
                    temp.append([word[i - 4] + word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], ' ', ' ', ' '])])
                elif i == 7:
                    temp.append(
                        [word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], word[i - 1]])])
                    temp.append([word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], word[i - 3], word[i - 2]])])
                    temp.append([word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 7], word[i - 6], word[i - 5], word[i - 4], word[i - 3]])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 7], word[i - 6], word[i - 5], word[i - 4], ' '])])
                    temp.append([word[i - 4] + word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 7], word[i - 6], word[i - 5], ' ', ' '])])
                elif i == 8:
                    temp.append(
                        [word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], word[i - 1]])])
                    temp.append([word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], word[i - 3], word[i - 2]])])
                    temp.append([word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 7], word[i - 6], word[i - 5], word[i - 4], word[i - 3]])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 8], word[i - 7], word[i - 6], word[i - 5], word[i - 4]])])
                    temp.append([word[i - 4] + word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 8], word[i - 7], word[i - 6], word[i - 5], ' '])])
                else:
                    temp.append(
                        [word[i], oneHotArray([word[i - 5], word[i - 4], word[i - 3], word[i - 2], word[i - 1]])])
                    temp.append([word[i - 1] + word[i],
                                 oneHotArray([word[i - 6], word[i - 5], word[i - 4], word[i - 3], word[i - 2]])])
                    temp.append([word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 7], word[i - 6], word[i - 5], word[i - 4], word[i - 3]])])
                    temp.append([word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 8], word[i - 7], word[i - 6], word[i - 5], word[i - 4]])])
                    temp.append([word[i - 4] + word[i - 3] + word[i - 2] + word[i - 1] + word[i],
                                 oneHotArray([word[i - 9], word[i - 8], word[i - 7], word[i - 6], word[i - 5]])])
        if index % 1000 == 0:
            print(index)
        index = index + 1
    return temp


def oneHotArray(arr):
    temp = []
    for i in arr:
        for j in one_hot(i):
            temp.append(j)

    return temp


def one_hot(char):
    temp = [0 for i in range(len(dictionary))]
    for i in range(len(dictionary)):
        if dictionary[i] == char:
            temp[i] = 1
    return temp


def randomText():
    temp = []
    with open(testText) as f:
        string = f.read().split()
        for s in string:
            n = "".join([i for i in s if i in dictionary])
            if len(n) > 3:
                x = random.randrange(1, len(n) - 1)
                temp.append([n[0:x], [n[0:x]]])
                temp.append([n[x:len(n)], [n[x:len(n)]]])
            elif len(n) > 0:
                temp.append([n, [n]])
    random.shuffle(temp)
    return temp


def newMakeTreeForNextWord(length):
    temp = getListForTree(vectorForNextWord(length))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(temp[1], temp[0])
    finalList = list(set(temp[0]))
    finalList.sort()
    return clf, finalList


def vectorForNextWord(length):
    arr = []
    with open(testText) as f:
        string = f.read().split()
        for i in range(len(string) - 2):
            if len(string[i]) == 0:
                continue
            string[i] = "".join([j for j in string[i] if j in dictionary])
            string[i + 1] = "".join([j for j in string[i + 1] if j in dictionary])
            minLen = min(length, len(string[i]))
            temp = [" " for j in range(length)]
            for j in range(minLen):
                temp[j] = string[i][j]
            arr.append([string[i + 1], oneHotArray(temp)])
    return arr


def algorithm(textList, clf, finalList):
    debug = 0
    maxBadWords = 0
    badWords = []
    goodWords = []
    while True:
        while True:
            boolean = True
            prepare = prepareWords(textList, clf, finalList)
            propabForWords = debugAlgo(prepare, textList)
            for i in range(int((len(propabForWords)))):
                print((i / ((len(textList) - 1))) * 100, "% len:", len(textList))
                for j in range(len(propabForWords[i])):
                    if len(propabForWords[i][j]) > 1 and textList[propabForWords[i][j][2]][0] != '' and \
                            textList[propabForWords[i][j][1]][0] != '' and isPartOfWordInDictionary(
                        textList[propabForWords[i][j][1]][0] + textList[propabForWords[i][j][2]][0]):
                        if propabForWords[i][j][2] != propabForWords[i][j][1]:
                            for k in textList[propabForWords[i][j][2]][1]:
                                textList[propabForWords[i][j][1]][1].append(k)
                            textList[propabForWords[i][j][1]][0] = textList[propabForWords[i][j][1]][0] + \
                                                                   textList[propabForWords[i][j][2]][0]
                            textList[propabForWords[i][j][2]][0] = ''
                            boolean = False
                            break
            for i in range(len(textList)):
                if i >= len(textList):
                    break
                if textList[i][0] == '':
                    textList.pop(i)
            if boolean or debug == 100:
                break
            debug = debug + 1
        goodWords = wordInDictionary(textList)
        goodWords = sorted(goodWords, key=lambda x: len(x[1]))
        badWords = wordNotInDictionary(textList)
        badWords = sorted(badWords, key=lambda x: len(x[1]))
        for i in goodWords:
            i[1] = sorted(i[1], key=lambda x: wordSimpleProbability(x), reverse=True)
        for i in badWords:
            i[1] = sorted(i[1], key=lambda x: wordSimpleProbability(x), reverse=True)
        for i in badWords:
            for j in i[0][1]:
                goodWords.append([[j, [j]]])
        if len(badWords) != maxBadWords:
            maxBadWords = len(badWords)
        else:
            break
        textList = [i[0] for i in goodWords]
    textList = [i[0] for i in goodWords]
    for i in badWords:
        goodWords.append([i[1][0], [i[1][0]]])
    return textList


def prepareWords(textList, clf, finalList):
    temp = []
    for i in range(len(textList)):
        string = textList[i][0]
        arrList = [" " for index in range(5)]
        minList = min(5, len(string))
        for k in range(minList):
            arrList[minList - k - 1] = string[-k - 1]
        prop = clf.predict_proba([oneHotArray(arrList)])
        for j in range(len(textList)):
            if isPartOfWordInDictionary(string):
                index = indexFromProp(textList[j][0], finalList)
                if prop[0][index] > 0.0 and isPartOfWordInDictionary(textList[i][0] + textList[j][0]):
                    temp.append([i, j, prop[0][index], index, textList[i][0], textList[j][0]])
    temp = sorted(temp, key=lambda x: x[2], reverse=True)
    return temp


def isPartOfWordInDictionary(string):
    temp = open(polishWords).read().split()
    for k in range(len(temp) - 1):
        if string in temp[k]:
            return True
    return False


def indexFromProp(word, finalList):
    for j in range(len(word), 1, -1):
        for i in range(len(finalList)):
            if finalList[i].strip() == "".join(word[0:j]).strip():
                return i
    return -1


def debugAlgo(arr, textList):
    temp = []
    for i in range(len(textList)):
        temp.append([])
    for i in arr:
        temp[i[0]].append([i[2], i[0], i[1]])
    for i in temp:
        i.append([0])
    temp = sorted(temp, key=lambda x: x[0], reverse=True)
    return temp


def wordInDictionary(textList):
    temp = []
    for i in textList:
        if i[0] == '':
            continue
        isIn = False
        for j in open(polishWords).read().split():
            if i[0] == j:
                isIn = True
                break
        if isIn:
            j = [k for k in open(polishWords).read().split() if isPartOfWordInDictionary(k)]
            temp.append([i, j])
    return temp


def wordNotInDictionary(textList):
    temp = []
    for i in textList:
        if i[0] == '':
            continue
        if i[0] not in open(polishWords).read().split():
            j = [k for k in open(polishWords).read().split() if isPartOfWordInDictionary(k)]
            temp.append([i, j])
    return temp


def wordSimpleProbability(word):
    string = open(polishDict).read().split()
    j = 0
    for k in string:
        if word == k:
            j = j + 1
    return j / len(string)


def mergeWordsWithDecisionTree(textList, clf, length, finalList):
    debug = 0
    while True:
        nextWordProp = []
        textList = [j for j in textList if j[0] != '']
        boolean = True
        for i in range(int((len(textList)))):
            string = [" " for _ in range(length)]
            temp = textList[i][0].split(" ")[-1]
            maxLength = min(len(temp), length)
            for j in range(maxLength):
                string[j] = temp[j]
            prop = clf.predict_proba([oneHotArray(string)])
            arr = [[i, j, prop[0][j]] for j in range(len(prop[0])) if prop[0][j] > 0.1]
            nextWordProp.append(arr)
        for j in range(len(nextWordProp)):
            nextWordProp[j] = sorted(nextWordProp[j], key=lambda x: x[2], reverse=True)
        nextWordProp = sorted(nextWordProp, key=lambda x: x[0][2], reverse=True)
        for i in range(len(nextWordProp)):
            node = nextWordProp[i]
            for j in range(len(node)):
                was = False
                for k in range(len(textList)):
                    if textList[node[j][0]][0] == '' or textList[k][0] == '':
                        continue
                    if finalList[node[j][1]] == textList[k][0].split(" ")[0] and node[j][0] != k:
                        textList[node[j][0]][0] = textList[node[j][0]][0] + " " + textList[k][0]
                        textList[k][0] = ''
                        boolean = False
                        was = True
                        break
                if was:
                    break
        if boolean or debug == 100:
            break
        debug = debug + 1
    for i in textList:
        print(i)
    return textList

def mergeWords(textList, clf, length, finalList):
    debug = 0
    mergedWords = []
    while True:
        nextWordProp = []
        textList = [j for j in textList if j[0] != '']
        boolean = True
        for i in range(int((len(textList)))):
            string = [" " for _ in range(length)]
            temp = textList[i][0].split(" ")[-1]
            maxLength = min(len(temp), length)
            for j in range(maxLength):
                string[j] = temp[j]
            prop = clf.predict_proba([oneHotArray(string)])
            arr = [[i, j, prop[0][j]] for j in range(len(prop[0])) if prop[0][j] > 0.1]
            nextWordProp.append(arr)
        for j in range(len(nextWordProp)):
            nextWordProp[j] = sorted(nextWordProp[j], key=lambda x: x[2], reverse=True)
        nextWordProp = sorted(nextWordProp, key=lambda x: x[0][2], reverse=True)
        for i in range(len(nextWordProp)):
            node = nextWordProp[i]
            for j in range(len(node)):
                was = False
                for k in range(len(textList)):
                    if textList[node[j][0]][0] == '' or textList[k][0] == '':
                        continue
                    if finalList[node[j][1]] == textList[k][0].split(" ")[0] and node[j][0] != k:
                        mergedWords.append([textList[node[j][0]][0] + " " + textList[k][0]])
                        textList[k][0] = ''
                        textList[node[j][0]][0] = ''
                        boolean = False
                        was = True
                        break
                if was:
                    break
        if boolean or debug == 50:
            break
        debug = debug + 1
    for i in textList:
        if i[0] != '':
            mergedWords.append([i[0]])
    return mergedWords

def mergeWordsWithNeuralNet(textList, nn, model, tokenizer, max_length, precision):
    debug = 1000
    while True:
        print(debug)
        change = False
        for i in range(int(len(textList))):
            if textList[i][0] == '' or len(textList[i][0].split()) == 0:
                continue
            temp = textList[i][0]
            nextWord = nn.generate_seq(model, tokenizer, max_length, temp, precision)
            isIn = False
            index = -1
            for j in range(int(len(textList))):
                if len(textList[j][0].split()) == 0:
                    continue
                t = textList[j][0].split()[0]
                if t == nextWord or t[0:len(t)-1] == nextWord:
                    isIn = True
                    index = j
                    change = True
                    break
            if isIn and index != -1:
                textList[i][0] = textList[i][0] + " " + textList[index][0]
                textList[index][0] = ''
                continue
        if not change:
            break
        temp = []
        for i in range(int(len(textList))):
            if textList[i][0] != '':
                temp.append(textList[i])
        textList = temp
        if debug == 0 or len(textList) == 1:
            break
        else:
            debug = debug - 1
    return textList

clf1, finalList = nextCharPredict()
textList = randomText()
clf2, nextWords = newMakeTreeForNextWord(32)
words = algorithm(textList, clf1, finalList)
''' create neural network '''
#nn = mp.neuralNetwork()
#data = open("./text").read()
#tokenizer = nn.getToken(data)
#sequences, vocab_size, max_length = nn.getSequences(data, tokenizer)
#model = nn.getModel(sequences, vocab_size, max_length)


''' merge words into sentences using Decision Tree '''
words = mergeWordsWithDecisionTree(words, clf2, 32, nextWords)

''' merge words into sentences using Decision Tree and Neural Network'''
#words = mergeWords(words, clf2, 32, nextWords)
#words = mergeWordsWithNeuralNet(words, nn, model, tokenizer, max_length-1, 1)

''' merge words into sentences using Neural Network '''
#words = mergeWordsWithNeuralNet(words, nn, model, tokenizer, max_length-1, 1)

print("sentences: ")
for i in words:
    print(i)
print("OK")

t1 = [j for i in words for j in i[0].split()]
t2 = open(testText).read().split()
print(stat.oneWordPercent(t1, t2))
print(stat.manyWordPercent(t1, t2, 2))
print(stat.manyWordPercent(t1, t2, 3))


#test neural network
#print('1: ', nn.generate_seq(model, tokenizer, max_length - 1, '3. Opiekunem ', 20))
#print('1: ', nn.generate_seq(model, tokenizer, max_length - 1, '4.', 20))
#print('1: ', nn.generate_seq(model, tokenizer, max_length - 1, '4. Student ', 20))
#print('2: ', nn.generate_seq(model, tokenizer, max_length - 1, 'Instytutu Informatyki i', 20))
#print('3: ', nn.generate_seq(model, tokenizer, max_length - 1, 'co najmniej stopień', 20))
#print('4: ', nn.generate_seq(model, tokenizer, max_length - 1, 'Recenzent pracy', 20))