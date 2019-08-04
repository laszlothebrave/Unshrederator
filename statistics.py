def oneWordPercent(text, testText):
    print(text)
    print(testText)
    length = len(text)
    l = 0
    for i in testText:
        if i in text:
            l = l + 1
    return l/length

def manyWordPercent(text, testText, size):
    length = len(text)-size
    l = 0
    for i in range(length):
        equal = True
        for j in range(len(testText)-size):
            equal = True
            for k in range(size):
                if text[i+k] != testText[j+k]:
                    equal = False
                    break
            if equal:
                l = l + 1
                break
    return l/length
