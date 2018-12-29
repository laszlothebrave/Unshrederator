from PIL import Image
import PIL.Image
from random import shuffle
import itertools
from pytesseract import image_to_string
import pytesseract
import copy
l = []
# def combinations(target,data):
#     global l
#     for i in range(len(data)):
#         new_target = copy.copy(target)
#         new_data = copy.copy(data)
#         new_target.append(data[i])
#         new_data = data[i+1:]
#         # print("".join(new_target))
#         string = "".join(new_target)
#         if string in text:
#             if string not in l: 
#                 l.append(string)
#                 print(l)
#         combinations(new_target, new_data)

def permutation(lst): 
    global l
    if len(lst) == 0: 
        return []  
    if len(lst) == 1: 
        return [lst]
    tt = []    
    temp = []
    for i in range(len(lst)): 
        m = lst[i] 
        remLst = lst[:i] + lst[i+1:]  
        for p in permutation(remLst): 
            temp = [m] + p
            string = "".join(temp)
            print(string)
            if string in text:
                tt.append([m] + p)
                if string not in l: 
                    l.append(string)
        # print(l)
    
    return tt

def load_words():
    with open('words.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words

def wordCombiner(x):
    permutation(x)
    print(l)

def startWordCombiner(x):
    temp = []
    for i in x:
        if (i.isalpha() or i.isdigit()):
            temp.append(i)
    wordCombiner(temp)

text = load_words()
pytesseract.pytesseract.tesseract_cmd = 'C:\\YOUR-PATH\\tesseract'
TESSDATA_PREFIX = 'C:/YOUR-PATH/Tesseract-OCR'
output = pytesseract.image_to_string(PIL.Image.open('C:\\YOUR-PATH\\book.jpg').convert("RGB"), lang='eng')
output = list(output)
shuffle(output)
startWordCombiner(output)