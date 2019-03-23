import PIL
import pytesseract as pytesseract
from PIL import Image
from collections import OrderedDict

FILE_TXT = "C:\\Users\\369254\\PycharmProjects\\szczypka\\text"
FILE_BMP = 'C:\\Users\\369254\\PycharmProjects\\szczypka\\Dokument.bmp'


def getDiff(source_file, dest_file):
    pass

def getText(source_file):
    with open(source_file) as f:
        str = f.read()
        characters = list(str)
    return "".join(OrderedDict.fromkeys(sorted(characters)))

def BMPToFile(source_path):
    output = pytesseract.image_to_string(PIL.Image.open(source_path).convert("RGB"), lang='eng')
    output = list(output)
    return output

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

#etykieta - litera
#vector - future :
    #zawiera informacje o kombinacjach z tą literą
    #np. następna, poprzednia, dwie następne, dwie porzednie
    #niskopoziomowe klasyfikatory oraz wyskokopoziomowe

def treeMaker(source_path):
    pass
print(getText(FILE_TXT))
list_of_chars = BMPToFile(FILE_BMP)
print(charAfterChar(FILE_TXT, list_of_chars))