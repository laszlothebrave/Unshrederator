class TextService(object):
    def __init__(self):
        self.text = []

    def setText(self, text):
        self.text = text

    def splitText(self, text):
        return list(set(text.split()))

    def readFile(self, filename):
        with open("./" + filename, 'r') as f:
            return f.read()

    def saveTextInFile(self, filename):
        with open("./" + filename, 'w') as f:
            string = ""
            for i in self.text:
                string = string + i + "\n"
            f.write(string)
            f.close()
