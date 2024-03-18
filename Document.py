class Document:

    def __init__(self, text, wordToIdMap, wordList, documentID, embedding):
        self.documentID = documentID
        self.wordIdArray = []
        self.wordFreArray = []
        self.embedding = embedding
        V = len(wordToIdMap)
        wordFreMap = {}
        ws = text.strip().split(' ')
        for w in ws:
            if w not in wordToIdMap:
                wId = V
                wordToIdMap[w] = V
                wordList.append(w)
                V += 1
            else:
                wId = wordToIdMap[w]

            if wId not in wordFreMap:
                wordFreMap[wId] = 1
            else:
                wordFreMap[wId] = wordFreMap[wId] + 1
        self.wordNum = wordFreMap.__len__()
        w = 0
        for wfm in wordFreMap:
            self.wordIdArray.append(wfm)
            self.wordFreArray.append(wordFreMap[wfm])
            w += 1
