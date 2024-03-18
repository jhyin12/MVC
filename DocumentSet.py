import json
from Document import Document
import numpy as np


class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0
        self.documents = []
        self.mean_embedding = None
        with open(dataDir) as input:
            line = input.readline()
            while line:
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                embedding = np.array(obj['embedding'])
                if self.mean_embedding is None:
                    self.mean_embedding = np.zeros_like(embedding, dtype=np.float64)
                self.mean_embedding += embedding
                document = Document(text, wordToIdMap, wordList, int(obj['tweetId']), embedding)
                self.documents.append(document)
                line = input.readline()

        self.mean_embedding /= self.D
        print("number of documents is ", self.D)
