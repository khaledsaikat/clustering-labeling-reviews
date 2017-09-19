#!/usr/bin/env python3
from collections import Counter
from textblob import TextBlob
from pprint import pprint
import helpers
import data_loader as dl

class Annotation:

        reviews = []

        rawReviews = []

        reviewsByGroup = {}

        topAspects = None


        def loadReviewsByLine(self, path):
            self.reviews = dl.loadJsonByLine(path)


        def loadRawReviews(self, path="../data/reviews_Electronics_5.json"):
            '''Load downloaded raw reviews'''
            self.rawReviews = dl.loadJsonByLine(path)
            self.reviewsByGroup = helpers.groupByProducts(self.rowReviews)


        def showTopCountReviews(self):
            '''Get products id which contains highest number of reviews
            Run loadRawReviews() before running this method
            '''

            c = Counter({k:len(v) for k,v in self.reviewsByGroup.items()})
            print(c.most_common(20))


        def writeAllReviewsByProduct(self, path, asin):
            '''Write selected product reviews to a file
            Run loadRawReviews() before running this method
            '''

            dl.writeJsonByLine(self.reviewsByGroup[asin], path)


        def filterReviews(self, minLength=150, maxLength=1500):
            ''' Filter reviews based on reviews length'''
            self.reviews = [v for v in self.reviews if len(v["reviewText"])>=minLength and len(v["reviewText"])<=maxLength ]


        def addAspectPlaceholder(self):
            '''Add aspect dict placeholder to each reviews'''
            reviews = []
            for review in self.reviews:
                 review["aspects"] = {}
                 reviews.append(review)
            self.reviews = reviews


        def splitText(self):
            '''Split text by line and store splited json to reviewTextJson key'''
            reviews = []
            for review in self.reviews:
                blob = TextBlob(review["reviewText"])
                textDict = {}
                for i,line in enumerate(blob.sentences):
                    textDict[i] = str(line)
                review["reviewTextJson"] = textDict
                reviews.append(review)
            self.reviews = reviews


        def putAspectsToLast(self):
            '''Change order of aspect key'''
            reviews = []
            for review in self.reviews:
                aspects = review["aspects"]
                del review["aspects"]
                review["aspects"] = aspects
                reviews.append(review)
            self.reviews = reviews


        def writeReviews(self, path):
            '''Write reviews to file'''
            dl.writeJsonToFile(self.reviews, path)


        def filterAddAspectWrite(self, path):
            '''Write first 100 reviews after length filtered and aspect placeholder added
            Run loadReviewsByLine() before running this method
            '''

            self.filterReviews()
            self.addAspectPlaceholder()
            dl.writeJsonByLine(self.reviews[0:100], path)


        def getManualAspects(self):
            aspects = []
            [aspects.extend(v["aspects"]) for v in self.reviews]
            return Counter(aspects)


        def _setTopAspects(self, minCount=5):
            aspects = self.getManualAspects()
            self.topAspects = Counter({k: v for k, v in dict(aspects).items() if v >= minCount})


        def showManualAspects(self, minCount=5):
            aspects = self.getManualAspects()
            print("Total count: ", len(aspects))
            print(aspects)

            self._setTopAspects(minCount)
            top = Counter({k:v for k,v in dict(aspects).items() if v >= minCount})
            print("\nTop count {} with min {} reviews".format(len(self.topAspects), minCount))
            pprint(elf.topAspects)


        def mergeAllReviewsToLines(self):
            '''Merge all reviews into a single list'''
            lines = []
            for review in self.reviews:
                lines.extend(review["reviewTextJson"].values())

            return lines


        def getAllLinesAspectsDict(self):
            '''Get line as key and aspects as list in a dict for all reviews'''
            self._setTopAspects()
            lines = {}
            index = 0
            for review in self.reviews:
                lineAspectsDict = self.getLineAspectsDict(review)
                aspects = lineAspectsDict if lineAspectsDict else None
                for k, v in review["reviewTextJson"].items():
                    lines[index] = lineAspectsDict[int(k)] if int(k) in lineAspectsDict else None
                    index += 1

            return lines


        def showLinesAspects(self):
            lines = self.getAllLinesAspectsDict()
            pprint(lines)
            print(len(lines))


        def getAspectsGruops(self):
            '''Return aspects groups with reviews line number'''
            lines = self.getAllLinesAspectsDict()
            aspects = {}
            aspects["none"] = []
            for lineNumber, lineAspects in lines.items():
                if lineAspects:
                    for aspectName in lineAspects:
                        if not aspectName in aspects:
                            aspects[aspectName] = []
                        aspects[aspectName].append(lineNumber)
                else:
                    aspects["none"].append(lineNumber)

            return aspects


        def getAspectsGruopsText(self):
            aspects = self.getAspectsGruops()
            lines = self.mergeAllReviewsToLines()
            for aspectName, lineNumbers in aspects.items():
                for index, lineNumber in enumerate(lineNumbers):
                    aspects[aspectName][index] = lines[lineNumber]
            #pprint(aspects)
            return aspects


        def getLineAspectsDict(self, review):
            '''Get line as key and aspects as list in a dict for a given review'''
            lineAspectsDict = {}
            for aspectName, aspectLines in review["aspects"].items():
                for aspectLine in aspectLines:
                    if not aspectLine in lineAspectsDict:
                        lineAspectsDict[aspectLine] = []
                    if aspectName in self.topAspects.keys():
                        lineAspectsDict[aspectLine].append(aspectName)

            return lineAspectsDict


        def replaceTextWithNumber(self):
            '''Replace text in aspects with reviews line number'''
            reviews = []
            for review in self.reviews:
                for aspectName, aspectText in review["aspects"].items():
                    if type(aspectText) == str:
                        lineIndex = self._getReviewIndex(aspectText, review["reviewTextJson"])
                        if lineIndex != False:
                            review["aspects"][aspectName] = [int(lineIndex)]
                reviews.append(review)
            self.reviews = reviews


        def _getReviewIndex(self, s, reviewsDict):
            '''Get line number of review'''
            for line, text in reviewsDict.items():
                if text.find(s) != -1:
                    return line
            return False



if __name__ == "__main__":
    a = Annotation()
    #a.loadRawReviews()
    #a.showTopCountReviews()
    a.reviews = dl.loadJsonFromFile("../data/headphone100.json")
    #a.splitText()
    #a.putAspectsToLast()
    #a.replaceTextWithNumber()
    #a.writeReviews("../data/s100.json")
    #a.showManualAspects()
    #a.showLinesAspects()
    print(a.getAspectsGruopsText())
    #dl.writeJsonToFile(a.getAspectsGruopsText(), "../data/headphone_aspects.json")
