#   This file takes in the modified sentences + their desired valance and
# evaluates the results based on sentiment

import sys
import pickle
from textblob import TextBlob


def convertSentFloatToClass(sentFloat):
    if ( sentFloat <= -0.5):
        return "vNeg"
    elif ( sentFloat <  -0.0):
        return "pNeg"
    elif ( sentFloat ==  0.0):
        return "neut"
    elif ( sentFloat <   0.5):
        return "pPos"
    elif ( sentFloat <=  1.0):
        return "vPos"
    else:
        print("Illegal sentiment float value seen." + str(sentFloat))
        return "ERROR"

def convert5ClassTo3Class(aClass):
    if aClass == "vNeg" or aClass == "pNeg":
        return "neg"
    if aClass == "vPos" or aClass == "pPos":
        return "pos"
    if aClass == "neut":
        return "neut"

if __name__ == '__main__':
    # load the caption + desired valance tuples
    with open(sys.argv[1], 'rb') as inFile:
        results = pickle.load(inFile)

    predCount = 0 # for accuracy calculation
    predCount_correct = 0 # for accuracy calculation
    classPredCounts = {} # for precision calculation
    classPredCounts_correct = {} # for precision calculation

    for r in results:
        modCaption = r[0]
        desiredSentimentClass = r[1]

        captDeats = TextBlob(modCaption)

        actualSentiment = captDeats.sentiment.polarity
        actualSentimentClass = convertSentFloatToClass(actualSentiment)

        # initialize counts in map if needed
        if actualSentimentClass not in classPredCounts:
            classPredCounts[actualSentimentClass] = 0
            classPredCounts_correct[actualSentimentClass] = 0

        # increment counts of predictions
        predCount += 1
        classPredCounts[actualSentimentClass] += 1

        # increment correct counts if correct
        # correct here meaning just it is negative, positve, or neutral
        if convert5ClassTo3Class(desiredSentimentClass)\
               == convert5ClassTo3Class(actualSentimentClass):
            predCount_correct += 1
            classPredCounts_correct[actualSentimentClass] += 1

    # print accuracy and precisions
    print("Accuracy: " + str(predCount_correct / predCount))
    for c in classPredCounts.keys():
        print(c + " precision: " + str(classPredCounts_correct[c] / classPredCounts[c]))

    # save all results in a pickle of a map
    evaluationMap = {}
    evaluationMap["predCount"] = predCount
    evaluationMap["predCount_correct"] = predCount_correct
    evaluationMap["classPredCounts"] = classPredCounts
    evaluationMap["classPredCounts_correct"] = classPredCounts_correct

    with open('evaluationCounts.pkl', 'wb') as outFile:
        pickle.dump(evaluationMap, outFile)
