#   This file takes in the modified sentences + their desired valance and
# evaluates the results based on sentiment

import sys
import pickle
from textblob import TextBlob


def convertSentFloatToClass(sentFloat):
    if ( sentFloat <  0.0):
        return "neg"
    elif ( sentFloat ==  0.0):
        return "neut"
    elif ( sentFloat <= 1.0):
        return "pos"
    else:
        print("Illegal sentiment float value seen." + str(sentFloat))
        return "ERROR"

# def convert5ClassTo3Class(aClass):
#     if aClass == "vNeg" or aClass == "pNeg":
#         return "neg"
#     if aClass == "vPos" or aClass == "pPos":
#         return "pos"
#     if aClass == "neut":
#         return "neut"

if __name__ == '__main__':
    # load the caption + desired valance tuples
    with open(sys.argv[1], 'rb') as inFile:
        results = pickle.load(inFile)

    positiveSentences = [(i,results[i]['pos']) for i in results if 'pos' in results[i]]
    negativeSentences = [(i,results[i]['neg']) for i in results if 'neg' in results[i]]

    results = [(cap[0], cap[1], 'pos') for cap in positiveSentences] + [(cap[0], cap[1], 'neg') for cap in negativeSentences]

    predCount = 0.0 # for accuracy calculation
    predCount_correct = 0.0 # for accuracy calculation
    classPredCounts = {} # for precision calculation
    classPredCounts_correct = {} # for precision calculation
    misTaggedExamples = {}

    for r in results:
        img = r[0]
        modCaption = r[1]
        desiredSentimentClass = r[2]

        captDeats = TextBlob(modCaption)

        actualSentiment = captDeats.sentiment.polarity
        actualSentimentClass = convertSentFloatToClass(actualSentiment)

        # initialize counts in map if needed
        if actualSentimentClass not in classPredCounts:
            classPredCounts[actualSentimentClass] = 0.0
            classPredCounts_correct[actualSentimentClass] = 0.0

        # increment counts of predictions
        predCount += 1.0
        classPredCounts[actualSentimentClass] += 1.0

        # increment correct counts if correct
        # correct here meaning just it is negative, positve, or neutral
        if actualSentimentClass == desiredSentimentClass:
            predCount_correct += 1.0
            classPredCounts_correct[actualSentimentClass] += 1.0

        else:
            if img not in misTaggedExamples:
                misTaggedExamples[img] = {}
                misTaggedExamples[img][desiredSentimentClass] = modCaption
                print('Wrong Generate Class: {} | Caption: {}'.format(desiredSentimentClass, modCaption))

    # print accuracy and precisions
    print("Accuracy: " + str(float(predCount_correct) / float(predCount)))
    for c in classPredCounts.keys():
        print(c + " precision: " + str(float(classPredCounts_correct[c]) / float(classPredCounts[c])))

    # save all results in a pickle of a map
    evaluationMap = {}
    evaluationMap["predCount"] = predCount
    evaluationMap["predCount_correct"] = predCount_correct
    evaluationMap["classPredCounts"] = classPredCounts
    evaluationMap["classPredCounts_correct"] = classPredCounts_correct

    with open('evaluationCounts.pkl', 'wb') as outFile:
        pickle.dump(evaluationMap, outFile)
