from stanfordcorenlp import StanfordCoreNLP
import json
import sys
import os
import pickle

# this function prepares the text from a review for parsing
# it removes html format text (i.e. <br /><br />)
# note: cannot yet remove capitlization or punctuation because StanfordCoreNLP
#        requires those features for its parsing
def prepReviewText(text):

    # remove formating text inside of < >
    startIdx = text.find("<")
    endIdx = text.find(">")
    while startIdx != -1 and endIdx != -1:
        text = text[: startIdx] + text[endIdx+1 :]
        startIdx = text.find("<")
        endIdx = text.find(">",startIdx)


    # remove escape characters (backslashes)
    escIdx = text.find("\\")
    while escIdx != -1:
        text = text[: escIdx] + text[escIdx+1 :]
        escIdx = text.find("\\", escIdx-1)

    # remove astriks (*)
    astIdx = text.find("*")
    while astIdx != -1:
        text = text[: astIdx] + text[astIdx+1 :]
        astIdx = text.find("*", astIdx-1)

    return text

#   This function gets all of the training tuples from the imdb data files
# and saves them in a pickle
#   This function only needs to be rerun if the pickle needs to be
# regenerated
def getTrainSentenceTuples():
    # list of all file paths to taining data (reviews)
    allTrainFiles = []

    allPosTrainFiles = os.listdir("aclImdb/train/pos/")
    for f in allPosTrainFiles: allTrainFiles.append( "aclImdb/train/pos/"+f )
    del allPosTrainFiles
    allNegTrainFiles = os.listdir("aclImdb/train/neg/")
    for f in allNegTrainFiles: allTrainFiles.append( "aclImdb/train/neg/"+f )
    del allNegTrainFiles

    # list of tuples (reviewText, sentiment) for all reviews
    trainData = []

    # full learning set is too large for memory, break it up by 5000 tuples
    fileCount = 0

    for f in allTrainFiles:
        with open(f, 'r') as inFile:
            # print("Opening " + f + " - length = " + str(len(trainData)))
            review = inFile.readline()
            review = prepReviewText(review)

            sentiment = int( f.split("_")[1].split(".")[0] )
            sentiment = convert10PointScaleTo1Point(sentiment)

            trainData.append( (review,sentiment) )

            # if len(trainData) % 50000 == 0:
            #     # save the training tuples
            #     with open('trainSentenceTuples_imdb_'+str(fileCount)+'.pkl', 'wb') as trainDataFile:
            #         pickle.dump(trainData, trainDataFile)
            #
            #     print("Pickle number " + str(fileCount) + "saved.")
            #     trainData = []
            #     fileCount += 1

    if len(trainData) > 0:
        # save the training tuples
        with open('trainSentenceTuples_imdb.pkl', 'wb') as trainDataFile:
            pickle.dump(trainData, trainDataFile)

#   This function gets all pairs from all training data and saves a list
# of the tuples (pairType, word0, word1, sentiment)
def getTrainPairTuples():

    # load the sentence tuples
    with open('trainSentenceTuples_imdb.pkl', 'rb') as trainTuplesIn:
        trainData = pickle.load(trainTuplesIn)

    # list of tuples (pairType, word0, word1, sentiment) for all pairs in all reviews
    allPairsWithSentiments = []

    # this is the object used to parse the sentences for POS / dependencies
    # load just this once to improve runtime
    nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')

    # Go through all training review text and find pairs + pair info
    for i in range(len(trainData)):
        # show status in terminal
        if i%100 == 0:
            print(str(i) + "/"+str(len(trainData))+ \
                 " training reviews parsed into pairs.")

        # get tuples for this review for every word pair
        reviewPairs = findPairs(trainData[i][0], nlp) # StanfordCoreNLP

        # append the review rating sentiment to every word pair tuple
        for j in range(len(reviewPairs)):
            temp = reviewPairs[j]
            reviewPairs[j] = (temp[0].lower(), # word pair type
                              temp[1].lower(), # word0
                              temp[2].lower(), # word1
                              trainData[i][1]) # sentiment

        # append the pair+sentiment tuple to the list of all pairs+sentiments
        allPairsWithSentiments += reviewPairs


    nlp.close() # Do not forget to close! The backend server will consume a lot memery.

    # save the review pair tuples
    with open('trainPairTuples_imdb.pkl', 'wb') as trainPairsDataFile:
        pickle.dump(allPairsWithSentiments, trainPairsDataFile)


#   This function processes through the train pair tuples and gets counts of
# - number of times sentiment group occurs (C)
# - number of times word + sent group occours ( N(f=w_i, C=c_j) )
# - number of words in the vocab
# - set of all seen pairs <- (for selection method, not bayes calcs)
# - number of pairs <- (for selection method, not bayes calcs)
def getTrainCounts():

    # - number of times sentiment group occurs (C)
    classCounts = {}

    # - number of times word + sent group occours ( N(f=w_i, C=c_j) )
    wordClassCount_tupMap = {}

    # - set of all seen pairs
    allPairs_tupSet = set()

    # - number of pairs (non-unique)
    pairCount = 0

    # - number of words in the vocab (unique)
    word_set = set()


    # open the review pair tuples
    pairTuples = []
    with open('trainPairTuples_imdb.pkl', 'rb') as trainPairsDataFile:
        pairTuples = pickle.load(trainPairsDataFile)


    for p in pairTuples:

        # - number of times sentiment group occurs (C)
        sentimentClass = p[3]
        if sentimentClass not in classCounts:
            classCounts[sentimentClass] = 0
        classCounts[sentimentClass] += 1

        # - number of times word + sent group occours ( N(f=w_i, C=c_j) )
        wordSentTup0 = (p[1],p[3])
        wordSentTup1 = (p[2],p[3])
        if wordSentTup0 not in wordClassCount_tupMap:
            wordClassCount_tupMap[wordSentTup0] = 0
        if wordSentTup1 not in wordClassCount_tupMap:
            wordClassCount_tupMap[wordSentTup1] = 0
        wordClassCount_tupMap[wordSentTup0] += 1
        wordClassCount_tupMap[wordSentTup1] += 1

        # - set of all seen pairs
        wordWordTup = (p[1],p[2])
        if wordWordTup not in allPairs_tupSet:
            allPairs_tupSet.add(wordWordTup)

        # - number of pairs
        pairCount += 1

        # - number of words in the vocab
        word_set.add(p[1])
        word_set.add(p[2])

    finalMap = {}
    finalMap["classCounts"] = classCounts
    finalMap["wordClassCount"] = wordClassCount_tupMap
    finalMap["allPairs"] = allPairs_tupSet
    finalMap["pairCount"] = pairCount
    finalMap["uniqueWordCount"] = len(word_set)

    # save the counts
    with open('trainCounts_imdb.pkl', 'wb') as trainCountsFile:
        pickle.dump(finalMap, trainCountsFile)


# from the annotated data get the adv-verb and adj-noun pairs of words
# input: String of sentences (corenlp will break sentences with natural english rules)
#        nlp object from StanfordCoreNLP to parse strings with
# output: list of tuples with info about pairs
#                                          t[0] = dependency type
#                                          t[1] = modified word idx (verb/noun)
#                                          t[2] = modifier word idx (adverb/adj)
def findPairs(sentences, nlp):

    properties = {'annotators': 'depparse', 'outputFormat': 'json'}

    # this is a map[string, list[map[string,list[...]]]]
    ann_dict = json.loads( nlp.annotate(sentences, properties=properties) )


    pairsInfo = []

    # for every sentence
    for sInfo in ann_dict['sentences']:
        # for every dependency
        for d in sInfo['basicDependencies']:
            if d['dep'] == 'advmod' or d['dep'] == 'amod': # if adverb-verb or adj-noun pair
                pairsInfo.append( (d['dep'],d['governorGloss'],d['dependentGloss']) )

    return pairsInfo

# this function converts a rank on a 10 points scale (1 to 10) to the
# 1 point (-1 to 1) score used to define sentiment
def convert10PointScaleTo1Point(score):
    if score == 5:
        score = 0
    elif score < 5:
        score = score - 6
    else:
        score = score - 5
    return score/5

# input file format should be one sentence per line with proper punctuation at
#  the end of sentences
if __name__ == '__main__':

    # if newData flag is included, regenerate the pickle file of the tuple
    if "-newSentences" in sys.argv:
        print("Generating new train sentences tuples from the imdb dataset")
        getTrainSentenceTuples()
    else:
        print("Using existing train sentences tuples from the imdb dataset")


    if "-newPairs" in sys.argv:
        print("Generating new train pair tuples from the imdb sentence tuples")
        getTrainPairTuples()
    else:
        print("Using existing train pair tuples from the imdb dataset")


    if "-newCounts" in sys.argv:
        print("Generating new counts of the imdb train pair data")
        getTrainCounts()
    else:
        print("Using existing count data of the imdb train pair data")
