## WARNING: Deprecated file. This file was originally used while the team was
##            still using imdb data. However, after making the switch to Amazon
##            data, this file was no longer maintained and may not entirely
##            work with more recent features or additions to the system.
# This file is used to parse amazon dataset data into a usable format.
# It takes in the raw data (as a review with metadata) and generates a pickle
#   containing the Noun-Adj and Verb-Adv pairs and their respective, scaled
#   review score. It also generates intermediate forms of the data as pickles.

from stanfordcorenlp import StanfordCoreNLP
import json
import sys
import os
import pickle


# Trim uneeded review metadata and create a pickle of a list of tuples
#   where a tupe is (full review text, review sentiment from stars)
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

    # notify the users how the data will be grouped
    if "-categories" in sys.argv:
        print("Grouping sentiments into 5 categories.")
    else:
        print("Grouping sentiments by float score.")

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


# Convert a 5 start (1 to 5) review sentiment to the 1 point (-1 to +1)
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


# from the annotated data get the adv-verb and adj-noun pairs of words
# input: String of sentences (corenlp will break sentences with natural english rules)
#        nlp object from StanfordCoreNLP to parse strings with
# output: list of tuples with info about pairs
#                                          t[0] = dependency type
#                                          t[1] = modified word idx (verb/noun)
#                                          t[2] = modifier word idx (adverb/adj)
def findPairs(sentences, nlp):

    properties = {'annotators': 'depparse,lemma', 'outputFormat': 'json'}

    # this is a map[string, list[map[string,list[...]]]]
    ann_dict = json.loads( nlp.annotate(sentences, properties=properties) )


    pairsInfo = []

    # for every sentence
    for sInfo in ann_dict['sentences']:
        # for every dependency
        for d in sInfo['enhancedPlusPlusDependencies']:
            if d['dep'] == 'advmod' or d['dep'] == 'amod': # if adverb-verb or adj-noun pair
                modifiedIdx = d['governor'] # index of noun/verb
                modifierIdx = d['dependent'] # index of adj/adv
                lemmatizedModifiedWord = sInfo['tokens'][modifiedIdx - 1]['lemma']
                lemmatizedModifierWord = sInfo['tokens'][modifierIdx - 1]['lemma']
                pairsInfo.append( (d['dep'],lemmatizedModifiedWord,lemmatizedModifierWord) )

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

    score = score/5

    if "-categories" in sys.argv:
        cat = ""
        if score <= -0.7:
            cat = "vNeg"
        elif score < 0.0:
            cat = "pNeg"
        elif score == 0.0:
            cat = "neut"
        elif score <= 0.7:
            cat = "pPos"
        elif score <= 1.0:
            cat = "vPos"
        return cat
    else:
        return score



#   This function processes through the train pair tuples and gets counts of
# - number of times sentiment group occurs (C)
# - number of times word + sent group occours ( N(f=w_i, C=c_j) )
# - number of words in the vocab
# - set of all seen pairs                       <- (for selection method, not bayes calcs)
# - number of pairs                             <- (for selection method, not bayes calcs)
# - 2D map of noun -> adjectives -> pair count  <- (for selection method, not bayes calcs)
# - 2D map of verb -> adverb -> pair count      <- (for selection method, not bayes calcs)
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

    # - 2D map of noun -> adjectives -> pair count
    nounAdjCount_map = {}

    # - 2D map of verb -> adverb -> pair count
    verbAdvCount_map = {}


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

        # - 2D map of noun -> adjectives -> pair count
        # - 2D map of verb -> adverb -> pair count
        modifiedWord = p[1]
        modifierWord  = p[2]
        relationship = p[0]
        if relationship == "amod":

            # if noun not yet in map, add it as empty map
            if modifiedWord not in nounAdjCount_map:
                nounAdjCount_map[modifiedWord] = {}

            # if adj not yet in [noun] map, add it as 0 count int
            if modifierWord not in nounAdjCount_map[modifiedWord]:
                nounAdjCount_map[modifiedWord][modifierWord] = 0

            # increment the count of this noun-adj pair
            nounAdjCount_map[modifiedWord][modifierWord] += 1

        elif relationship == "advmod":

            # if verb not yet in map, add it as empty map
            if modifiedWord not in verbAdvCount_map:
                verbAdvCount_map[modifiedWord] = {}

            # if adverb not yet in [verb] map, add it as 0 count int
            if modifierWord not in verbAdvCount_map[modifiedWord]:
                verbAdvCount_map[modifiedWord][modifierWord] = 0

            # increment the count of this adverb-adj pair
            verbAdvCount_map[modifiedWord][modifierWord] += 1


    finalMap = {}
    finalMap["classCounts"] = classCounts
    finalMap["wordClassCount"] = wordClassCount_tupMap
    finalMap["allPairs"] = allPairs_tupSet
    finalMap["pairCount"] = pairCount
    finalMap["uniqueWordCount"] = len(word_set)
    finalMap["nounAdjCount"] = nounAdjCount_map
    finalMap["verbAdvCount"] = verbAdvCount_map


    import pdb; pdb.set_trace()

    # save the counts
    # if the data is broken into sentiment categories, save to different file
    if "vNeg" in finalMap["classCounts"].keys():
        with open('trainCounts_cats_imdb.pkl', 'wb') as trainCountsFile:
            pickle.dump(finalMap, trainCountsFile)
    else:
        with open('trainCounts_imdb.pkl', 'wb') as trainCountsFile:
            pickle.dump(finalMap, trainCountsFile)



# The main function processes command line flags to decide which part(s) of
#   the parsing process to run
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
