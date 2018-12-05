import os
import pickle
import sys
import gzip
from stanfordcorenlp import StanfordCoreNLP

import parse_imdbData # use many of the same functions

amazonDataDir = "./amazonRawData"
amazonSentenceTupDir = "./amazon_sentenceTuples"
amazonPairTupDir = "./amazon_pairTuples"
amazonCountsDir = "./amazon_counts"

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)



# to match imdb parsing, we need to save a pickle of a list of tuples (text, sentiment)
def getTrainSentenceTuples():

    # list of all file paths to taining data (reviews)
    allTrainFiles = []

    for gzFile in os.listdir(amazonDataDir):

        # list of tuples (reviewText, sentiment) for all reviews
        trainData = []

        # notify the users how the data will be grouped
        if "-categories" in sys.argv:
            print("Grouping sentiments into 5 categories.")
        else:
            print("Grouping sentiments by float score.")

        reviewsFromGZ = parse(amazonDataDir+"/"+gzFile)
        print("Processing Amazon reviews from " + gzFile)

        # print("Opening " + f + " - length = " + str(len(trainData)))
        for review in reviewsFromGZ:
            reviewText = parse_imdbData.prepReviewText(review['reviewText'])

            sentiment = float( review['overall'] )
            sentiment = convert5PointScaleTo1Point(sentiment)

            trainData.append( (reviewText,sentiment) )


        if len(trainData) > 0:
            # save the training tuples
            with open(amazonSentenceTupDir+'/trainSentenceTuples_amazon_'+gzFile+'.pkl', 'wb') as trainDataFile:
                pickle.dump(trainData, trainDataFile)



def convert5PointScaleTo1Point(floatSent):
    floatSent = floatSent-3 # now ranges -2 to +2
    floatSent = floatSent / 2 # now ranges from -1 to +1

    if "-categories" in sys.argv:
        cat = ""
        if floatSent <= -0.7:
            cat = "vNeg"
        elif floatSent < 0.0:
            cat = "pNeg"
        elif floatSent == 0.0:
            cat = "neut"
        elif floatSent <= 0.7:
            cat = "pPos"
        elif floatSent <= 1.0:
            cat = "vPos"
        return cat
    else:
        return floatSent



def getTrainPairTuples():

    # list of tuples (pairType, word0, word1, sentiment) for all pairs in all reviews
    allPairsWithSentiments = []

    for tupInFile in os.listdir(amazonSentenceTupDir):
        # load the sentence tuples
        with open(amazonSentenceTupDir+"/"+tupInFile, 'rb') as trainTuplesIn:
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
                     " training reviews parsed into pairs for "+tupInFile+".")

            # get tuples for this review for every word pair
            reviewPairs = parse_imdbData.findPairs(trainData[i][0], nlp) # StanfordCoreNLP

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
        with open('trainPairTuples_amazon_'+tupInFile+'.pkl', 'wb') as trainPairsDataFile:
            pickle.dump(allPairsWithSentiments, trainPairsDataFile)


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

    for tupInFile in os.listdir(amazonPairTupDir):
        pairTuples = []
        with open(amazonPairTupDir+"/"+tupInFile, 'rb') as trainPairsDataFile:
            pairTuples = pickle.load(trainPairsDataFile)

        for p in pairTuples:

            # - number of times sentiment group occurs (C)
            sentimentClass = convert5CatTo2Cat(p[3])
            if sentimentClass not in classCounts:
                classCounts[sentimentClass] = 0
            classCounts[sentimentClass] += 1

            # - number of times word + sent group occours ( N(f=w_i, C=c_j) )
            wordSentTup0 = (p[1],convert5CatTo2Cat(p[3]))
            wordSentTup1 = (p[2],convert5CatTo2Cat(p[3]))
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


    # save the counts
    # if the data is broken into sentiment categories, save to different file
    if "vNeg" in finalMap["classCounts"].keys() or "neg" in finalMap["classCounts"].keys():
        with open(amazonCountsDir+"/trainCounts_cats_amazon.pkl", 'wb') as trainCountsFile:
            pickle.dump(finalMap, trainCountsFile)
    else:
        with open(amazonCountsDir+"/trainCounts_amazon.pkl", 'wb') as trainCountsFile:
            pickle.dump(finalMap, trainCountsFile)


def convert5CatTo3Cat(cat):
    if cat == "vNeg" or cat == "pNeg":
        return "neg"
    elif cat == "vPos" or cat == "pPos":
        return "pos"
    elif cat == "neut":
        return "neut"
    else:
        raise ValueError('An illegal category was seen in the 5cat to 3cat converter.\
                            cat = '+cat)


def convert5CatTo2Cat(cat):
    if cat == "vNeg" or cat == "pNeg":
        return "neg"
    elif cat == "vPos" or cat == "pPos":
        return "pos"
    elif cat == "neut":
        return "neg"
    else:
        raise ValueError('An illegal category was seen in the 5cat to 2cat converter.\
                            cat = '+cat)



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
