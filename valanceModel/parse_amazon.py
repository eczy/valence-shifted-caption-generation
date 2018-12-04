import os
import pickle
import sys
import gzip
from stanfordcorenlp import StanfordCoreNLP

import parse_imdbData # use many of the same functions

amazonDataDir = "./amazonRawData"
amazonSentenceTupDir = "./amazon_sentenceTuples"

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
