# This file implements the class used to create data for probabilisitic model

import pickle

# This class is used as part of the custom word bigram model.
# It tracks and computes using counts and probabilities of unigrams and bigrams
# (Originally implemented a Naive Bayes Model, hence the commented out NB code)
class sentimentModel:

    # - number of times sentiment group occurs (C)
    _classCounts_map = {}

    # - number of times word + sent group occours ( N(f=w_i, C=c_j) )
    _wordClassCount_tupMap = {}

    # - set of all seen pairs
    allPairs_tupSet = set()

    # - number of pairs (non-unique) (bigrams)
    pairCount_int = 0

    # - number of unigrams
    uniCount_int = 0

    # - number of words in the vocab (unique)
    _uniqueWordCount = 0

    # - 2D map of noun -> adjectives -> pair count
    nounAdjCount_map = {}

    # - 2D map of verb -> adverb -> pair count
    verbAdvCount_map = {}

    # basic constructor
    #   Loads in necessary information from pickle count file in
    # anticipation of method calls
    def __init__(self):
        # open the data counts map from the imdb training data
        countsMap = {}
        # with open('trainCounts_imdb.pkl', 'rb') as trainCountsFile:
        with open('amazon_counts/trainCounts_cats_amazon.pkl', 'rb') as trainCountsFile:
            countsMap = pickle.load(trainCountsFile)

        self._classCounts_map         = countsMap["classCounts"]
        self._wordClassCount_tupMap   = countsMap["wordClassCount"]
        self.allPairs_tupSet         = countsMap["allPairs"]
        self.pairCount_int           = countsMap["pairCount"]
        self.uniCount_int            = countsMap["numUnigrams"]
        self._uniqueWordCount         = countsMap["uniqueWordCount"]
        self.unigramCount_map         = countsMap["unigramCount"]
        self.nounAdjCount_map         = countsMap["nounAdjCount"]
        self.verbAdvCount_map         = countsMap["verbAdvCount"]



    # return the probability of the class given the words
    # P(c) * PROD_i( P(x_i | c) )
    def P_c_givenW0W1(self, sentClass, w0, w1):
        # P(c) = N(C = c) / Number of samples (N)
        N = 0
        for numSampsInClass in self._classCounts_map.values():
            N += numSampsInClass

        Pc = self._classCounts_map[sentClass] / N


        # if never seen before word, set count to zero and use smoothing
        if (w0,sentClass) not in self._wordClassCount_tupMap:
            self._wordClassCount_tupMap[(w0,sentClass)] = 0
        if (w1,sentClass) not in self._wordClassCount_tupMap:
            self._wordClassCount_tupMap[(w1,sentClass)] = 0

        P_w0_c = ( self._wordClassCount_tupMap[(w0,sentClass)] + 1 ) \
                  / ( self._classCounts_map[sentClass] + self._uniqueWordCount )
        # TEST DEBUG - Ignore the noun / verb
        P_w1_c = ( self._wordClassCount_tupMap[(w1,sentClass)] + 1 ) \
                  / ( self._classCounts_map[sentClass] + self._uniqueWordCount )
        # P_w1_c = 1.0

        P_c_w0w1 = Pc * P_w0_c * P_w1_c

        # print("Prob of class " + sentClass + " = " + str(P_c_w0w1))

        # NB Output - Return P(w | c) to avoid bias towards pos due to size
        # return P_c_w0w1

        # Custom output
        return P_w0_c


    # return the class predicted by the model given only the words
    # c = argmax [ P_ci_givenW0W1 ] for all ci
    # w0 = adj/adverb , w0 = noun/verb
    def predictedClass(self, w0, w1):

        # map of class -> probability it is this class
        probMap = {}

        # go through each class and get its prob
        for k in self._classCounts_map.keys():
            probMap[k] = self.P_c_givenW0W1(k, w0, w1)

        # return the class of argmax
        maxP = 0.0
        argMaxC = -2
        for c in probMap.keys():
            if probMap[c] > maxP:
                maxP = probMap[c]
                argMaxC = c

        return argMaxC


    # return confidence of this guess
    # 0.0 == equal chance of it being right or wrong
    # < 0.0 == prediciton is increasingly incorrect
    # > 0.0 == prediction is increasinly correct
    def predConfidence(self, sentClass, w0, w1):
        # classProbs = {}
        # for k in self._classCounts_map.keys():
        #     classProbs[k] = 0.0

        posProb = self.P_c_givenW0W1('pos', w0, w1)
        negProb = self.P_c_givenW0W1('neg', w0, w1)

        confidence = 0
        if sentClass == 'neg':
            confidence = negProb / posProb
        elif sentClass == 'pos':
            confidence = posProb / negProb

        return (confidence-1)
