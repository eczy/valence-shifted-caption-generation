import pickle

# Naive Bayes Model
# assume conditional independence
#   This model is a black box that is used to find the probabilty given a
# word pair and a sentiment class
class sentimentModel:

    # - number of times sentiment group occurs (C)
    _classCounts_map = {}

    # - number of times word + sent group occours ( N(f=w_i, C=c_j) )
    _wordClassCount_tupMap = {}

    # - set of all seen pairs
    _allPairs_tupSet = set()

    # - number of pairs (non-unique)
    _pairCount_int = 0

    # - number of words in the vocab (unique)
    _uniqueWordCount = 0

    # basic constructor
    #   Loads in necessary information from pickle count file in
    # anticipation of method calls
    def __init__(self):
        # open the data counts map from the imdb training data
        countsMap = {}
        with open('trainCounts_imdb.pkl', 'rb') as trainCountsFile:
            countsMap = pickle.load(trainCountsFile)

        self._classCounts_map         = countsMap["classCounts"]
        self._wordClassCount_tupMap   = countsMap["wordClassCount"]
        self._allPairs_tupSet         = countsMap["allPairs"]
        self._pairCount_int           = countsMap["pairCount"]
        self._uniqueWordCount                = countsMap["_uniqueWordCount"]



    # return the probability of the class given the words
    # P(c) * PROD_i( P(x_i | c) )
    def P_c_givenW0W1(self, sentClass, w0, w1):
        # P(c) = N(C = c) / Number of classes
        Pc = self._classCounts_map[sentClass] / len(self._classCounts_map.keys())

        # if never seen before word, set count to zero and use smoothing
        if (w0,sentClass) not in self._wordClassCount_tupMap:
            self._wordClassCount_tupMap[(w0,sentClass)] = 0
        if (w1,sentClass) not in self._wordClassCount_tupMap:
            self._wordClassCount_tupMap[(w1,sentClass)] = 0

        P_w0_c = ( self._wordClassCount_tupMap[(w0,sentClass)] + 1 ) \
                  / ( self._classCounts_map[sentClass] + self._uniqueWordCount )
        P_w1_c = ( self._wordClassCount_tupMap[(w1,sentClass)] + 1 ) \
                  / ( self._classCounts_map[sentClass] + self._uniqueWordCount )

        P_c_w0w1 = Pc * P_w0_c * P_w1_c

        return P_c_w0w1


    # return the class predicted by the model given only the words
    # c = argmax [ P_ci_givenW0W1 ] for all ci
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