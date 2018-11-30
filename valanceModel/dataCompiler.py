############################################
############################################
########## DEPRECATED - DO NOT USE ##########
############################################
############################################

from stanfordcorenlp import StanfordCoreNLP
import json
import sys


# This function takes in sentences and creates a list of tuples which associate
#  adv-verb and noun-adj pairs with the sentiment of their full sentence
# input: String of punctuation-seperated sentences
# output: list of tuples of (pairType, modifiedWord, modifierWord, sentenceSentiment)
def findPairSentimentInfo(sentences):
    allPairsInfo = []

    nlp = StanfordCoreNLP(r'/Users/drew/Programs/stanford-corenlp-full-2018-10-05', memory='8g')


    properties = {'annotators': 'depparse,sentiment', 'outputFormat': 'json'}

    # this is a map[string, list[map[string,list[...]]]]
    ann_dict = json.loads( nlp.annotate(sentences, properties=properties) )


    ordered_sentencesSentiment  = getSentiments(ann_dict)
    ordered_pairsInfo           = getPairs(ann_dict)


    # append sentiment to pair tuple
    # then append full tuple to list of all pair tuples
    PairWithSentimentInfo = []
    for i in range(len(ordered_pairsInfo)):
        for p in ordered_pairsInfo[i]:
            PairWithSentimentInfo.append(p + (ordered_sentencesSentiment[i],))


    nlp.close() # Do not forget to close! The backend server will consume a lot memery.

    return PairWithSentimentInfo

# from the annotated data, gets the sentiment of a single sentence
#  input:  annotated data of all sentences
#  output: list of avg sentiment for each sentence
def getSentiments(ann_dict):
    allAvgSentiments = []
    # this is a list of percentages of how much the sent fits a category
    # 5 categories: very negative, negative, neutral, positive, very positive
    for sInfo in ann_dict['sentences']:
        sentimentDist = sInfo['sentimentDistribution']
        sentimentAvg  = sentimentDist[0]* -1.0 +\
                        sentimentDist[1]* -0.5 +\
                        sentimentDist[2]*  0   +\
                        sentimentDist[3]*  0.5 +\
                        sentimentDist[4]*  1.0

        # to avoid getting brought off by minor calibration differences, neutral out small sentiment values
        if ( -0.3 < sentimentAvg and sentimentAvg < 0.3 ): sentimentAvg = 0.0
        allAvgSentiments.append(sentimentAvg)

    return allAvgSentiments

# from the annotated data get the adv-verb and adj-noun pairs of words
# input:  annotated data of all sentences
# output: 2D list - 1st dim: sentence number, 2nd dim: tuple info about pairs
#                                          t[0] = dependency type
#                                          t[1] = modified word idx (verb/noun)
#                                          t[2] = modifier word idx (adverb/adj)
def getPairs(ann_dict):
    pairsInfo = []

    for sInfo in ann_dict['sentences']:
        pairsInfo.append([])
        for d in sInfo['basicDependencies']:
            if d['dep'] == 'advmod' or d['dep'] == 'amod': # if adverb-verb or adj-noun pair
                # dependencies indices are one-off from token indices
                # => need -1
                pairsInfo[-1].append( (d['dep'],d['governorGloss'],d['dependentGloss']) )

    return pairsInfo


# input file format should be one sentence per line with proper punctuation at
#  the end of sentences

## DREW: Leaving off though. Right now the system generates its own semantic
##       ranking through the StanfordCoreNLP program. However, there are labeled
##       datasets available (i.e. imdb reviews) that could be used in place of
##       ranking the sentence sentiment. This may prove more accurate AND may
##       be more closely related to what we are trying to represent: the process
##       of describing something factually but with an underlying positive or
##       negative bias
if __name__ == '__main__':
    allPairsAndSentiments = []
    i = -1
    for f in sys.argv[1:]:
        with open(f, 'r') as inFile:
            lines=inFile.readlines()

            while (i >= len(lines)):
                data = ""

                # StanfordCoreNLP code fails if string too long, so break up
                #  into groups of 100 sentences
                for _ in range(100):
                    i += 1
                    if i >= len(lines): break
                    data += lines[i]

                pairsAndSentiments = findPairSentimentInfo(data)
                allPairsAndSentiments += pairsAndSentiments
                print(pairsAndSentiments)


    # Some sample sentences
    # "We briefly went to the ocen and watched the massive waves crash down on the shore.\
    #  That ugly man is aggressively banging the table.\
    #  Okay, but do you think the U.S. has the best fries?\
    #  The puppy is adorable.\
    #  A ragged, growing army of migrants resumes march toward US.\
    #  A vulnerable, growing group of mothers and kids journey towards the US."
