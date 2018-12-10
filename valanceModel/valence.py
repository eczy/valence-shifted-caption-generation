from stanfordcorenlp import StanfordCoreNLP
import ProbModel as NBM
from textblob import Word
import json
import math
import numpy as np
from random import randint, sample

# Expect dictionary of bigram counts for NN-JJ / ADV-VB
# Expect a model that takes in pairs as input and outputs
# valence rank.

# There should be a list of adjecrtives that are too common
# for use and should be treated as stop words.

# Additionally, we should randomize over the top n adjectrives,
# so if we see the same noun twice we do not necessarily get
# the same word. Or, otherwise, we could get a set of synonyms
# for a given noun and randomize over this set.

NOUN_TAGS = ['NN', 'NNS']
VERB_TAGS = ['VB','VBD','VBN','VBG','VBP','VBZ']

class mySentence:
	def __init__(self, sentence, nlp, numPossible = 50, numChosen = 50):

		self.model = NBM.sentimentModel()
		self.words, self.lemmas, self.tags = [], [], []
		self.readSentence(sentence, nlp)

		# numPossible represents the top n words taken from the corpus, of which
		# we will choose numChosen of them for possible replacement words.
		# From each replacement word, we will choose numSynonyms synonyms as
		# further possible replacement words.
		self.numPossible = numPossible
		self.numChosen = numChosen
		self.numSynonyms = 3

		self.nouns = [self.lemmas[i] for i in range(len(self.lemmas)) if self.tags[i] in NOUN_TAGS]
		self.verbs = [self.lemmas[i] for i in range(len(self.lemmas)) if self.tags[i] in VERB_TAGS]

		self.adjectives = self.getAdjectives()
		self.adverbs = self.getAdverbs()

	def readSentence(self, sentence, nlp):
		output = json.loads(nlp.annotate(sentence, properties = {
			"annotators": "tokenize,ssplit,parse,sentiment,lemma",
			"outputFormat": "json",
			"ssplt.eolonly": "true",
			"enforceRequirements": "false"
		}))

		for a in output['sentences']:
			for d in a['tokens']:
				self.lemmas.append(d['lemma'])
				self.words.append(d['word'])
				self.tags.append(d['pos'])
		return

	def getAdjectives(self):
		adj_dict = {n:{} for n in self.nouns}
		for word in self.nouns:
			possible = self.model.nounAdjCount_map[word]
			adj_dict[word] = dict.fromkeys(self.possibleReplacements(word, possible),0)
		adj_dict = self.valenceRank(adj_dict)
		return adj_dict

	def getAdverbs(self):
		adv_dict = {v:{} for v in self.verbs}
		for word in self.verbs:
			possible = self.model.verbAdvCount_map[word]
			adv_dict[word] = dict.fromkeys(self.possibleReplacements(word, possible),0)
		adv_dict = self.valenceRank(adv_dict)
		return adv_dict

	def possibleReplacements(self, word, possible):
		keywords = ['pos', 'neg']
		final = []
		possible = self.PMI(word, possible)
		for key in keywords:
			words = [(k,possible[k]) for k in possible if self.model.predictedClass(k, word) == key]
			words_sorted = sorted([k for k in words], key=lambda x:x[1], reverse=True)
			chosen = sample(range(0, min(self.numPossible, len(words_sorted))), min(self.numChosen, len(words_sorted)))
			final.append(list(set(words_sorted[i][0] for i in chosen)))
		final = set(final[0] + final[1])
		# for a in chosen:
		# 	final.update(set(synonyms(possible_sorted[a][1], self.numSynonyms)))
		return final

	def PMI(self, word, possible):
		PMI_dict = {}
		totalNumBigrams = self.model.pairCount_int
		totalNumUnigrams = self.model.uniCount_int
		for modifier in possible.keys():

			countAdj = self.model.unigramCount_map[modifier]
			countWord = self.model.unigramCount_map[word]
			bigramCount = possible[modifier]

			prob_bigram = bigramCount / totalNumBigrams
			probAdj = countAdj / totalNumUnigrams
			probWord = countWord / totalNumUnigrams

			PMI_check = (prob_bigram != 0) and (probAdj != 0) and (probWord != 0)

			PMI = math.log(prob_bigram / (probAdj * probWord), 2) if PMI_check else 0
			if countAdj > 50:
				PMI_dict[modifier] = PMI
		return PMI_dict
		# return possible

	def valenceRank(self, input_dict):
		for noun in input_dict.keys():
			for adj in input_dict[noun]:
				input_dict[noun][adj] = self.model.predictedClass(adj, noun)
		return input_dict


def synonyms(word, maxSyns):
	syns, ants = [], []
	for syn in Word(word).synsets:
		for l in syn.lemmas():
			syns.append(l.name())
			if l.antonyms():
				ants.append(l.antonyms()[0].name())
	final = [syns[i] for i in sample(range(0, len(syns)), min(maxSyns, len(syns)))]
	for ant in ants:
		final.append(ant)
	return final


# if __name__ == '__main__':
# 	nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')
# 	s = mySentence("two men sat in a room and ate the food", nlp)
# 	print(s.adjectives)
