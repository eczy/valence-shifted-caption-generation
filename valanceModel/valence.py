from stanfordcorenlp import StanfordCoreNLP
import NaiveBayesModel as NBM
from textblob import Word
import json
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
	def __init__(self, sentence):

		self.model = NBM.sentimentModel()
		self.words, self.lemmas, self.tags = [], [], []
		self.readSentence(sentence)

		# numPossible represents the top n words taken from the corpus, of which
		# we will choose numChosen of them for possible replacement words.
		# From each replacement word, we will choose numSynonyms synonyms as 
		# further possible replacement words.
		self.numPossible = 20
		self.numChosen = 20
		self.numSynonyms = 3
		
		self.nouns = [self.lemmas[i] for i in range(len(self.lemmas)) if self.tags[i] in NOUN_TAGS]
		self.verbs = [self.lemmas[i] for i in range(len(self.lemmas)) if self.tags[i] in VERB_TAGS]

		self.adjectives = self.getAdjectives()
		self.adverbs = self.getAdverbs()

	def readSentence(self, sentence):
		nlp = StanfordCoreNLP('http://localhost', port=9000, timeout=30000)
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
		nlp.close()
		return

	def getAdjectives(self):
		adj_dict = {n:{} for n in self.nouns}
		for word in self.nouns:
			possible = self.model.nounAdjCount_map[word]
			adj_dict[word] = dict.fromkeys(self.possibleReplacements(possible),0)
		adj_dict = self.valenceRank(adj_dict)
		return adj_dict

	def getAdverbs(self):
		adv_dict = {v:{} for v in self.verbs}
		for word in self.verbs:
			possible = self.model.verbAdvCount_map[word]
			adv_dict[word] = dict.fromkeys(self.possibleReplacements(possible),0)
		adv_dict = self.valenceRank(adv_dict)
		return adv_dict

	def possibleReplacements(self, possible):
		possible_sorted = sorted([(possible[k],k) for k in possible], key=lambda x:x[0], reverse=True)
		chosen = sample(range(0, min(self.numPossible, len(possible_sorted))), self.numChosen)
		final = set(possible_sorted[i][1] for i in chosen)
		for a in chosen:
			final.update(set(synonyms(possible_sorted[a][1], self.numSynonyms)))
		return final

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
















