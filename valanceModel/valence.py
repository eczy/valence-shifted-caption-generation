from stanfordcorenlp import StanfordCoreNLP
import NaiveBayesModel as NBM
import numpy as np
from textblob import TextBlob
from textblob import Word
import operator
import pdb
from random import randint

# Expect dictionary of bigram counts for NN-JJ / ADV-VB
# Expect a model that takes in pairs as input and outputs
# valence rank.

# There should bea list of adjecrtives that are too common
# for use and should be treated as stop words.

# Additionally, we should randomize over the top n adjectrives,
# so if we see the same noun twice we do not necessarily get
# the same word. Or, otherwise, we could get a set of synonyms
# for a given noun and randomize over this set.

NOUN_TAGS = ['NN', 'NNS']
VERB_TAGS = ['VB','VBD','VBN','VBG','VBP','VBZ']

class mySentence:
	def __init__(self, sentence):
		# self.nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g'))
		# self.text = nlp.annotate(sentence)
		self.text = TextBlob(sentence)
		self.numPossible = 20
		self.words = self.text.words
		self.nouns = [Word(t[0]).lemmatize() for t in self.text.tags if t[1] in NOUN_TAGS]
		self.verbs = [Word(t[0]).lemmatize() for t in self.text.tags if t[1] in VERB_TAGS]
		self.model = NBM.sentimentModel()
		self.adjectives = self.getAdjectives()
		self.adverbs = self.getAdverbs()
		self.output = ""

	def getPolarity(self):
		return self.text.polarity

	def getAdjectives(self):
		adj = {n:{} for n in self.nouns}
		for word in self.nouns:
			possible = self.model.nounAdjCount_map[word]
			possible_sorted = sorted([(possible[k],k) for k in possible], key=lambda x:x[0], reverse=True)
			chosen_adj = possible_sorted[randint(0, min(self.numPossible, len(possible_sorted)))]
			# syns, ants = synonyms(chosen_adj[1])
		return adj

	def getAdverbs(self):
		adv = {v:{} for v in self.verbs}
		for word in self.verbs:
			possible = self.model.verbAdvCount_map[word]
			possible_sorted = sorted([(possible[k],k) for k in possible], key=lambda x:x[0], reverse=True)
			adv[word] = possible_sorted[randint(0, min(self.numPossible, len(possible_sorted)))]
		return adv

	def insertWords(self):
		for word in self.words:
			if word in self.nouns:
				adj = self.adjectives[word][randint(0, len(self.adjectives[word]))][1]
				self.output += adj + " "
			# elif word in self.verbs:
			# 	adv = self.adverbs[word][randint(0, self.numPossible)][1]
			# 	self.output += adv + " "
			self.output += word + " "
		return

def synonyms(word):
	syns, ants = [], []
	for syn in Word(word).synsets:
		for l in syn.lemmas():
			syns.append(l.name())
			if l.antonyms():
				ants.append(l.antonyms()[0].name())
	return syns, ants


if __name__ == '__main__':
	s = mySentence("the man watch the movie")
	print(s.text)
	print(s.adjectives)
















