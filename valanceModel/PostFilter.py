# This file defines the PostFilter object, which is used to select a single 
# positive and negative adjective for every noun in the generate caption.
# The object allows the user to specify a few different types of filters
# that can be used in the processing of selecting one adjective from many.

import csv
import os
import numpy as np
from valence import mySentence
from stanfordcorenlp import StanfordCoreNLP
import pickle

# Defines the PostFilter Object
class PostFilter:
	_words = {}
	_positiveWords = {}
	_negativeWords = {}
	_hostile = False
	_strong = False
	_power = False
	_pain = False
	_feel = False
	_emotion = False
	_filter = False
	_filters = ['hostile', 'strong', 'power', 'pain', 'feel', 'emotion']
	_opinion = False
	_GIL = False

	# Creates a PostFilter Object
	# dataset = <opinion> | <GIL>
	# 	- Determines whether to read from the opinion lexicon or GIL lexicon
	# Next set of flags only used if dataset=<GIL>. These are tags used in the GIL
	# 	- hostile = <True/False>
	#	- strong = <True/False>
	#	- power = <True/False>
	#	- pain = <True/False>
	# 	- feel = <True/False>
	# 	- emotion = <True/False> 
	def __init__(self, dataset='opinion', hostile=False, strong=False, power=False, pain=False, feel=False, emotion=False):
		self._hostile = hostile
		self._strong = strong
		self._power = power
		self._pain = pain
		self._feel = feel
		self._emotion = emotion
		# Check to see if any filtering needs to be done
		if dataset == 'opinion':
			self._opinion = True
			if not os.path.exists('opinionWords.pkl'):
				self.parseOpinionWords('positive-words.txt', pos=True)
				self.parseOpinionWords('negative-words.txt', neg=True)
			else:
				self.loadOpinionWords('opinionWords.pkl')
		elif dataset == 'GIL':
			self._GIL = True
			if self._hostile or self._strong or self._power or self._pain or self._feel or self._emotion:
				self._filter = True
			if not os.path.exists(os.getcwd() + '/generalInquirerLexicon.pkl'):
				self.parseGeneralInquirerLexicon('inquirerbasic.csv')
			else:
				self.loadGeneralInquirerLexicon(os.getcwd() + '/generalInquirerLexicon.pkl')
		else:
			print("dataset must be one of <opinion> or <GIL>")
			exit()

	# parses pos/neg adjectives from opinion lexicon and creates pickle
	def parseOpinionWords(self, filename, pos=False, neg=False):
		with open(filename, 'r', errors='ignore') as f:
			for line in f:
				if pos:
					self._words[line.rstrip().lower()] = 'pos'
					self._positiveWords[line.rstrip().lower()] = pos
				elif neg:
					self._words[line.rstrip().lower()] = 'neg'
					self._negativeWords[line.rstrip().lower()] = neg
		opinionWords = {}
		opinionWords['words'] = self._words
		opinionWords['pos'] = self._positiveWords
		opinionWords['neg'] = self._negativeWords
		with open('opinionWords.pkl', 'wb') as f:
			pickle.dump(opinionWords, f)
		print("Finished reading opinion words list")
		return

	# loads opinion lexicon from pickle file
	def loadOpinionWords(self, filename):
		with open(filename, 'rb') as f:
			opinionWords = pickle.load(f)
			self._words = opinionWords['words']
			self._positiveWords = opinionWords['pos']
			self._negativeWords = opinionWords['neg']
		print("Finished loading opinion words")

	# loads GIL lexicon from pickle
	def loadGeneralInquirerLexicon(self, filename):
		generalInquirerLexicon = pickle.load(open(filename, 'rb'))
		self._words = generalInquirerLexicon['words']
		print("Finished Loading General Inquirer Lexicon")

	# parses GIL lexicon and creates pos/neg words and tags, and saves as pickle
	def parseGeneralInquirerLexicon(self, filename):
		with open(filename, 'r') as csvfile:
			csvReader = csv.reader(csvfile, delimiter=',')
			for row in csvReader:
				word = row[0].lower()
				subInformation = {'hostile': True if row[4] else False,
									'strong': True if row[5] else False,
									'power': True if row[6] else False,
									'pain': True if row[9] else False,
									'feel': True if row[10] else False,
									'emotion': True if row[12] else False
								}

				self._words[word] = {}
				for key in subInformation:
					self._words[word][key] = subInformation[key]


			generalInquirerLexicon = {}
			generalInquirerLexicon['words'] = self._words
			pickle.dump(generalInquirerLexicon, open('generalInquirerLexicon.pkl', 'wb'))
			print("Finished Parsing General Inquirer Lexicon")

	# Selects one positive and one negative word per noun/verb
	def subClassWords(self, candidateDict):
		valenceDict = {}
		valenceDict['pos'] = []
		valenceDict['neut'] = []
		valenceDict['neg'] = []
		for item in candidateDict:
			word = item[0]
			valenceClass = item[1]
			valenceDict[valenceClass].append(word)
		for key in valenceDict:
			if len(valenceDict[key]) > 0:
				randNum = np.random.randint(0, len(valenceDict[key]))
				valenceDict[key] = valenceDict[key][randNum]	
			else:
				valenceDict[key] = ''
		return valenceDict

	# filter adjectives and adverbs based on whether you need pos/neg valence, and whether you want union/intersection of filters
	def filter(self, sentence, nlp, union=True, intersection=False):
		s = mySentence(sentence, nlp)
		NBM = s.model
		candidateAdjectives = {}
		candidateAdverbs = {}
		for noun in s.adjectives:
			if self._opinion:
				commonAdjectives = list(set([(a.lower(),s.adjectives[noun][a]) for a in s.adjectives[noun] if self.isFine(a.lower(), s.adjectives[noun][a], noun, NBM)]))
				candidateAdjectives[noun] = self.subClassWords(commonAdjectives)
			elif self._GIL:
				commonAdjectives = list(set([(a,s.adjectives[noun][a]) for a in s.adjectives[noun] if a.lower() in self._words]))
				# No Filters set, break up candidates by valence class
				if not self._filter:
					candidateAdjectives[noun] = self.subClassWords(commonAdjectives)
				else:
					filteredAdjectives = {}
					for pair in commonAdjectives:
						if union:
							if self.inUnion(pair[0]):
								filteredAdjectives[pair[0]] = pair[1]
						elif intersection:
							if self.inIntersection(pair[0]):
								filteredAdjectives[pair[0]] = pair[1]

				candidateAdjectives[noun] = self.subClassWords(filteredAdjectives)

		for verb in s.adverbs:
			if self._opinion:
				commonAdverbs = list(set([(a.lower(),s.adverbs[verb][a]) for a in s.adverbs[verb] if self.isFine(a.lower(), s.adverbs[verb][a], verb, NBM)]))
				candidateAdverbs[verb] = self.subClassWords(commonAdverbs)
			elif self._GIL:
				commonAdverbs = list(set([(a,s.adverbs[verb][a]) for a in s.adverbs[verb] if a.lower() in self._words]))
				# No Filters set, break up candidates by valence class
				if not self._filter:
					candidateAdverbs[verb] = self.subClassWords(commonAdverbs)
				else:
					filteredAdverbs = {}
					for pair in commonAdverbs:
						if union:
							if self.inUnion(pair[0]):
								filteredAdverbs[pair[0]] = pair[1]
						elif intersection:
							if self.inIntersection(pair[0]):
								filteredAdverbs[pair[0]] = pair[1]

					candidateAdverbs[verb] = self.subClassWords(filteredAdverbs)
		return candidateAdjectives, candidateAdverbs

	# Used for GIL lexicon with filters set
	def inUnion(self, word):
		for f in self._filters:
			if self._words[word.upper()][f]:
				return True
		return False

	# Used for GIL lexicon with filters set
	def inIntersection(self, word, filters, pos=True, neg=True):
		for f in self._filters:
			if not self._words[word.upper()][f]:
				return False
		return True

	# Used for opinion lexicon
	def isFine(self, s, tag, word, NBM):
		if NBM.predConfidence(tag, s, word) < 0.5:
			return False
		# if tag == 'pos':
		# 	return s in self._positiveWords
		# if tag == 'neg':
		# 	return s in self._negativeWords
		return True


if __name__ == '__main__':
	nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')
	filteredAdjectivesAndAdverbs = PostFilter()
	adj, adv = filteredAdjectivesAndAdverbs.filter('A person wearing a yellow shirt is standing in water .', nlp)
	nlp.close()
	print(adj, adv)
