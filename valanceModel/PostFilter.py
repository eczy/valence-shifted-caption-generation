import csv
import os
import numpy as np
from valence import mySentence
import pickle

class PostFilter:
	_words = {}
	_hostile = False
	_strong = False
	_power = False
	_pain = False
	_feel = False
	_emotion = False
	_filter = False
	_filters = ['hostile', 'strong', 'power', 'pain', 'feel', 'emotion']

	# Creates a post filter object with certain sets of filters
	def __init__(self, hostile=False, strong=False, power=False, pain=False, feel=False, emotion=False):
		self._hostile = hostile
		self._strong = strong
		self._power = power
		self._pain = pain
		self._feel = feel
		self._emotion = emotion
		# Check to see if any filtering needs to be done
		if self._hostile or self._strong or self._power or self._pain or self._feel or self._emotion:
			self._filter = True
		if not os.path.exists(os.getcwd() + '/generalInquirerLexicon.pkl'):
			self.parseGeneralInquirerLexicon('inquirerbasic.csv')
		else:
			self.loadGeneralInquirerLexicon(os.getcwd() + '/generalInquirerLexicon.pkl')

	def loadGeneralInquirerLexicon(self, filename):
		generalInquirerLexicon = pickle.load(open(filename, 'rb'))
		self._words = generalInquirerLexicon['words']
		print("Finished Loading General Inquirer Lexicon")

	# Internal function that parses the GIL csv and tags words with certain categories
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

	def subClassWords(self, candidateDict):
		valenceDict = {}
		valenceDict['vNeg'] = []
		valenceDict['pNeg'] = []
		valenceDict['neut'] = []
		valenceDict['pPos'] = []
		valenceDict['vPos'] = []

		for item in candidateDict:
			word = item[0]
			valence = float(item[1])
			print((word, valence))
			if valence <= -0.5: valenceDict['vNeg'].append(word)
			elif valence < 0.0: valenceDict['pNeg'].append(word)
			elif valence == 0.0: valenceDict['neut'].append(word)
			elif valence < 0.5: valenceDict['pPos'].append(word)
			else: valenceDict['vPos'].append(word)

		return valenceDict

	# filter adjectives and adverbs based on whether you need pos/neg valence, and whether you want union/intersection of filters
	def filter(self, sentence, union=True, intersection=False):
		s = mySentence(sentence)
		candidateAdjectives = {}
		candidateAdverbs = {}
		for noun in s.adjectives:
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

	def inUnion(self, word):
		for f in self._filters:
			if self._words[word.upper()][f]:
				return True
		return False


	def inIntersection(self, word, filters, pos=True, neg=True):
		for f in self._filters:
			if not self._words[word.upper()][f]:
				return False
		return True


if __name__ == '__main__':
	filteredAdjectivesAndAdverbs = PostFilter()
	adj, adv = filteredAdjectivesAndAdverbs.filter('the person walks')
	print(adj, adv)
