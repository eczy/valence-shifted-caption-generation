import csv
import os
import numpy as np
from valence import mySentence

class PostFilter:
	_totalWords = 0
	_positiveCount = 0
	_negativeCount = 0
	_positiveWords = {}
	_negativeWords = {}
	_hostile = False
	_strong = False
	_power = False
	_pain = False
	_feel = False
	_emotion = False

	# Creates a post filter object with certain sets of filters
	def __init__(self, hostile=False, strong=False, power=False, pain=False, feel=False, emotion=False):
		self._hostile = hostile
		self._strong = strong
		self._power = power
		self._pain = pain
		self._feel = feel
		self._emotion = emotion
		if not os.path.exists(os.getcwd() + '/genaralInquirerLexicon.pkl'):
			self.parseGeneralInquirerLexicon('inquirerbasic.csv')

	# Internal function that parses the GIL csv and tags words with certain categories
	def parseGeneralInquirerLexicon(self, filename):
		with open(filename, 'r') as csvfile:
			csvReader = csv.reader(csvfile, delimiter=',')

			rowCount = 0
			negCount = 0
			posCount = 0

			for row in csvReader:
				rowCount += 1
				if rowCount == 1:
					rowCount += 1
					continue

				word = row[0].lower()
				subInformation = {'hostile': True if row[4] else False,
									'strong': True if row[5] else False,
									'power': True if row[6] else False,
									'pain': True if row[9] else False,
									'feel': True if row[10] else False,
									'emotion': True if row[12] else False
								}

				if row[1] == 'Positiv':
					self._positiveWords[word] = {}
					for key in subInformation:
						self._positiveWords[word][key] = subInformation[key]
					posCount += 1
				else:
					self._negativeWords[word] = {}
					for key in subInformation:
						self._negativeWords[word][key] = subInformation[key]
					negCount += 1

	# filter adjectives and adverbs based on whether you need pos/neg valence, and whether you want union/intersection of filters
	def filter(self, sentence, pos=False, neg=False, union=False, intersection=False):
		s = mySentence(sentence)
		candidateAdjectives = {}
		candidateAdverbs = {}
		for noun in s.adjectives:
			if pos:
				commonAdjectives = list(set([a for a in s.adjectives[noun] if a.lower() in self._positiveWords and s.adjectives[noun][a] > 0.0]))
			if neg:
				commonAdjectives = list(set([a for a in s.adjectives[noun] if a.lower() in self._negativeWords and s.adjectives[noun][a] < 0.0]))

			if not self._hostile and not self._strong and not self._power and not self._pain and not self._feel and not self._emotion:
				randAdj = np.random.randint(0,len(commonAdjectives))
				candidateAdjectives[noun] = commonAdjectives[randAdj]
			else:
				filters = []
				if self._hostile: filters.append('hostile')
				if self._strong: filters.append('strong')
				if self._power: filters.append('power')
				if self._pain: filters.append('pain')
				if self._feel: filters.append('feel')
				if self._emotion: filters.append('emotion')

				filteredAdjectives = []

				for adjective in s.adjectives[noun]:
					if union:
						if self.inUnion(adjective, filters, pos=pos, neg=neg):
							filteredAdjectives.append(adjective)
					elif intersection:
						if self.inIntersection(adjective, filters, pos=pos, neg=neg):
							filteredAdjectives.append(adjective)

				filteredAdjectives = list(set(filteredAdjectives))

				randAdj = np.random.randint(0,len(commonAdjectives))
				candidateAdjectives[noun] = commonAdjectives[randAdj]

		for verb in s.adverbs:
			print(s.adverbs[verb])
			if pos:
				commonAdverbs = list(set([a for a in s.adverbs[verb] if a.lower() in self._positiveWords and s.adverbs[verb][a] > 0.0]))
			if neg:
				commonAdverbs = list(set([a for a in s.adverbs[verb] if a.lower() in self._negativeWords and s.adverbs[verb][a] < 0.0]))


			if not self._hostile and not self._strong and not self._power and not self._pain and not self._feel and not self._emotion:
				randAdv = np.random.randint(0,len(commonAdverbs))
				candidateAdverbs[verb] = commonAdverbs[randAdv]
			else:
				filters = []
				if self._hostile: filters.append('hostile')
				if self._strong: filters.append('strong')
				if self._power: filters.append('power')
				if self._pain: filters.append('pain')
				if self._feel: filters.append('feel')
				if self._emotion: filters.append('emotion')

				filteredAdverbs = []

				for adverb in s.adverbs[verb]:
					if union:
						if self.inUnion(adverb, filters, pos=pos, neg=neg):
							filteredAdverbs.append(adverb)
					elif intersection:
						if self.inIntersection(adverb, filters, pos=pos, neg=neg):
							filteredAdverbs.append(adverb)

				filteredAdverbs = list(set(filteredAdverbs))

				randAdv = np.random.randint(0,len(commonAdverbs))
				candidateAdverbs[verb] = commonAdverbs[randAdv]

		return candidateAdjectives, candidateAdverbs

	def inUnion(self, word, filters, pos=True, neg=True):
		if pos:
			for f in filters:
				if self._positiveWords[word.upper()][f]:
					return True
			return False
		if neg:
			for f in filters:
				if self._negativeWords[word.upper()][f]:
					return True
			return False

	def inIntersection(self, word, filters, pos=True, neg=True):
		if pos:
			for f in filters:
				if not self._positiveWords[word.upper()][f]:
					return False
			return True
		if neg:
			for f in filters:
				if not self._negativeWords[word.upper()][f]:
					return False
			return True





if __name__ == '__main__':
	filteredAdjectivesAndAdverbs = PostFilter()
	adj, adv = filteredAdjectivesAndAdverbs.filter('the person walks', pos=True, union=True)
	print(adj, adv)
