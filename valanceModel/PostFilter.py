import csv
import os
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
	_emotion = False

	def __init__(self, hostile=False, strong=False, power=False, pain=False, emotion=False):
		self._hostile = hostile
		self._strong = strong
		self._power = power
		self._pain = pain
		self._emotion = emotion
		if not os.path.exists(os.getcwd() + '/genaralInquirerLexicon.pkl'):
			self.parseGeneralInquirerLexicon('inquirerbasic.csv')


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

				word = row[0]
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

	def filter(sentence):
		s = mySentence(sentence)
		adjectives = s.adjectives
		adverbs = s.adjverbs

		commonAdjectives = set([a for a in adjectives if a in self._positiveWords])
		commonAdverbs = set([a for a in adjectives if a in self._positiveWords])




if __name__ == '__main__':
	filteredAdjectivesAndAdverbs = PostFilter()
	filteredAdjectivesAndAdverbs.filter(word)	
