from valence import mySentence
from PostFilter import PostFilter as PF
import progressbar
from stanfordcorenlp import StanfordCoreNLP
import pickle
import os


def main():

	captionListFile = os.getcwd() + '/../captionList.pkl'
	captionList = pickle.load(open(captionListFile, 'rb'))
	outFile = os.getcwd() + '/../generatedCaptionsNoAdverbs.txt'
	postFilter = PF()
	numCaptions = len(captionList)
	nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')
	count = 0
	with open(outFile, 'w') as f:
		print("Starting Caption Generation with Sentiment")
		with progressbar.ProgressBar(max_limit=numCaptions) as bar:
			for caption in captionList:
				try:
					adj, adv = postFilter.filter(caption, nlp)
				except KeyError:
					continue
				
				bar.update(count)
				count += 1

				outputCategories = set()
				for k, v in adj.items():
					outputCategories.update([k1 for k1 in v.keys()])

				output = generateOutput(caption, adj, adv, list(outputCategories), nlp)
				f.write(caption + '\n')
				for category, sentence in zip(outputCategories, output):
					f.write('{}: {}\n'.format(category, sentence))
	nlp.close()

def generateOutput(caption, adj, adv, outputCategories, nlp):
	allOutputs = []
	myCaption = mySentence(caption, nlp)
	for outputType in outputCategories:
		output = ""
		space = " "
		for word, lemma in zip(myCaption.words, myCaption.lemmas):
			if lemma in myCaption.nouns:
				output = output + adj[lemma][outputType] + space
			# elif lemma in myCaption.verbs:
			# 	output = output + adv[lemma][outputType] + space
			output = output + word + space
		allOutputs.append(output)
	return allOutputs

if __name__ == "__main__":
	main()
