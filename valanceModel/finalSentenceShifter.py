from valence import mySentence
from PostFilter import PostFilter as PF
import progressbar
from stanfordcorenlp import StanfordCoreNLP
import pickle
import os


def main():

	captionListFile = os.getcwd() + 'Evan Image Files'
	captionList = pickle.load(open(captionListFile, 'rb'))
	outFile = os.getcwd() + '/../generatedCaptionsNoAdverbsNewModel.txt'
	postFilter = PF()
	numCaptions = len(captionList.keys())
	newCaptionDict = {k:{} for k in captionList}
	generatedCaptionsAndClasses = {}
	generatedCaptionsAndClasses['pos'] = []
	generatedCaptionsAndClasses['neg'] = []
	nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')
	count = 0
	with open(outFile, 'w') as f:
		print("Starting Caption Generation with Sentiment")
		with progressbar.ProgressBar(max_limit=numCaptions) as bar:
			for image in captionList:
				caption = captionList[image]
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
					if category == 'pos':
						generatedCaptionsAndClasses['pos'].append(sentence)
					elif category == 'neg':
						generatedCaptionsAndClasses['neg'].append(sentence)
					f.write('{}: {}\n'.format(category, sentence))
					newCaptionDict[image][category] = sentence
	with open('test_caption_generated.pkl', 'wb') as f:
		pickle.dump(newCaptionDict, f)
	with open('caption_classes_list.pkl', 'wb') as f:
		pickle.dump(newCaptionDict, f)
	nlp.close()

# This sentence was used for demos and for debugging. It 
# just performs one iteration of the main function without
# reading in external files.
def individualSentenceGeneration(caption):
	nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')
	postFilter = PF()
	adj, adv = postFilter.filter(caption, nlp)
	try:
		adj, adv = postFilter.filter(caption, nlp)
	except KeyError as e:
		print("Unable to generate caption: Message: {}".format(e))
		return
	print(adj)
	print(adv)
	outputCategories = set()
	for k, v in adj.items():
		outputCategories.update([k1 for k1 in v.keys()])

	outputCategories = list(outputCategories)
	output = generateOutput(caption, adj, adv, list(outputCategories), nlp)
	for category, sentence in zip(outputCategories, output):
		print('{}: {}\n'.format(category, sentence))
	nlp.close()
	return

# This function takes in a caption and a set of adjectives
# and adverbs that pair with the nouns and verbs in the caption,
# and has this for all outputCategories. Using these sets,
# the adjectives and adverbs are inserted in their proper location.
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
	# individualSentenceGeneration("A man with a spatula is laughing")
