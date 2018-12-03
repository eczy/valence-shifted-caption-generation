from valence import mySentence
from PostFilter import PostFilter as PF


def main():

	caption = "I ran to the park to play with my friend"

	filteredAdjectivesAndAdverbs = PF()

	adj, adv = filteredAdjectivesAndAdverbs.filter(caption)

	outputCategories = set()
	for k, v in adj.items():
		outputCategories.update([k1 for k1 in v.keys()])

	output = generateOutput(caption, adj, adv, list(outputCategories))
	print(output)
	print(outputCategories)

def generateOutput(caption, adj, adv, outputCategories):
	allOutputs = []
	myCaption = mySentence(caption)
	for outputType in outputCategories:
		output = ""
		space = " "
		for word, lemma in zip(myCaption.words, myCaption.lemmas):
			if lemma in myCaption.nouns:
				print(adj[lemma])
				output = output + adj[lemma][outputType] + space
			# elif lemma in myCaption.verbs:
			# 	output = output + adv[lemma][outputType] + space
			output = output + word + space
		allOutputs.append(output)
	return allOutputs

if __name__ == "__main__":
	main()
