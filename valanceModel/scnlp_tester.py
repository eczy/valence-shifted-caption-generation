# core nlp tester
import json

from stanfordcorenlp import StanfordCoreNLP

sentences = "Hi, my name is Drew. I like tacos and big dogs."


nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05', memory='8g')



properties = {'annotators': 'depparse,lemma', 'outputFormat': 'json'}


ann_dict = json.loads( nlp.annotate(sentences, properties=properties) )


import code
code.interact(local=locals())

#ann_dict['sentences'][0]['basicDependencies']
