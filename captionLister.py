#   This file just takes a text file of line-seperated captions and generates
# a list of the captions and saves them as a pickle

import sys
import pickle

captionList = []

for f in sys.argv[1:]:
    with open(f, 'r') as captionFile:
        lines = captionFile.readlines()
        for l in lines:
            captionList.append(l.strip())

with open('captionList.pkl', 'wb') as captionOutFile:
    pickle.dump(captionList, captionOutFile)


# # to load up the caption list, copy this code:
# with open('captionList.pkl', 'rb') as captionListInFile:
#     captionList = pickle.load(captionListInFile)


# import code
# code.interact(local=locals())
