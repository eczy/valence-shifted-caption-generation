# 595-valence-shifting-captions


![alt text](https://github.com/eczy/595-valence-shifting-captions/blob/master/projectFlow.png)


## Installation

This project assumes that you are using an anaconda/miniconda environment running Python 3.6.
If you already have a Python 3.6 environment, activiate it. Otherwise, follow these steps to create and activate a new environment.
1. Install a recent version of Anaconda
2. `conda create -n test595ProjInstall python=3.6 anaconda`
3. `source activate test595ProjInstall`
4. You should now be in the new environment and should see "(test595ProjInstall)" in your command line prompt


Dependencies:
- `keras==1.2.2`
- `tensorflow==0.12.1`
- `tqdm`
- `numpy`
- `pandas`
- `matplotlib`
- `pillow`
- `jupyterlab` (or jupyter notebook if you prefer -- package dependency is then `jupyter`)
- `stanfordcorenlp` (python wrapper)
- `textblob`
- `progressbar2`


1. Clone this repo
2. Install the above dependencies using `pip install`
3. Download StanfordCoreNLP 3.9.2 zip from http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip and unpack the zip inside 595-valence-shifting-captions/
4. Download the Amazon data submitted by the team - the necessary data has already been parsed (parsing yourself will take several days of computing time). Unzip the file and move `amazonRawData/`, `amazon_counts/`, `amazon_pairTuples/`, and `amazon_sentenceTuples/` into `595-valence-shifting-captions/valanceModel/`
5. Download the Flickr8k dataset submitted by the team and move both `Flickr8k_text/` and `Flicker8k_Dataset/` into `595-valence-shifting-captions/Image-Captioning/`


## Processing an Example Image
1. Run Jupyter lab or Jupyter notebook *AS ROOT* (this is unfortunately required by StanfordCoreNLP).
The command is `jupyter lab` or `jupyter notebook` depending on your preference.
    - Note: if you are not using your base anaconda environment, you should add the
environment you will be using to the set of ipython kernels so that the installed
dependencies can be used by the notebook. If this is the case, then activate your desired environment and run
`ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"`
2. Open `Image Captioning InceptionV3-minimal.ipynb` using the desired kernel (see above).
3. Restart the kernel and run the full notebook. Shifted captions for 5 images from
the testing dataset will be generated at the bottom of the page.
