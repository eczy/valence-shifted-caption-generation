# 595-valence-shifting-captions

Stanford CoreNLP required to make this work
Download the zip from https://stanfordnlp.github.io/CoreNLP/ and unzip it inside the head directory of your repo clone



![alt text](https://github.com/eczy/595-valence-shifting-captions/blob/master/projectFlow.png)

## Running a Minimal Example
This procedure assumes that you are using an anaconda/miniconda environment running
Python 3.

Dependencies:
- Keras 1.2.2
- Tensorflow 0.12.1
- tqdm
- numpy
- pandas
- matplotlib
- pillow
- jupyterlab (or jupyter notebook if you prefer -- package dependency is then `jupyter`)

1. Clone this repo
2. Install the above dependencies
3. cd into the `Image-Captioning` submodule
4. Download the Flickr8k dataset and paste both `Flickr8k_text/` and `Flicker8k_Dataset/` in the current directory.
5. Run Jupyter lab or Jupyter notebook *AS ROOT* (this is unfortunately required by StanfordCoreNLP).
The command is `jupyter lab` or `jupyter notebook` depending on your preference.
    - Note: if you are not using your base anaconda environment, you should add the
environment you will be using to the set of ipython kernels so that the installed
dependencies can be used by the notebook. If this is the case, then activate your desired environment and run
`ipykernel install --user --name <env_name> --display-name "Python (<env_name>)"`
6. Open `Image Captioning InceptionV3-minimal.ipynb` using the desired kernel (see above).
7. Restart the kernel and run the full notebook. Shifted captions for 5 images from
the testing dataset will be generated at the bottom of the page.
