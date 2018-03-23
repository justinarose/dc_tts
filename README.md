# Using Discriminators to Transfer Vocal Style on end-to-end TTS Systems

Code repository for cs 230 project

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3 (Note that the API of `tf.contrib.layers.layer_norm` has changed since 1.3)
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data
I trained the model on two different speech datasets. <p> 1. [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) <br/> 2. [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

LJ Speech Dataset was used as the dataset for the labeled distribution. Similarly VCTK was used as the target data distribution. Note the hyperparameter that need to be set for the target data directory, which expects a file named fnames.txt with all the files paths to be added.


## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) and [VCTK](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. (If you want to do preprocessing, set prepro True`.
  * STEP 2. Run `python train.py 1` for training Text2Mel. (If you set prepro True, run python prepro.py first)
  * STEP 3. Run `python train.py 2` for training SSRN.

You can do STEP 2 and 3 at the same time, if you have more than one gpu card.

## Audio files
Generated audio files can be found in subdirectories, /20k_original_samples/, /british-160k/, and /british-80k/.
  
