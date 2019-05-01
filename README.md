# CAML + HAN models for MIMIC data

This code is for CPSC 490: Senior Project, as part of the graduation requirements for the Computer Science major.

This includes code for the paper [Explainable Prediction of Medical Codes from Clinical Text](https://arxiv.org/abs/1802.05695) as well as for [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174)

CAML code is from from Mullenbach et al., which can be found at [https://github.com/jamesmullenbach/caml-mimic](https://github.com/jamesmullenbach/caml-mimic) 


## Dependencies
* Python 3.6
* pytorch 0.3.0
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.20.3
* jupyter-notebook 5.0.0
* gensim 3.2.0
* nltk 3.2.4

## Running the model
Both commands should be run from inside the learn2 folder.
To train the Hierarchical Attention model from scratch, run the following command:
```
python training.py ../mimicdata/mimic3/train_50.csv ../mimicdata/mimic3/vocab.csv 50 han 100 --dropout 0.2 --patience 10 --lr 0.0001 --batch-size 16 --gpu
```
Otherwise, to replicate the results of the data in the final report, run the following:
```
python training.py ../mimicdata/mimic3/train_50.csv ../mimicdata/mimic3/vocab.csv 50 han 50 --dropout 0.2 --patience 10 --lr 0.0001 --batch-size 16 --gpu --test-model model_best_f1_micro.pth
```


