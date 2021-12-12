# Hyperpartisan Detection In News Articles

_Project for Adv NLP Course (CS7.501.M21) at IIITH_

**Team 6:-** Sentiment Detected: HMMMMMM

**Members:-**
- **Akshett Rai Jindal** *(2019114001)*
- **Suyash Vardhan Mathur** *(2019114006)*

## Link to Downloadables

| Downloadable         | Link                                                                                 |
|----------------------|--------------------------------------------------------------------------------------|
| Bilm-tf repo         | https://github.com/allenai/bilm-tf                                                   |
| Trained Model        | https://drive.google.com/drive/folders/1i5qDhrat7Gs5jMVtRywcPQBKpD15zYMO?usp=sharing |
| Subjectivity Dataset | https://www.cs.cornell.edu/people/pabo/movie-review-data/                            |

## Directory Structure

```
Sentiment Detected: HMMMMMM_6
├── analysis
│  ├── model_stats.ipynb
│  ├── sentence_subjectivity_classifier.ipynb
│  └── stats
│     ├── baseline_epochs.txt
│     ├── baseline_plus_epochs.txt
│     ├── bert_normal_epochs.txt
│     ├── bert_subjective_epochs.txt
│     └── sentence_subjectivity_epochs.txt
├── dataset
│  ├── articles
│  │  ├── article.xsd
│  │  ├── articles-training-byarticle-20181122.xml
│  │  └── ground-truth-training-byarticle-20181122.xml
│  └── subjectivity_classifier
│     ├── plot.tok.gt9.5000
│     ├── quote.tok.gt9.5000
│     └── subjdata.README
├── README.md
├── src
│  ├── bert
│  │  ├── bert_model.py
│  │  ├── bert_subjective_model.py
│  │  ├── requirements.txt
│  │  └── subjective_sentences.json
│  ├── elmo
│  │  ├── bilmtf
│  │  ├── elmo_model.py
│  │  ├── elmo_subjective_model.py
│  │  └── requirements.txt
│  └── subjectivity_classifier
│     ├── get_subjective_sentences.py
│     └── train_model.py
├── Team6_Proposal.pdf
└── Team6_Report.pdf
```

## Running

### Subjectivity Classifier

* To train the subjectivity classifier model, download the dataset from the link above
* Install the same requirements as mentioned in ELMo
* It also requires the `bilm-tf` repo as we are training on the ELMo embeddings
* Then run the command `PYTHONPATH=<path_to_bilm_folder_in_bilm_tf> python3 <filename>.py`

### ELMo Models

* The required packages are written in `requirements.txt` file.
* As we are using pretrained ELMo embeddings, so download the pretrained files from the
  Bilm-tf repo (link above)
* In the code, update the links to various files according to your directory structure
* Run the command `PYTHONPATH=<path_to_bilm_folder_in_bilm_tf> python3 <filename>.py`

### BERT Models

* The required packages are written in `requirements.txt` file. Note that these are
  a bit different from ELMo ones in regards to the versions, so use `--upgrade` flag
* For the BERT model with subjectivity classification, the `json` file provided should be
  in the same directory to gather information about the subjective sentences.
* In the code, update the links to various files according to your directory structure
* Run the command `python3 <filename>.py`

## Analysis

* The `analysis` folder in the submission contains the various losses and accuracies
  for training and validation data for all the epochs of all the models.
* It also contains the code to generate the graphs by parsing this data.
