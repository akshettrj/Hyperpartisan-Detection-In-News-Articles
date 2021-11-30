import xml.etree.ElementTree as ET
import random
from math import floor
import tensorflow as tf
import numpy as np
from scipy import spatial
from elmo.bilm import Batcher, BidirectionalLanguageModel, weight_layers
from nltk.tokenize import sent_tokenize, word_tokenize

articles_xmltree_file = './data/articles-training-byarticle-20181122.xml'
articles_tree = ET.parse(articles_xmltree_file)
articles_root = articles_tree.getroot()

truth_xmltree_file = './data/ground-truth-training-byarticle-20181122.xml'
truth_tree = ET.parse(truth_xmltree_file)
truth_root = truth_tree.getroot()

dataset_articles_text = []
dataset_articles_truth_value = []

for articles_child, truth_child in zip(articles_root, truth_root):
    articles_attributes = articles_child.attrib
    truth_attributes = truth_child.attrib

    article_text = [p.text for p in articles_child if p.text is not None]
    article_text = " ".join(article_text)

    dataset_articles_text.append(article_text)
    dataset_articles_truth_value.append(bool(truth_attributes["hyperpartisan"]))

num_articles = len(dataset_articles_text)

# Store the corresponding indices
training_articles = random.sample(range(num_articles), floor(num_articles*0.8))
testing_articles = [_ for _ in range(num_articles) if _ not in training_articles]


# For ELMo
vocabulary_file_path = "./elmo/vocabulary.txt"
options_file_path = "./elmo/options.json"
weights_file_path = "./elmo/weights.hdf5"

batcher = Batcher(vocabulary_file_path, 50)
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
bilm = BidirectionalLanguageModel(options_file_path, vocabulary_file_path)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)


for i in training_articles:

    article_text = dataset_articles_text[i]
    article_sentences = sent_tokenize(article_text)
    article_words = [word_tokenize(sentence) for sentence in article_sentences]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        context_ids = batcher.batch_sentences(article_words)

        # emb = article_elmo_embeddings[sent_no, word_no]
        article_elmo_embeddings = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_character_ids: context_ids}
        )

        article_sentences_embeddings = np.array([
            np.average(article_elmo_embeddings[sent_no,:])
            for sent_no in range(num_articles)
        ])

        print(article_sentences_embeddings.shape)
    break

    pass
