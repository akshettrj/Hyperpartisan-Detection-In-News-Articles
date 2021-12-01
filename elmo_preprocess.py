import xml.etree.ElementTree as ET
import random
from math import floor
import tensorflow as tf
import numpy as np
from scipy import spatial
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from nltk.tokenize import sent_tokenize, word_tokenize

from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Input, Flatten, Dense, Activation,Average
from keras.layers import Concatenate,Dropout,Conv1D,MaxPooling1D,BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

def conv1d_BN(max_len, embed_size):
    '''
    CNN with Batch Normalisation.
    :param max_len: maximum sentence numbers, default=200
    :param embed_size: ELMo embeddings dimension, default=1024
    :return: CNN with BN model
    '''
    filter_sizes = [2, 3, 4, 5, 6]
    num_filters = 128
    inputs = Input(shape=(max_len,embed_size), dtype='float32')
    conv_0 = Conv1D(num_filters, kernel_size=(filter_sizes[0]))(inputs)
    act_0 = Activation('relu')(conv_0)
    bn_0 = BatchNormalization(momentum=0.7)(act_0)

    conv_1 = Conv1D(num_filters, kernel_size=(filter_sizes[1]))(inputs)
    act_1 = Activation('relu')(conv_1)
    bn_1 = BatchNormalization(momentum=0.7)(act_1)

    conv_2 = Conv1D(num_filters, kernel_size=(filter_sizes[2]))(inputs)
    act_2 = Activation('relu')(conv_2)
    bn_2 = BatchNormalization(momentum=0.7)(act_2)

    conv_3 = Conv1D(num_filters, kernel_size=(filter_sizes[3]))(inputs)
    act_3 = Activation('relu')(conv_3)
    bn_3 = BatchNormalization(momentum=0.7)(act_3)

    conv_4 = Conv1D(num_filters, kernel_size=(filter_sizes[4]))(inputs)
    act_4 = Activation('relu')(conv_4)
    bn_4 = BatchNormalization(momentum=0.7)(act_4)

    maxpool_0 = MaxPooling1D(pool_size=(max_len - filter_sizes[0]))(bn_0)
    maxpool_1 = MaxPooling1D(pool_size=(max_len - filter_sizes[1]))(bn_1)
    maxpool_2 = MaxPooling1D(pool_size=(max_len - filter_sizes[2]))(bn_2)
    maxpool_3 = MaxPooling1D(pool_size=(max_len - filter_sizes[3]))(bn_3)
    maxpool_4 = MaxPooling1D(pool_size=(max_len - filter_sizes[4]))(bn_4)

    concatenated_tensor = Concatenate()([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=1, activation='sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)
    #model = multi_gpu_model(model, gpus=gpus)
    model.summary()
    model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')
    return model

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
bilm = BidirectionalLanguageModel(options_file_path, weights_file_path)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

for i in training_articles:

    article_text = dataset_articles_text[i]

    article_sentences = sent_tokenize(article_text)
    if len(article_sentences) > 200:
        article_sentences = article_sentences[:200]

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
            for sent_no in range(len(article_sentences))
        ])

    padded_embedding = sequence.pad_sequences([article_sentences_embeddings], maxlen=200, dtype='float32')[0]
