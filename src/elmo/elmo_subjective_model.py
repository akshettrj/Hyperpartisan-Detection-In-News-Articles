import xml.etree.ElementTree as ET
import random
from math import floor
import tensorflow as tf
import numpy as np
from bilm import Batcher, BidirectionalLanguageModel, weight_layers
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import sys

from keras.preprocessing import sequence
from keras.layers import Input, Flatten, Dense, Activation
from keras.layers import Concatenate,Conv1D,MaxPooling1D,BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

class SentenceCNN(nn.Module):
    def __init__(self, D):
        super(SentenceCNN, self).__init__()

        self.conv_3 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=3)
        self.conv_4 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=4)
        self.conv_5 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=5)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(300, 1)

    def forward(self, x):
        # x = self.embedding(x)
        x = x.transpose(1,2)
        x3 = F.relu(self.conv_3(x))
        x4 = F.relu(self.conv_4(x))
        x5 = F.relu(self.conv_5(x))
        x3 = nn.MaxPool1d(kernel_size = 38)(x3)
        x4 = nn.MaxPool1d(kernel_size = 37)(x4)
        x5 = nn.MaxPool1d(kernel_size = 36)(x5)
        out = torch.cat([x3, x4, x5], 2)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)

def conv1d_BN(max_len, embed_size):
    num_filters = 128
    filter_sizes = list(range(2, 7))
    maxpools = []

    inputs = Input(shape=(max_len,embed_size), dtype='float32')

    for i in range(5):
        conv_i = Conv1D(num_filters, kernel_size=(filter_sizes[i]))(inputs)
        act_i = Activation('relu')(conv_i)
        bn_i = BatchNormalization(momentum=0.7)(act_i)
        maxpools.append(MaxPooling1D(pool_size=(max_len - filter_sizes[i]))(bn_i))

    concatenated_tensor = Concatenate()(maxpools)
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=1, activation='sigmoid')(flatten)

    model = Model(inputs=inputs, outputs=output)
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
dataset_articles_id = []

for articles_child, truth_child in zip(articles_root, truth_root):
    articles_attributes = articles_child.attrib
    truth_attributes = truth_child.attrib

    article_text = [p.text for p in articles_child if p.text is not None]
    article_text = " ".join(article_text)

    if truth_attributes["id"] in [
            "0000219", "0000258", "0000364"
            ]:
        continue

    sentences = sent_tokenize(article_text)
    if len(sentences) == 0:
        continue

    # Add the values to their respective lists
    dataset_articles_text.append(article_text.encode('ascii', 'ignore').decode())
    if truth_attributes["hyperpartisan"] == "true":
        dataset_articles_truth_value.append(1)
    else:
        dataset_articles_truth_value.append(0)
    dataset_articles_id.append(truth_attributes["id"])


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

dataset_elmo_embeddings = []

seed = int(sys.argv[1])
random.seed(seed)

subjectivity_classifier_model = torch.load('./model_epoch_61')
subjectivity_classifier_model.eval()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in tqdm(range(len(dataset_articles_text))):

        article_text = dataset_articles_text[i]

        article_sentences = sent_tokenize(article_text)

        article_words = [word_tokenize(sentence) for sentence in article_sentences]
        np.set_printoptions(threshold=sys.maxsize)

        article_words += [["Apple"] * 40]

        context_ids = batcher.batch_sentences(article_words)

        article_elmo_embeddings = sess.run(
            elmo_context_input['weighted_op'],
            feed_dict={context_character_ids: context_ids}
        )

        subjective_sentences_embeddings = []
        objective_sentences_embeddings = []

        for indx, sentence_embedd in enumerate(article_elmo_embeddings[:-1]):

            is_subjective = subjectivity_classifier_model(
                    torch.FloatTensor(np.array([sentence_embedd[:40]]))
                    ).detach().numpy()[0][0] < 0

            if is_subjective:
                subjective_sentences_embeddings.append(sentence_embedd)
            else:
                objective_sentences_embeddings.append(sentence_embedd)

        if len(subjective_sentences_embeddings) > 0 and len(objective_sentences_embeddings) > 0:
            final_embeddings = np.concatenate((subjective_sentences_embeddings, objective_sentences_embeddings))
        elif len(subjective_sentences_embeddings) > 0:
            final_embeddings = subjective_sentences_embeddings
        else:
            final_embeddings = objective_sentences_embeddings

        article_sentences_embeddings = np.average(
                final_embeddings, axis=1
                )


        article_sentences_embeddings = article_sentences_embeddings[:200]
        padded_embedding = sequence.pad_sequences([article_sentences_embeddings], maxlen=200, dtype='float32')[0]

        dataset_elmo_embeddings.append(padded_embedding)

dataset_elmo_embeddings_np = np.array(dataset_elmo_embeddings)
dataset_articles_id_np = np.array(dataset_articles_id)
dataset_articles_truth_value_np = np.array(dataset_articles_truth_value)

cnn_model = conv1d_BN(200, 1024)
checkpoints = ModelCheckpoint(filepath='./models/CNN_elmo_checkpoint.hdf5',
                              verbose=0, monitor='val_acc', save_best_only=True)


history = cnn_model.fit(
        x=dataset_elmo_embeddings_np[training_articles], y=dataset_articles_truth_value_np[training_articles],
        batch_size=32, verbose=0, epochs=100,
        validation_data=(dataset_elmo_embeddings_np[testing_articles], dataset_articles_truth_value_np[testing_articles]),
        callbacks=[checkpoints]
        )

from sklearn.metrics import classification_report

y_pred = cnn_model.predict(dataset_elmo_embeddings_np[testing_articles], batch_size=32, verbose=1)
y_pred = y_pred.flatten()

y_pred_bool = []
for pred in y_pred:
    if pred >= 0.5:
        y_pred_bool.append(1)
    else:
        y_pred_bool.append(0)

y_pred_bool = np.array(y_pred_bool)

print(classification_report(dataset_articles_truth_value_np[testing_articles], y_pred_bool))

cv_score = history.history['val_acc'][-1]

with open("./results.txt", "a") as rf:
    rf.write(f"Final score of Baseline Model for seed = {seed} is {cv_score}\n")

print("Final Score of the Baseline Model:", cv_score)
