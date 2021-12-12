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

    # Add the values to their respective lists
    dataset_articles_text.append(article_text.encode('ascii', 'ignore').decode())
    if truth_attributes["hyperpartisan"] == "true":
        dataset_articles_truth_value.append(1)
    else:
        dataset_articles_truth_value.append(0)
    dataset_articles_id.append(truth_attributes["id"])

    if len(dataset_articles_text) > 100:
        break

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

if len(sys.argv) == 1:
    seed = 10
else:
    seed = int(sys.argv[1])
random.seed(seed)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in tqdm(range(len(dataset_articles_text))):

        try:
            article_text = dataset_articles_text[i]

            article_sentences = sent_tokenize(article_text)
            if len(article_sentences) > 200:
                article_sentences = article_sentences[:200]

            article_words = [word_tokenize(sentence) for sentence in article_sentences]
            np.set_printoptions(threshold=sys.maxsize)

            context_ids = batcher.batch_sentences(article_words)

            article_elmo_embeddings = sess.run(
                elmo_context_input['weighted_op'],
                feed_dict={context_character_ids: context_ids}
            )

            article_sentences_embeddings = np.average(
                    article_elmo_embeddings, axis=1
                    )

            padded_embedding = sequence.pad_sequences([article_sentences_embeddings], maxlen=200, dtype='float32')[0]

            dataset_elmo_embeddings.append(padded_embedding)

        except Exception as err:
            print(dataset_articles_id[i])
            print(dataset_articles_text[i])
            print(err)
            exit(0)

dataset_elmo_embeddings_np = np.array(dataset_elmo_embeddings)
dataset_articles_id_np = np.array(dataset_articles_id)
dataset_articles_truth_value_np = np.array(dataset_articles_truth_value)

cnn_model = conv1d_BN(200, 1024)
checkpoints = ModelCheckpoint(filepath='./models/CNN_elmo_checkpoint.hdf5',
                              verbose=0, monitor='val_acc', save_best_only=True)

history = cnn_model.fit(
        x=dataset_elmo_embeddings_np[training_articles], y=dataset_articles_truth_value_np[training_articles],
        batch_size=32, verbose=0, epochs=1,
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
    rf.write(f"Final score for seed = {seed} is {cv_score}\n")

print("Final Score of the Model:", cv_score)
