# %% [markdown]
# In this notebook we describe the process of creating a deep learning model from scratch in PyTorch. We implement the approched described in this [paper](https://arxiv.org/pdf/1408.5882.pdf) for classifying sentences using Convolutional Neural Networks. In particular, we will classify sentences into "subjective" or "objective".

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %matplotlib inline

# %% [markdown]
# ## Subjectivity Dataset

# %% [markdown]
# The subjectivity dataset has 5000 subjective and 5000 objective processed sentences. To get the data:
# ```
# wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
# ```
# From the **README** file:
# - quote.tok.gt9.5000 contains 5000 subjective sentences (or snippets)
# - plot.tok.gt9.5000 contains 5000 objective sentences

# %% [markdown]
# The following code chunk download and unzip the data into a folder **data**

# %%
# ! wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rotten_imdb.tar.gz
# ! wget http://nlp.stanford.edu/data/glove.6B.zip
# ! mkdir data
# ! tar -xvf rotten_imdb.tar.gz -C data
# ! unzip glove.6B.zip -d data

# %% [markdown]
# ## Reading and splitting data

# %%
import numpy as np
from collections import defaultdict
import re

def read_file(path):
    """ Read file returns a shuttled list.
    """
    with open(path, encoding = "ISO-8859-1") as f:
        content = np.array(f.readlines())
    return content

def get_vocab(content):
    """Computes Dict of counts of words.
    Computes the number of times a word is on a document.
    """
    vocab = defaultdict(float)
    for line in content:
        words = set(line.split())
        for word in words:
            vocab[word] += 1
    return vocab


# %% [markdown]
# It's a common and good practice to divide the original dataset into **training**, **validation** and **test**. The **training** dataset would be used to learn the parameters of the model, the **validation** dataset to set hyperparameters such as the number of epochs or the learning rate and finally, the metrics would be calculated over the **test** dataset.

# %%
PATH = './dataset/'
sub_content = read_file(PATH + "quote.tok.gt9.5000")
obj_content = read_file(PATH + "plot.tok.gt9.5000")
sub_content = np.array([line.strip().lower() for line in sub_content])
obj_content = np.array([line.strip().lower() for line in obj_content])
sub_y = np.zeros(len(sub_content))
obj_y = np.ones(len(obj_content))
X = np.append(sub_content, obj_content)
y = np.append(sub_y, obj_y)

# %%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
# getting vocab from training sets
data_vocab = get_vocab(X_train)


# %% [markdown]
# ## Embedding Layer

# %% [markdown]
# ### Initializing embedding layer with Glove embeddings

# %% [markdown]
# To get glove pre-trained embeddings:
#     `wget http://nlp.stanford.edu/data/glove.6B.zip`

# %% [markdown]
# In this section I am keeping the whole Glove embeddings. You can decide to keep just the words on your training set.

# %%
# ! head -2 data/glove.6B.50d.txt

# %% [markdown]
# We would like to initialize the embeddings from our model with the pre-trained Glove embeddings. After initializing we should "freeze" the embeddings at least initially. The rationale is that we first want the network to learn weights for the other parameters that were randomly initialized. After that phase, we could finetune the embeddings to our task.
#
# `embed.weight.requires_grad = False` freezes the embedding parameters.

# %% [markdown]
# The following code initializes the embedding. Here `V` is the vocabulary size and `D` is the embedding size. `pretrained_weight` is a numpy matrix of shape `(V, D)`.

# %% [markdown]
# After loading word embeddings and defining our final vocabulary, it is necessary to create an embedding matrix that holds the pretrained weights in an appropriate format as an input for the model.

# %% [markdown]
# ## Encoding data

# %% [markdown]
# Computers are not able to process raw text as humans do, this is why we should encode text into numeric vectors. We will be using 1D Convolutional neural networks as our model. CNNs assume a fixed input size so we need to assume a fixed size and truncate or pad the sentences as needed. Let's find a good value to set our sequence length to.

# %%

from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf
from tqdm import tqdm
from nltk.tokenize import word_tokenize

# For ELMo
vocabulary_file_path = "./elmo/vocabulary.txt"
options_file_path = "./elmo/options.json"
weights_file_path = "./elmo/weights.hdf5"

batcher = Batcher(vocabulary_file_path, 50)
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
bilm = BidirectionalLanguageModel(options_file_path, weights_file_path)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

# %%
def encode_sentence(s, s_name):
    # enc = np.zeros(N, dtype=np.int32)
    # enc1 = np.array([vocab2index.get(w, vocab2index["UNK"]) for w in s.split()])
    # l = min(N, len(enc1))
    # enc[:l] = enc1[:l]
    # return enc

    a = int(len(s)/100)
    word_embeddings = []

    for i in tqdm(range(a)):
        with tf.Session() as sess:
            try:
                sess.run(tf.global_variables_initializer())

                sentences = s[100*i:100*(i+1)]
                words = [word_tokenize(sentence)[:40] for sentence in sentences]

                context_ids = batcher.batch_sentences(words)

                # emb = article_elmo_embeddings[sent_no, word_no]
                word_embeddings += sess.run(
                    elmo_context_input['weighted_op'],
                    feed_dict={context_character_ids: context_ids}
                ).tolist()

            except Exception as err:
                print(f"Failed in {s_name} on sentence number: {i}")
                print(err)
                exit(0)

    return word_embeddings

print("Getting ELMo embeddings for sentences")
x_train = encode_sentence(X_train, "X_train")
x_val = encode_sentence(X_val, "X_val")
print("Got ELMo embeddings for sentences")

# %% [markdown]
# ## 1D Convolutional Model

# %% [markdown]
# Notation:
# * V -- vocabulary size
# * D -- embedding size
# * N -- MAX Sentence length

# %%
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


# %% [markdown]
# ## Training and Evaluation

# %% [markdown]
# Note that we are not bodering with mini-batches since our dataset is small.

# %%
D = 1024
model = SentenceCNN(D) # .cuda()


# %%
def evaluate(model, x, y):
    model.eval()
    y_hat = model(x)
    loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_pred = y_hat > 0
    correct = (y_pred.float() == y).float().sum()
    accuracy = correct/y_pred.shape[0]
    return {'loss': loss.item(), 'accuracy': accuracy.item()}


# %%
# accuracy of a random model should be around 0.5
print("Running on validation data for randomly initialized dataset")
x_val = torch.FloatTensor(x_val)
y_val = torch.FloatTensor(y_val).unsqueeze(1) # .cuda()

# %%
evaluate(model, x_val, y_val)


# %%
def train_epocs(model, epochs=10, lr=0.01):
    metrics = []
    parameters = filter(lambda p: p.requires_grad, model.parameters()) #filters parameters with p.requires_grad=True
    optimizer = torch.optim.Adam(parameters, lr=lr)
    model.train()
    for i in range(epochs):
        model.train()
        x = torch.FloatTensor(x_train)  # # .cuda()
        y = torch.Tensor(y_train).unsqueeze(1)
        y_hat = model(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ev = evaluate(model, x, y)
        val_ev = evaluate(model, x_val, y_val)
        metrics.append(ev)
        print("Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(i, ev['loss'], ev['accuracy'], val_ev['loss'], val_ev['accuracy']))
        torch.save(model, f"./models/model_epoch_{i}")
    return metrics


# %%
train_epocs(model, epochs=500, lr=0.001)
print('validation set:', evaluate(model, x_val, y_val))
# %%
# %% [markdown]
# ## References

# %% [markdown]
# The CNN is adapted from here https://github.com/junwang4/CNN-sentence-classification-pytorch-2017/blob/master/cnn_pytorch.py.
# Code for the original paper can be found here https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py.
