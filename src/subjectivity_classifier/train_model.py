import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import floor

from bilm import Batcher, BidirectionalLanguageModel, weight_layers
import tensorflow as tf
from tqdm import tqdm
from nltk.tokenize import word_tokenize

def read_file(path):
    with open(path, encoding = "ISO-8859-1") as f:
        content = np.array(f.readlines())
    return content

sub_content = None
obj_content = None
with open('./dataset/quote.tok.gt9.5000', encoding = "ISO-8859-1") as f:
    sub_content = np.array(f.readlines())
with open('./dataset/plot.tok.gt9.5000', encoding = "ISO-8859-1") as f:
    obj_content = np.array(f.readlines())

sub_content = np.array([line.strip().lower() for line in sub_content])
obj_content = np.array([line.strip().lower() for line in obj_content])

sub_y = np.zeros(len(sub_content))
obj_y = np.ones(len(obj_content))
X = np.append(sub_content, obj_content)
y = np.append(sub_y, obj_y)

num_sentences = len(X)

training_articles = random.sample(range(num_sentences), floor(num_sentences*0.8))
testing_articles = [_ for _ in range(num_sentences) if _ not in training_articles]
X_train = [X[i] for i in training_articles]
y_train = [y[i] for i in training_articles]
X_val = [X[i] for i in testing_articles]
y_val = [y[i] for i in testing_articles]

vocabulary_file_path = "./elmo/vocabulary.txt"
options_file_path = "./elmo/options.json"
weights_file_path = "./elmo/weights.hdf5"

batcher = Batcher(vocabulary_file_path, 50)
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
bilm = BidirectionalLanguageModel(options_file_path, weights_file_path)

context_embeddings_op = bilm(context_character_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

def encode_sentence(s, s_name):

    a = int(len(s)/100)
    word_embeddings = []

    for i in tqdm(range(a)):
        with tf.Session() as sess:
            try:
                sess.run(tf.global_variables_initializer())

                sentences = s[100*i:100*(i+1)]
                words = [word_tokenize(sentence)[:40] for sentence in sentences]

                context_ids = batcher.batch_sentences(words)

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

class SentenceCNN(nn.Module):
    def __init__(self, D):
        super(SentenceCNN, self).__init__()

        self.conv_3 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=3)
        self.conv_4 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=4)
        self.conv_5 = nn.Conv1d(in_channels=D, out_channels=100, kernel_size=5)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(300, 1)

    def forward(self, x):
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


D = 1024
model = SentenceCNN(D) # .cuda()

def evaluate(model, x, y):
    model.eval()
    y_hat = model(x)
    loss = F.binary_cross_entropy_with_logits(y_hat, y)
    y_pred = y_hat > 0
    correct = (y_pred.float() == y).float().sum()
    accuracy = correct/y_pred.shape[0]
    return {'loss': loss.item(), 'accuracy': accuracy.item()}


print("Running on validation data for randomly initialized dataset")
x_val = torch.FloatTensor(x_val)
y_val = torch.FloatTensor(y_val).unsqueeze(1) # .cuda()

evaluate(model, x_val, y_val)


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


train_epocs(model, epochs=500, lr=0.001)
print('validation set:', evaluate(model, x_val, y_val))
