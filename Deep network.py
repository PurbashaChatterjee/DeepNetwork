'''
Created on Sep 21, 2017

@author: purbasha
'''

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Input, Embedding, Dropout, GlobalMaxPooling1D, Flatten, TimeDistributed, LSTM, ConvLSTM2D
from keras.layers.core import Dense, Lambda, Reshape, Permute, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.layers import merge, Merge
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers.recurrent import Recurrent, _time_distributed_dense
from keras.engine import InputSpec
from keras.engine.topology import Layer, InputSpec
import numpy as np
import os,sys,csv,math
import keras
import pandas as pd
from sklearn.model_selection  import train_test_split
from keras.layers import BatchNormalization
from PlotLearning import PlotLearning
from sklearn.metrics import f1_score


hidden_dims = 200
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
k=128
emb_dim = 50
FILTER_LENGTH = 5
MAX_NB_WORDS = 14568
MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.1
print('Indexing word vectors.')
embeddings_index = {}
f = open('VMShare/glove.6B.50d.txt','r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

texts = []  # list of text samples
filename = 'ProcessedData.csv'
filereader = pd.read_csv(filename, sep='\t', header = None)

txt_cmt = []
txt_topic = []
txt_label = []

filereader.columns = ["Answers","Score","Depth","Questions","Ups","Downs","Gilded","Time","Junk"]

def maxScore():
    scores = pd.qcut((filereader['Score']),3, labels=[0,1,2])
    return scores      

def normScore():
    '''
    Normalizing the score and creating lists of comments,topic and labels
    '''
    scores =  maxScore()
    txt_cmt = filereader["Answers"].tolist()
    txt_topic = filereader["Questions"].tolist()         
    txt_label = scores
    return txt_cmt, txt_topic, txt_label 

txt_cmt, txt_topic, txt_label = normScore()

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(txt_cmt)
sequences_cmt = tokenizer.texts_to_sequences(txt_cmt)
tokenizer.fit_on_texts(txt_topic)
sequences_topic = tokenizer.texts_to_sequences(txt_topic)
labels = np.array(txt_label)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_cmt = pad_sequences(sequences_cmt, maxlen=MAX_SEQUENCE_LENGTH)
data_topic = pad_sequences(sequences_topic, maxlen=MAX_SEQUENCE_LENGTH)

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, emb_dim))

def embedding():
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix        

ans_train, ans_val, qns_train, qns_val, score_train, score_val = \
   train_test_split(data_cmt, data_topic, labels, test_size=0.2, random_state=1)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
ocnt =0 
zcnt = 0
thcnt = 0
       
NFILTERS = 16
def create_cnn_nn(embWeights=None, wordRnnSize=100, 
                                  dropWordEmb = 0.2, dropWordRnnOut = 0.2):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, emb_dim,  weights=[embedding()],input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
    model.add(Reshape((MAX_SEQUENCE_LENGTH,emb_dim,1,1)))
    model.add(Conv2D(NFILTERS, kernel_size=(3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(Dropout(dropWordRnnOut))
    return model

if __name__ == '__main__':

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    cnn_nn = create_cnn_nn()
    processed_a = cnn_nn(sequence_1_input)
    processed_b = cnn_nn(sequence_2_input)
    print processed_a, processed_b

    x = Reshape((MAX_SEQUENCE_LENGTH, emb_dim*NFILTERS))(processed_b)
    x = Permute((2,1))(x)
    qns_output1 = GlobalMaxPooling1D()(x)
    
    x = Reshape((MAX_SEQUENCE_LENGTH, emb_dim*NFILTERS))(processed_a)
    x = Permute((2,1))(x)
    ans_output1 = GlobalMaxPooling1D()(x)
    
    qns_ans_tensors = merge([qns_output1, ans_output1], mode='concat')
    x = Dense(200, activation = 'relu', kernel_regularizer='l2')(qns_ans_tensors)
    x = Dense(50, activation = 'relu', kernel_regularizer='l2')(x)
    deep_output = Dense(3, activation = 'softmax', kernel_regularizer='l2')(x)
    deep_cnn_model = Model(inputs = [sequence_1_input, sequence_2_input], outputs = deep_output)
    print(deep_cnn_model.summary())

plot_learning = PlotLearning()    
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5)
deep_cnn_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_cb1 = keras.callbacks.ModelCheckpoint('/home/purbasha/models/model.h5', monitor='val_loss', verbose=0, 
                                      save_best_only=True, save_weights_only=False)
model_cb2 = keras.callbacks.TensorBoard(log_dir="/home/purbasha/logs")
wide_deep_dense_model.fit([ans_train, qns_train], score_train,
          validation_data=([ans_val, qns_val], score_val),
          batch_size=64, epochs=2, callbacks=[model_cb2, plot_learning])

test_scores = wide_deep_dense_model.predict([ans_val, qns_val])
roc_pr_curves(score_val, test_scores, "CNN", 1, 1)
f1_score(score_val, test_scores, "CNN", 1, 1)
