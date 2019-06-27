import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import random
import keras
import os
import json
from keras import regularizers
from keras.models import Model
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding
from keras.optimizers import RMSprop
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import Bidirectional,LSTM
from lstmEmbedding.lstm_model import LSTM_Model
from lstmEmbedding.utils import read_data
from lstmEmbedding.utils import read_label
from lstmEmbedding.utils import wv2embedding
from lstmEmbedding.attention import AttentionLayer

num_classes = 43
epochs = 20

#欧式距离
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

#余弦距离
def cosine(vects):
    x, y = vects
    #求模
    x_norm = K.sqrt(K.sum(K.square(x), axis=1))
    y_norm = K.sqrt(K.sum(K.square(y), axis=1))
    #内积
    x_y = K.sum(K.dot(x, y), axis=1,keepdims=True)
    cosin = x_y/(x_norm * y_norm)
    return cosin

#曼哈顿距离
def mandist(vects):
    x, y = vects
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))



def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    for i in range(num_classes):
        j = len(digit_indices[i])
        for k in range(j):
            z1 = digit_indices[i][k]
            for m in range(10):
                inc = random.randrange(1, j)
                dn = (k + inc) % j
                z2 = digit_indices[i][dn]
                pairs += [[x[z1], x[z2]]]
                labels += [1]
#             while len(digit_indices[dn]) == 0:
#                 inc = random.randrange(1, num_classes)
#                 dn = (i + inc) % num_classes
#             p = len(digit_indices[dn])
#             for q in range(p):
#                 z3 = digit_indices[dn][q]
#                 pairs += [[x[z1], x[z3]]]
#                 labels += [0]   

    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    embedded_seq = Embedding(input_dim=len(id2token),
                             output_dim=embedding_dim,
                             input_length=maxlen,
                             weights=[wv_embeddings],
                             trainable=True)(input)
    l_lstm = LSTM(units=lstm_unit,
                  activation=lstm_activation,
                  return_sequences=True,
                  kernel_regularizer=regularizers.l2(reg),
                  dropout=drop_out,
                  unroll=True)(embedded_seq)

    l_att = AttentionLayer()(l_lstm)
    return Model(input, l_att)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


CONFIG = json.load(open("config.json", 'r'))

DATA_CONFIG = CONFIG.get("data")
MODEL_CONFIG = CONFIG.get("model")
LSTM_CONFIG = CONFIG.get("lstm_config")

# 训练与验证数据
train_file = DATA_CONFIG.get('train')
validate_file = DATA_CONFIG.get('validate')
test_file = DATA_CONFIG.get('test')
label_file = DATA_CONFIG.get('label')

train_data_path = os.path.join('train_lstm_data', train_file)
validate_data_path = os.path.join('train_lstm_data', validate_file)
test_data_path = os.path.join('train_lstm_data', test_file)
label_path = os.path.join('train_lstm_data', label_file)


# word2vec 初始化路径
wv_model_file = MODEL_CONFIG.get('word2vec')
wv_model_path = os.path.join('w2v_model', wv_model_file)

data_dict = wv2embedding(wv_model_path)
wv_embeddings = data_dict['wv_embeddings']
id2token = data_dict['id2token']
token2id = data_dict['token2id']

label2id = read_label(label_path)
print(label2id)

# 设置 lstm 模型参数
maxlen           =  LSTM_CONFIG.get("maxlen", 40)
lstm_unit        =  LSTM_CONFIG.get("units", 128)
lstm_activation  =  LSTM_CONFIG.get("activation", None)
batch_size       =  LSTM_CONFIG.get("batch_size", 10)
drop_out         =  LSTM_CONFIG.get("dropout", 0.2)
reg              =  LSTM_CONFIG.get("l2_reg", 0.03)
embedding_dim    =  LSTM_CONFIG.get("embedding_dim", 256)
num_classes      =  LSTM_CONFIG.get("num_classes")


# 读取训练数据
df_data = read_data(token2id, label2id, maxlen, train_data_path, drop_stop_word=True)
print(df_data)
df_test = read_data(token2id, label2id, maxlen, test_data_path, drop_stop_word=True)
x_train = np.array(df_data['data'].tolist())
print(x_train.shape)
y_train = np.array(df_data['labels'].tolist())
print(y_train.shape)
x_validate = x_train
y_validate = y_train
x_test = np.array(df_test['data'].tolist())
y_test = np.array(df_test['labels'].tolist())
input_shape = x_train.shape[1:]
print(input_shape)

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
print(tr_pairs.shape)
print(tr_y.shape)
print(np.count_nonzero(tr_y))

# digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)
# print(te_pairs.shape)
# print(te_y.shape)
# print(np.count_nonzero(te_y))
# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape, dtype='int32', name='sequence1')
input_b = Input(shape=input_shape, dtype='int32', name='sequence2')

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(mandist,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
# keras.utils.plot_model(model, "siamModel.png", show_shapes=True)
model.summary()

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                    batch_size=64,
                    epochs=epochs,
                    validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')
plt.subplot(1, 2, 2)
plot_train_history(history, 'accuracy', 'val_accuracy')
plt.show()

model.save("lstm_model/siameseLSTM430.model")

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))