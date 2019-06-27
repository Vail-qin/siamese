import numpy as np
import jieba
jieba.set_dictionary('/home/yao.qin/semantic_analysis/dict.txt')
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional
import matplotlib.pyplot as plt
import os
import gensim
import gensim.models
import numpy as np
import random
import pandas as pd
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SiameseNetwork:
    def __init__(self):
        self.train_path = os.path.join('siamese', 'sent_pair_new.csv')
        self.test_path = os.path.join('siamese','test.csv')
#         self.test_path = os.path.join('siamese','test.csv')
#         self.vocab_path = os.path.join('data', 'vocab2.txt')
        self.embedding_file = os.path.join('w2v_model', 'skipgram_320_2.vec')
        self.model_path = os.path.join('siamese_model_518.h5')
        self.stop_words = self.load_stop_words()
        self.data_dict = self.wv2embedding()
        self.word_dict = self.data_dict['id2token']
        self.token2id = self.data_dict['token2id']
        self.embedding_matrix = self.data_dict['wv_embeddings']
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 20
        self.BATCH_SIZE = 512
        self.NUM_CLASSES = 2
        self.VOCAB_SIZE = len(self.word_dict)
        self.LIMIT_RATE = 0.95
        self.TIME_STAMPS = self.select_best_length()
        self.data = self.read_data()
        self.testdata = self.read_test_data()

    '''根据样本长度,选择最佳的样本max-length'''
    def select_best_length(self):
        len_list = []
        max_length = 0
        cover_rate = 0.0
        for line in open(self.train_path):
            line = line.strip().split(',')
            if not line:
                continue
            sent = line[0]
            sent_len = len(sent)
            len_list.append(sent_len)
        all_sent = len(len_list)
        sum_length = 0
        len_dict = Counter(len_list).most_common()
        for i in len_dict:
            sum_length += i[1]*i[0]
        average_length = sum_length/all_sent
        for i in len_dict:
            rate = i[1]/all_sent
            cover_rate += rate
            if cover_rate >= self.LIMIT_RATE:
                max_length = i[0]
                break
        print('average_length:', average_length)
        print('max_length:', max_length)
        return max_length

    '''构造数据集'''
#     def build_data(self):
#         sample_x = []
#         sample_y = []
#         sample_x_left = []
#         sample_x_right = []
#         sample_x_test = []
#         sample_y_test = []
#         sample_x_left_test = []
#         sample_x_right_test = []
#         vocabs = {'UNK'}
#         for line in open(self.train_path):
#             line = line.rstrip().split(',')
#             if not line:
#                 continue
#             sent_left = line[0]
#             sent_right = line[1]
#             label = line[4]
#             sample_x_left.append([char for char in sent_left if char])
#             sample_x_right.append([char for char in sent_right if char])
#             sample_y.append(label)
#             for char in [char for char in sent_left + sent_right if char]:
#                 vocabs.add(char)
                
#         for line in open(self.test_path):
#             line = line.rstrip().split(',')
#             if not line or line[7]=='target':
#                 continue
#             sent_left = line[2]
#             sent_right = line[3]
#             label = line[7]
#             sample_x_left_test.append([char for char in sent_left if char])
#             sample_x_right_test.append([char for char in sent_right if char])
#             sample_y_test.append(label)
#             for char in [char for char in sent_left + sent_right if char]:
#                 vocabs.add(char)
                
#         print(len(sample_x_left), len(sample_x_right))
#         sample_x = [sample_x_left, sample_x_right]

#         datas = [sample_x, sample_y]
#         word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
#         self.write_file(list(vocabs), self.vocab_path)
#         return datas,word_dict
        

    '''将数据转换成keras所需的格式'''
#     def modify_data(self):
#         sample_x = self.datas[0]
#         sample_y = self.datas[1]
#         sample_x_left = sample_x[0]
#         sample_x_right = sample_x[1]
#         left_x_train = [[self.word_dict[char] for char in data] for data in sample_x_left]
#         right_x_train = [[self.word_dict[char] for char in data] for data in sample_x_right]
#         y_train = [int(i) for i in sample_y]
#         left_x_train = pad_sequences(left_x_train, self.TIME_STAMPS)
#         right_x_train = pad_sequences(right_x_train, self.TIME_STAMPS)
#         y_train = np.expand_dims(y_train, 2)
#         return left_x_train, right_x_train, y_train

    '''保存字典文件'''
#     def write_file(self, wordlist, filepath):
#         with open(filepath, 'w+') as f:
#             f.write('\n'.join(wordlist))

    '''加载预训练词向量'''
#     def load_pretrained_embedding(self):
#         embeddings_dict = {}
#         with open(self.embedding_file, 'r') as f:
#             for line in f:
#                 values = line.strip().split(' ')
#                 if len(values) < 300:
#                     continue
#                 word = values[0]
#                 coefs = np.asarray(values[1:], dtype='float32')
#                 embeddings_dict[word] = coefs
#         print('Found %s word vectors.' % len(embeddings_dict))
#         return embeddings_dict

    '''加载词向量矩阵'''
#     def build_embedding_matrix(self):
#         embedding_dict = self.load_pretrained_embedding()
#         embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
#         for word, i in self.word_dict.items():
#             embedding_vector = embedding_dict.get(word)
#             if embedding_vector is not None:
#                 embedding_matrix[i] = embedding_vector
#         return embedding_matrix


    def read_data(self,drop_stop_word=True):
        '''
        读取数据，返回msg的id列表，返回type的id列表；一条msg一条列表，对应的type为一个label
        :param word_id_dict: a dict of 'word' and 'id', the key is 'word', and the value 'id'
        :param path:
        :return:
        '''
        df = pd.read_csv(self.train_path,names=['msg','msg1','type1','type2','label'])
        maxlen = self.TIME_STAMPS
        df['tokens'] = df.msg.apply(jieba.lcut, HMM=False)
        df['tokens1'] = df.msg1.apply(jieba.lcut, HMM=False)
        if drop_stop_word:
            df['tokens'] = df.tokens.apply(lambda x: [t for t in x if t not in self.stop_words])
            df['tokens1'] = df.tokens1.apply(lambda x: [t for t in x if t not in self.stop_words])
        df['word_sequence'] = df.tokens.apply(lambda x: list(map(lambda t: self.token2id.get(t, self.token2id.get('UNK')), x)))
        df['word_sequence1'] = df.tokens1.apply(lambda x: list(map(lambda t: self.token2id.get(t, self.token2id.get('UNK')), x)))
        df['left'] = df.word_sequence.apply(lambda x: x[:maxlen] if len(x) >= maxlen else x + [0] * (maxlen - len(x)))
        df['right'] = df.word_sequence1.apply(lambda x: x[:maxlen] if len(x) >= maxlen else x + [0] * (maxlen - len(x)))
        return df
    
    
    def read_test_data(self,drop_stop_word=True):
        df = pd.read_csv(self.test_path)
        maxlen = self.TIME_STAMPS
        df['tokens'] = df.msg.apply(jieba.lcut, HMM=False)
        df['tokens1'] = df.corpus.apply(jieba.lcut, HMM=False)
        if drop_stop_word:
            df['tokens'] = df.tokens.apply(lambda x: [t for t in x if t not in self.stop_words])
            df['tokens1'] = df.tokens1.apply(lambda x: [t for t in x if t not in self.stop_words])
        df['word_sequence'] = df.tokens.apply(lambda x: list(map(lambda t: self.token2id.get(t, self.token2id.get('UNK')), x)))
        df['word_sequence1'] = df.tokens1.apply(lambda x: list(map(lambda t: self.token2id.get(t, self.token2id.get('UNK')), x)))
        df['left'] = df.word_sequence.apply(lambda x: x[:maxlen] if len(x) >= maxlen else x + [0] * (maxlen - len(x)))
        df['right'] = df.word_sequence1.apply(lambda x: x[:maxlen] if len(x) >= maxlen else x + [0] * (maxlen - len(x)))
        return df
        
    
    
    def modify_data(self):
        df_data= self.data
        left_x_train = df_data['left'].tolist()
        right_x_train = df_data['right'].tolist()
        y_train = df_data['label'].tolist()
        y_train = np.expand_dims(y_train, axis=1)
        return left_x_train, right_x_train, y_train 
    
    
    def modify_test_data(self):
        df_data= self.testdata
        left_x_test = df_data['left'].tolist()
        right_x_test = df_data['right'].tolist()
        y_test = df_data['target'].tolist()
        y_test = np.expand_dims(y_test, axis=1)
        return left_x_test, right_x_test, y_test
        
        


    # read wv model from file.
    def wv2embedding(self):
        wv_model_ = gensim.models.KeyedVectors.load_word2vec_format(self.embedding_file)
        data_dict = {}
        # set vecotr size.
        vector_size = wv_model_.vector_size
        # generate a unknown token.
        unk_vector = np.random.uniform(-5, 5, (1, vector_size))
        null_vector = np.zeros((1, vector_size))
        print(unk_vector.shape, null_vector.shape)
        # generate embeddings data.
        data_dict['wv_embeddings'] = np.concatenate([null_vector, wv_model_.wv.vectors, unk_vector], axis=0)
        data_dict['id2token'] = wv_model_.wv.index2word.copy()
        data_dict['id2token'].insert(0, 'NULL')
        data_dict['id2token'].append('UNK')
        data_dict['token2id'] = {token: index for index, token in enumerate(data_dict['id2token'])}
        return data_dict
    
    def load_stop_words(self):
        '''
        加载停用词，并返回一个停用词列表
        :return:
        '''
        stop_words = []
        print(os.path.realpath(os.path.curdir))
        path = os.path.join('data', 'stop_words.txt')
        fr = open(path,'r',encoding='utf-8')
        lines = fr.readlines()
        for line in lines:
            wd = line.strip()
            stop_words.append(wd)
        return set(stop_words)

    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    def exponent_neg_manhattan_distance(self, sent_left, sent_right):
        return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))

    '''基于欧式距离的字符串相似度计算'''
    def euclidean_distance(self, sent_left, sent_right):
        sum_square = K.sum(K.square(sent_left - sent_right), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    '''基于余弦距离的字符串相似度计算'''
    def cosine_distance(self,sent_left, sent_right):
        x_norm = K.sqrt(K.sum(sent_left*sent_left, axis=1,keepdims=True))
        y_norm = K.sqrt(K.sum(sent_right*sent_right, axis=1,keepdims=True))
        x_y = K.sum(sent_left*sent_right, axis=1,keepdims=True)
#         cosin = 1- K.dot(sent_left, sent_right)/(K.l2_normalize(sent_left)*K.l2_normalize(sent_right))
        return x_y / (x_norm * y_norm)


    def contrastive_loss(self,y_true, y_pred):
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    
    def accuracy(self,y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast((1 - y_pred) >= 0.85, y_true.dtype)))
        
        


    '''搭建编码层网络,用于权重共享'''
    def create_base_network(self, input_shape):
        input = Input(shape=input_shape)
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
        lstm1 = Dropout(0.2)(lstm1)
        lstm2 = Bidirectional(LSTM(32))(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        return Model(input, lstm2)

    '''搭建网络'''
    def bilstm_siamese_model(self):
        embedding_layer = Embedding(self.VOCAB_SIZE,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)

        left_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')
        right_input = Input(shape=(self.TIME_STAMPS,), dtype='float32')

        encoded_left = embedding_layer(left_input)
        encoded_right = embedding_layer(right_input)

        shared_lstm = self.create_base_network(input_shape=(self.TIME_STAMPS, self.EMBEDDING_DIM))
        left_output = shared_lstm(encoded_left)
        right_output = shared_lstm(encoded_right)

        distance = Lambda(lambda x: self.exponent_neg_manhattan_distance(x[0], x[1]),
                          output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

        model = Model([left_input, right_input], distance)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        return model


    '''训练模型'''
    def train_model(self):
        left_x_train, right_x_train, y_train = self.modify_data()
        print(y_train)
        model = self.bilstm_siamese_model()
        check_point = ModelCheckpoint(filepath=self.model_path,
                                      monitor='val_loss',
                                      save_best_only=True,
                                      period=1,
                                      verbose=1,
                                      save_weights_only=True,
                                      mode='min')
        history = model.fit(
                              x=[left_x_train, right_x_train],
                              y=y_train,
                              validation_split=0.1,
                              batch_size=self.BATCH_SIZE,
                              epochs=self.EPOCHS,
                              callbacks=[check_point]
                            )
#         self.draw_train(history)
#         model.save(self.model_path)
        return model
    
    
#      '''测试模型'''
    def test_model(self):
        left_x_test, right_x_test, y_test = self.modify_test_data()
        print(y_test)
        model = self.bilstm_siamese_model()
        model.load_weights(self.model_path)
#         rr = model.evaludate([left_x_test, right_x_test], y_test)
#         print("Test loss: {}.\nTest accuracy: {}".format(rr[0], rr[1]))
        y=model.predict(x=[left_x_test, right_x_test],batch_size=self.BATCH_SIZE)
        print(y)
        prediction_list = y.tolist()
        test=pd.DataFrame(columns=['prob'],data=prediction_list)
        test.to_csv("siamese/test_result518_1.csv",index=None,encoding='utf-8')
        accuracy = 0
        for i in range(len(prediction_list)):
            if prediction_list[i][0] < 0.5:
                predict_pro = 0
            else:
                predict_pro = 1
            if predict_pro == y_test[i]:
                accuracy += 1
        print(accuracy / len(y_test))
        
        

    '''绘制训练曲线'''
    def draw_train(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


if __name__ == '__main__':
    handler = SiameseNetwork()
#     handler.train_model()
    handler.test_model()
    

