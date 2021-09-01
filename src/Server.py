from flask import *
from flask_socketio import SocketIO,emit
from flask_socketio import *
import json
import codecs
import pickle
import time
import os, gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras.layers import Dense, Dropout, Flatten
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
import random


class MlpMode:
    def load_mnistdata(self, data_folder):  # load mnist
        files = [
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
        ]

        paths = []
        for fname in files:
            paths.append(os.path.join(data_folder, fname))

        with gzip.open(paths[0], 'rb') as lbpath:
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[1], 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

        with gzip.open(paths[2], 'rb') as lbpath:
            y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(paths[3], 'rb') as imgpath:
            x_test = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

        x_train = x_train.astype('float') / 255  # 归一化[0-1]之间
        x_test = x_test.astype('float') / 255  # 归一化[0-1]之间

        y_train = to_categorical(y_train, 10)  # 对标签进行one-hot 编码
        y_test = to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def load_fatiguedata(self, data_folder):           #load fatigue
        files = [
            'fatiguedata.csv', 'fatiguelabel.csv',
        ]

        fileData = data_folder + '//' + files[0]
        fileLabel = data_folder + '//' + files[1]
        fatiguedata = pd.read_csv(fileData, header=1)  # 读取数据
        fatiguelabel = pd.read_csv(fileLabel, header=1)  # 读取label
        fatigutsize = np.array(fatiguedata).shape

        # index = [i for i in range(len(fatiguedata))]
        # random.shuffle(index)
        # data = fatiguedata[index]
        # label = fatiguelabel[index
        x_train = fatiguedata[1:300]  #data
        x_test = fatiguedata[300:fatigutsize[0]]

        y_train = fatiguelabel[1:300]  #label
        y_test = fatiguelabel[300:fatigutsize[0]]

        x_train = x_train.astype('float') / 255
        x_test = x_test.astype('float') / 255

        y_train = to_categorical(y_train - 1, 4)
        y_test = to_categorical(y_test - 1, 4)

        return (x_train, y_train), (x_test, y_test)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='iso-8859-1')
        return dict

    def load_cifar10(self, data_folder):  # load cifar
        files = [
            'data_batch_1', 'data_batch_2',
            'data_batch_3', 'data_batch_4',
            'data_batch_5'
        ]

        cifarData = []
        cifarLabel = []
        for filename in files:
            file = data_folder + '//' + filename  # 文件的路径

            dict_train_batch = self.unpickle(file)  # 将data_batch文件读入到数据结构(字典)中
            data_train_batch = dict_train_batch.get('data')  # 字典中取data
            labels = dict_train_batch.get('labels')  # 字典中取labels
            cifarData.extend(data_train_batch)
            cifarLabel.extend(labels)

        x_train = data_train_batch[1:40000]  # data
        x_test = data_train_batch[40000:50000]

        y_train = labels[1:40000]  # label
        y_test = labels[40000:50000]

        x_train = x_train.astype('float') / 255  # 归一化[0-1]之间
        x_test = x_test.astype('float') / 255  # 归一化[0-1]之间

        y_train = to_categorical(y_train, 10)  # 对标签进行one-hot 编码
        y_test = to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def constructModel(self, train_data):  # 构建模型
        self.model = tf.keras.Sequential()
        self.model.add(Flatten(input_shape=train_data.shape[1:]))
        # now: model.output_shape==(None,65536)
        # self.model.add(Dense( 64, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.2))
        # self.model.add(Dense(4, activation='softmax'))

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return self.model

    def trainMlp(self, weights, train_data, train_labels, batch_size, epoch):
        self.modelTrain(train_data, train_labels, weights, batch_size, epoch)
        weights = self.model.weights
        return weights

    def modelTrain(self, weights, train_data, train_labels, batch_size, epochs):
        self.model.set_weights(weights)

        hist = self.model.fit(train_data, train_labels, batch_size, epochs, validation_split=0.2, verbose=1, shuffle=True)

        currentWeights = self.model.get_weights()

        return currentWeights

    def modelEvaluate(self, test_data, test_labels):
        score = self.model.evaluate(test_data, test_labels, verbose=0)
        accuracy = 100 * score[1]
        print('Test accuracy: %.4f%%' % accuracy)

class secaggserver:
    def __init__(self, host, port, n, k):
        self.n = n
        self.k = k
        self.aggregate = 0
        self.finalAgregate = []
        self.y = 5
        self.iterTime = 0
        self.totalTime = 50
        self.host = host
        self.port = port
        self.numkeys = 0
        self.responses = 0
        self.readyNum = 0
        self.respset = set()
        self.resplist = []
        self.ready_client_ids = set()
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.register_handles()
        self.acc = []
        self.loss = []
        self.mlp = MlpMode()
        (train_data, train_labels), (test_data, test_labels) = self.mlp.load_mnistdata('MINST')
        # (train_data, train_labels), (test_data, test_labels) = self.mlp.load_fatiguedata('fatiguedata')
        # (train_data, train_labels), (test_data, test_labels) = self.mlp.load_cifar10('cifar10\\cifar-10-batches-py')
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.model = self.mlp.constructModel(self.train_data)



    def initialize_parameters_he(self, layers_dims):       #init weight
        np.random.seed(3)
        parameters = {}
        initParameters = []
        L = len(layers_dims) - 1  # integer representing the number of layers

        for l in range(1, L + 1):
            ### START CODE HERE ### (≈ 2 lines of code)
            parameters['W' + str(l)] = np.random.randn(layers_dims[l - 1], layers_dims[l]) * np.sqrt(
                2 / layers_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layers_dims[l],))
            ### END CODE HERE ###

            initParameters.append(parameters['W' + str(l)])
            initParameters.append(parameters['b' + str(l)])

        return initParameters

    def register_handles(self):

        @self.socketio.on("poxyReady")
        def handle_ready():
            self.readyNum += 1
            if self.readyNum == self.n:
                initWeights = self.initialize_parameters_he([784, 512, 512, 10])  # distribute grad to poxys

                msg = {
                    'W1': initWeights[0].tolist(),
                    'b1': initWeights[1].tolist(),
                    'W2': initWeights[2].tolist(),
                    'b2': initWeights[3].tolist(),
                    'W3': initWeights[4].tolist(),
                    'b3': initWeights[5].tolist()
                }

                print('------------------------distribute initweights to poxys-----------------------')

                for sid in self.ready_client_ids:
                    print('send grad to', sid)
                    self.socketio.emit('serverToPoxyGrad', msg, room=sid)
                    # self.socketio.emit('serverToPoxyGrad', 1, room=sid)
                self.readyNum = 0

        @self.socketio.on("wakeup")
        def handle_wakeup():
            print("Recieved wakeup from", request.sid)
            self.numkeys += 1

        @self.socketio.on("receive")
        def handle_rev():
            self.numkeys += 1
            if self.numkeys == self.n:
                for sid in self.ready_client_ids:
                    self.socketio.emit('disconnect', room=sid)


        @self.socketio.on("connect")
        def handle_connect():
            print(request.sid, " Connected")
            self.ready_client_ids.add(request.sid)
            print('Connected devices:', self.ready_client_ids)
            self.socketio.emit('test', 1, room=request.sid)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
            print(self.ready_client_ids)


        @self.socketio.on('poxyToServerGrad')
        def on_poxytoservergrad(*args):
            print('receive grad from poxys')
            msg = args[0]
            currentWeight = []
            currentWeight.append(np.array(msg['W1']))
            currentWeight.append(np.array(msg['b1']))
            currentWeight.append(np.array(msg['W2']))
            currentWeight.append(np.array(msg['b2']))
            currentWeight.append(np.array(msg['W3']))
            currentWeight.append(np.array(msg['b3']))

            self.aggregate += np.array(currentWeight, dtype=object)
            self.responses += 1

            if self.responses == self.k and self.iterTime < self.totalTime:
                self.aggregate = np.array(self.aggregate) / self.k - self.y
                self.finalAgregate.append(self.aggregate)

                msg = {
                    'W1': self.aggregate[0].tolist(),
                    'b1': self.aggregate[1].tolist(),
                    'W2': self.aggregate[2].tolist(),
                    'b2': self.aggregate[3].tolist(),
                    'W3': self.aggregate[4].tolist(),
                    'b3': self.aggregate[5].tolist()
                }

                self.iterTime += 1
                self.responses = 0                              #reset some variable
                self.aggregate = 0

                if self.iterTime == self.totalTime:
                    print('iterTime:', self.iterTime)
                    endTime = time.clock()
                    print('the final time：', endTime - startTime)

                    for gradAggerate in self.finalAgregate:
                        self.model.set_weights(gradAggerate)
                        # print('testData:', self.test_data, 'testLabels:', self.test_labels)
                        score = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
                        self.acc.append(score[1])
                        self.loss.append(score[0])

                    dataframe = pd.DataFrame({'acc': self.acc, 'loss': self.loss})  # save data
                    dataframe.to_csv('80%mnist.csv', index=False, sep=',')  # 40%, 60%, 80%

                    plt.figure(figsize=(6, 6))
                    xAxis = []
                    for i in range(self.totalTime):
                        xAxis.append(i)

                    plt.plot(xAxis, self.acc[0:self.totalTime])
                    plt.plot(xAxis, self.acc[0:self.totalTime], 'ro')
                    plt.grid(True)
                    # plt.legend(loc=0)  # 图例位置自动
                    plt.axis('tight')
                    plt.xlabel('iterTimes')
                    plt.ylabel('accuracy')

                    plt.figure(figsize=(6, 6))
                    plt.plot(xAxis, self.loss[0:self.totalTime])
                    plt.plot(xAxis, self.loss[0:self.totalTime], 'ro')
                    plt.grid(True)
                    # plt.legend(loc=0)  # 图例位置自动
                    plt.axis('tight')
                    plt.xlabel('iterTimes')
                    plt.ylabel('loss')

                    plt.show()
                else:
                    print('--------------------distribute weights to poxys-------------------')
                    print('-------------------', self.iterTime, '--------------------------- ')
                    for sid in self.ready_client_ids:
                        emit('serverToPoxyGrad', msg, room=sid)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)



startTime = 0
if __name__=="__main__":
    startTime = time.clock()
    server = secaggserver("192.168.1.223", 2018, 2, 2)
    print("listening on 192.168.1.181:2018")
    server.start()