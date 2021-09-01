# import gevent
# from gevent import monkey
# monkey.patch_all()
from socketIO_client import SocketIO, LoggingNamespace
from random import randrange
import numpy as np
from copy import deepcopy
import codecs
import pickle
import json
import os, gzip
import math
import time
import threading
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import *
from tensorflow.keras.layers import Dense, Dropout, Flatten
import pandas as pd


class MlpMode:
    def load_mnistdata(self, data_folder):  
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

        x_train = x_train.astype('float') / 255 
        x_test = x_test.astype('float') / 255  

        y_train = to_categorical(y_train, 10)  
        y_test = to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)

    def load_fatiguedata(self, data_folder):  # load fatigue
        files = [
            'fatiguedata.csv', 'fatiguelabel.csv',
        ]

        fileData = data_folder + '//' + files[0]
        fileLabel = data_folder + '//' + files[1]
        fatiguedata = pd.read_csv(fileData, header=1)  
        fatiguelabel = pd.read_csv(fileLabel, header=1)  
        fatigutsize = np.array(fatiguedata).shape

        x_train = fatiguedata[1:300]  # data
        x_test = fatiguedata[300:fatigutsize[0]]

        y_train = fatiguelabel[1:300]  # label
        y_test = fatiguelabel[300:fatigutsize[0]]

        x_train = x_train.astype('float') / 255  
        x_test = x_test.astype('float') / 255  

        y_train = to_categorical(y_train - 1, 4)  
        y_test = to_categorical(y_test - 1, 4)

        return (x_train, y_train), (x_test, y_test)

    def constructModel(self, train_images):  
        self.model = tf.keras.Sequential()
        self.model.add(Flatten(input_shape=train_images.shape[1:]))
        # now: model.output_shape==(None,65536)
        # self.model.add(Dense(64, activation='relu'))  
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

    def trainMlp(self, weights, train_images, train_labels, batch_size, epoch):
        self.modelTrain(train_images, train_labels, weights, batch_size, epoch) 
        weights = self.model.weights
        return weights

    def modelTrain(self, weights, train_images, train_labels, batch_size, epochs):  
        self.model.set_weights(weights)  

        hist = self.model.fit(train_images, train_labels, batch_size, epochs, validation_split=0.2, verbose=1, shuffle=True)

        currentWeights = self.model.get_weights()

        return currentWeights

    def modelEvaluate(self, test_images, test_labels): 
        score = self.model.evaluate(test_images, test_labels, verbose=0)  
        accuracy = 100 * score[1]
        print('Test accuracy: %.4f%%' % accuracy)

class SecAggregator:
    def __init__(self,common_base,common_mod, mixData, poxyMixData):
        self.secretkey = randrange(common_mod)
        self.base = common_base
        self.mod = common_mod
        self.pubkey = (self.base**self.secretkey) % self.mod
        self.sndkey = randrange(common_mod)
        self.keys = {}
        self.id = ''
        self.mlp = MlpMode()
        self.mixData = mixData
        self.poxyMixData = poxyMixData
        self.othersShare = dict()
        self.batch_size = 100
        self.epoch = 1
        self.weights = []
        self.y = 5
        self.train_images = []
        self.train_labels = []
        self.othersShareCount = 0

    def trainClient(self, weights, train_images, train_labels):   
        weights = self.mlp.modelTrain(weights, train_images, train_labels, self.batch_size, self.epoch)

        weightsToArray = np.array(weights, dtype=object)

        # finalWeight = (weightsToArray + self.poxyMixData + self.mixData).tolist()
        finalWeight = (weightsToArray + self.mixData + self.y).tolist()

        return finalWeight

    def reedSolomnCode(self, mixData, clientNum):      # RScode
        print('mixData:', mixData, 'clientNum:', clientNum)
        originalData = self.getData(mixData, clientNum)  

        print('original data:', originalData)

        rowDim = clientNum - 1  
        colDim = clientNum

        vanderSeed = [math.pow(2, i) for i in range(colDim)]
        reedSolomnVanderData = np.concatenate((np.identity(clientNum), np.vander(vanderSeed, rowDim).transpose()[::-1]),
                                              axis=0)                 #get vanderdata
        print('reedSolomnVanderData:', reedSolomnVanderData)

        reedSolomnCodedData = reedSolomnVanderData.dot(originalData)  #get rscoded data
        print('reedSolomnCodedData:', reedSolomnCodedData)

        codedData = reedSolomnCodedData[clientNum:clientNum + clientNum - 1]

        return reedSolomnCodedData, reedSolomnVanderData, codedData

    def getData(self, mixData, clientNum):
        originalData = []
        averageData = mixData / clientNum
        leftAverage = mixData

        for i in range(clientNum):
            if i == 0:
                originalData.append(averageData)
                leftAverage -= averageData
            elif i == clientNum - 1:
                originalData.append(leftAverage)
            else:
                if i % 2 == 0:
                    originalData.append(averageData)
                    leftAverage -= averageData
                else:
                    originalData.append(0)
        return originalData

class secaggclient:
    def __init__(self, serverhost, serverport, mixData, poxyData):
        self.sio = SocketIO(serverhost, serverport, LoggingNamespace)
        self.aggregator = SecAggregator(3, 100103, mixData, poxyData)
        self.id = ''
        self.keys = {}
        self.clientsId = []
        self.iterTimes = 0
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.b1 = []
        self.b2 = []
        self.b3 = []
        self.pos = 0
        self.w1FragNum = 0
        self.w2FragNum = 0
        self.gradResponse = 0
        self.everySendFrag = 15     # 15*512*8
        self.iterSendNum = 0

    def start(self):
        (train_images, train_labels), (test_images, test_labels) = self.aggregator.mlp.load_mnistdata('MINST')
        self.train_images = train_images[20000:23999]
        self.train_labels = train_labels[20000:23999]
        # (train_images, train_labels), (test_data, test_labels) = self.aggregator.mlp.load_fatiguedata('fatiguedata')
        # self.train_images = train_images[1:100]
        # self.train_labels = train_labels[1:100]
        self.aggregator.mlp.constructModel(self.train_images)  
        self.register_handles()
        print("Starting")
        self.sio.emit("wakeup")
        self.sio.wait()

    def codeToClient(self):
        print('send rscode to poxy')
        encodeTimeStart = time.clock()
        reedSolomnData, vanderCheckArray, vanderData = self.aggregator.reedSolomnCode(self.aggregator.mixData, len(self.clientsId))
        encodeTimeEnd = time.clock()
        print('encode time:', encodeTimeEnd - encodeTimeStart)

        msg = {
            'reedSolomnData': reedSolomnData.tolist(),
            'vanderData': vanderData.tolist(),
            'vanderCheckArray': vanderCheckArray.tolist(),
        }

        self.sio.emit("clientToPoxyRsCode", msg)

    def threadTrain(self, currentWeight, train_images, train_labels):
        weight = self.aggregator.trainClient(currentWeight, train_images, train_labels)
        return weight


    def register_handles(self):
        def on_connect(*args):
            msg = args[0]
            self.sio.emit("connect")
            print("Connected and recieved this message", msg['message'])

        def on_poxytoclient_Id(*args):
            msg = args[0]
            self.clientsId.clear()
            self.clientsId = msg['clientIds']
            self.aggregator.mixData = msg['mixData']
            print('Get ClientId from poxy')
            self.codeToClient()

        def on_poxytoclient_newId(*args):
            msg = args[0]
            self.clientsId.clear()
            self.clientsId = msg['clientIds']
            self.aggregator.mixData = msg['mixData']
            print('Get new clientIds from poxy')
            self.codeToClient()

        def on_gradW1Frag(*args):                          #---rev grad w1,b1,w2,b2,w3,b3------------#
            msg = args[0]
            self.w1.extend(np.array(msg['W1']))
            print('rcv w1 frag from poxy, current w1 shape:', np.array(self.w1).shape)
            self.sio.emit('gradFrag')

        def on_gradW2Frag(*args):
            msg = args[0]
            self.w2.extend(np.array(msg['W2']))
            print('rcv w2 frag from poxy', np.array(self.w2).shape)
            self.sio.emit('gradFrag')

        def on_gradW3Frag(*args):
            print('rcv w3 frag from poxy')
            msg = args[0]
            self.w3 = np.array(msg['W3'])
            self.sio.emit('gradFrag')

        def on_gradB1Frag(*args):
            print('rcv b1 frag from poxy')
            msg = args[0]
            self.b1 = np.array(msg['b1'])
            self.sio.emit('gradFrag')

        def on_gradB2Frag(*args):
            print('rcv b2 frag from poxy')
            msg = args[0]
            self.b2 = np.array(msg['b2'])
            self.sio.emit('gradFrag')

        def on_gradB3Frag(*args):
            print('rcv b3 frag from poxy')
            msg = args[0]
            self.b3 = np.array(msg['b3'])
            self.sio.emit('gradFrag')

        def sendGrad():
            self.curFragSize = np.array(self.aggregator.weights[0]).shape
            print('shape:', self.curFragSize)
            self.iterSendNum = self.curFragSize[0] // self.everySendFrag

            startIndex = 0
            endIndex = self.everySendFrag

            sendFrag = self.aggregator.weights[0][startIndex:endIndex]

            msg = {
                'W1': np.array(sendFrag).tolist(),
            }
            print('iterTime:', self.iterSendNum)
            print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)
            self.sio.emit('gradW1Frag', msg)

        def on_askGrad(*args):       #simulate absent, poxy ask client to send grad
            print('ask grad from client')
            sendGrad()

        def on_gradFinish(*args):
            print('grad rcv finished')
            currentWeight = []
            currentWeight.append(np.array(self.w1))
            currentWeight.append(np.array(self.b1))
            currentWeight.append(np.array(self.w2))
            currentWeight.append(np.array(self.b2))
            currentWeight.append(np.array(self.w3))
            currentWeight.append(np.array(self.b3))

            self.aggregator.weights = self.aggregator.trainClient(currentWeight, self.train_images, self.train_labels)

            self.w1 = []
            self.w2 = []
            self.w3 = []
            self.b1 = []
            self.b2 = []
            self.b3 = []

            self.iterTimes += 1

            if self.iterTimes > 1:
                print('continueClientToPoxyGrad')                 #continue to train
                sendGrad()
            else:
                print('simulateAbsentClientToPoxyGrad')            #simulate absent
                self.sio.emit("simulateAbsentClientToPoxyGrad")

        def on_gradFrag(*args):                              #---send grad msg------------#
            print('rcv poxy gradFrag request')
            if self.pos == 0:
                self.w1FragNum += 1
                if self.w1FragNum < self.iterSendNum:
                    startIndex = self.everySendFrag * self.w1FragNum
                    endIndex = self.everySendFrag * (self.w1FragNum + 1)
                    print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)  # -----W1----
                    sendFrag = self.aggregator.weights[0][startIndex:endIndex]
                    msg = {
                        'W1': np.array(sendFrag).tolist(),
                    }
                    self.sio.emit('gradW1Frag', msg)
                elif self.w1FragNum == self.iterSendNum:
                    startIndex = self.everySendFrag * self.w1FragNum
                    endIndex = self.curFragSize[0]
                    print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)
                    sendFrag = self.aggregator.weights[0][startIndex:endIndex]
                    msg = {
                        'W1': np.array(sendFrag).tolist(),
                    }
                    self.sio.emit('gradW1Frag', msg)
                elif self.w1FragNum > self.iterSendNum:
                    print('send b1')  # -----b1----
                    msg = {
                        'b1': np.array(self.aggregator.weights[1]).tolist(),
                    }
                    self.sio.emit('gradB1Frag', msg)
                    self.pos += 1
            elif self.pos == 1:
                self.curFragSize = np.array(self.aggregator.weights[2]).shape  # -----w2----
                self.iterSendNum = self.curFragSize[0] // self.everySendFrag
                startIndex = 0
                endIndex = self.everySendFrag
                print('itertime:', self.iterSendNum)
                print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                sendFrag = self.aggregator.weights[2][startIndex:endIndex]
                msg = {
                    'W2': np.array(sendFrag).tolist(),
                }
                self.sio.emit('gradW2Frag', msg)
                self.pos += 1
            elif self.pos == 2:
                self.w2FragNum += 1
                if self.w2FragNum < self.iterSendNum:  # -----w2----
                    startIndex = self.everySendFrag * self.w2FragNum
                    endIndex = self.everySendFrag * (self.w2FragNum + 1)
                    sendFrag = self.aggregator.weights[2][startIndex:endIndex]
                    msg = {
                        'W2': np.array(sendFrag).tolist(),
                    }
                    print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                    self.sio.emit('gradW2Frag', msg)
                elif self.w2FragNum == self.iterSendNum:
                    startIndex = self.everySendFrag * self.w2FragNum
                    endIndex = self.curFragSize[0]
                    sendFrag = self.aggregator.weights[2][startIndex:endIndex]
                    msg = {
                        'W2': np.array(sendFrag).tolist(),
                    }
                    print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                    self.sio.emit('gradW2Frag', msg)
                elif self.w2FragNum > self.iterSendNum:  # -----b2----
                    msg = {
                        'b2': np.array(self.aggregator.weights[3]).tolist(),
                    }
                    print('send b2')
                    self.sio.emit('gradB2Frag', msg)
                    self.pos += 1
            elif self.pos == 3:  # -----W3----
                msg = {
                    'W3': np.array(self.aggregator.weights[4]).tolist(),
                }
                print('send W3')
                self.sio.emit('gradW3Frag', msg)
                self.pos += 1
            elif self.pos == 4:
                msg = {
                    'b3': np.array(self.aggregator.weights[5]).tolist(),  # -----b3----
                }
                print('send b3')
                self.sio.emit('gradB3Frag', msg)
                self.pos += 1
            elif self.pos == 5:
                self.pos = 0
                self.w1FragNum = 0
                self.w2FragNum = 0
                print('send gradFinish')
                self.sio.emit('gradFinish')

        def on_poxytoclient_grad(*args):
            msg = args[0]
            print('Get grad from poxy, machine learning starting')

            currentWeight = []
            currentWeight.append(np.array(msg['W1']))
            currentWeight.append(np.array(msg['b1']))
            currentWeight.append(np.array(msg['W2']))
            currentWeight.append(np.array(msg['b2']))
            currentWeight.append(np.array(msg['W3']))
            currentWeight.append(np.array(msg['b3']))

            # self.aggregator.weights = self.aggregator.trainClient(currentWeight, self.train_images, self.train_labels)

            msg = {
                'W1': self.aggregator.weights[0].tolist(),
                'b1': self.aggregator.weights[1].tolist(),
                'W2': self.aggregator.weights[2].tolist(),
                'b2': self.aggregator.weights[3].tolist(),
                'W3': self.aggregator.weights[4].tolist(),
                'b3': self.aggregator.weights[5].tolist()
            }

            self.iterTimes += 1

            if self.iterTimes > 1:
                print('continueClientToPoxyGrad')
                self.sio.emit("continueClientToPoxyGrad", msg)
                # self.sio.emit("continueClientToPoxyGrad", 1)
            else:
                print('simulateAbsentClientToPoxyGrad')
                self.sio.emit("simulateAbsentClientToPoxyGrad", msg)
                # self.sio.emit("simulateAbsentClientToPoxyGrad", 1)

        def on_poxyrequire_share(*args):
            msg = args[0]
            print('poxy requires share of other client', msg)

            absentClientId = msg['absentClientIds']
            absentClientMixShares = dict()
            # print('absentId:', absentClientId)
            for clientId in absentClientId:
                print('clientId:', clientId)
                absentClientMixShares[clientId] = self.aggregator.othersShare[clientId]

            self.sio.emit("absentShare", absentClientMixShares)

        def on_others_share(*args):
            msg = args[0]
            print('Save the shares of other clients', msg)

            self.aggregator.othersShare[msg['sid']] = msg['value']

            self.aggregator.othersShareCount += 1

            if self.aggregator.othersShareCount == len(self.clientsId):
                if self.iterTimes > 0:
                    print('clientNewIdAndRsFinish')
                    self.sio.emit("clientNewIdAndRsFinish")
                else:
                    print('clientIdAndRsFinish')
                    self.sio.emit("clientIdAndRsFinish")
                self.aggregator.othersShareCount = 0

        def on_disconnect(*args):
            print("Disconnected")
            self.sio.emit("disconnect")


        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('late', on_disconnect)
        self.sio.on('poxyToClientGrad', on_poxytoclient_grad)
        self.sio.on('poxyRequireShare', on_poxyrequire_share)
        self.sio.on('othersClientShare', on_others_share)
        self.sio.on('poxyToClientId', on_poxytoclient_Id)
        self.sio.on('poxyToClientNewId', on_poxytoclient_newId)
        self.sio.on('gradW1Frag', on_gradW1Frag)
        self.sio.on('gradW2Frag', on_gradW2Frag)
        self.sio.on('gradW3Frag', on_gradW3Frag)
        self.sio.on('gradB1Frag', on_gradB1Frag)
        self.sio.on('gradB2Frag', on_gradB2Frag)
        self.sio.on('gradB3Frag', on_gradB3Frag)
        self.sio.on('gradFinish', on_gradFinish)
        self.sio.on('gradClientFrag', on_gradFrag)
        self.sio.on('absentAskGrad', on_askGrad)


if __name__=="__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')                           #for GPU
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #
    # print('physical_devices:', physical_devices)
    print('client1_1')

    s = secaggclient("127.0.0.1", 2020, -3, 3) #mixData:-3, -7, 10, poxyData:-3, 3
    s.start()
    print("Ready")