# import gevent
# from gevent import monkey
# monkey.patch_all()
from flask import *
import flask_socketio
import socketIO_client
from socketIO_client import SocketIO, LoggingNamespace
from flask_socketio import *
import json
import codecs
import pickle
import numpy as np
import threading
import time
import copy


class secaggserver:
    def __init__(self, host, port, n, k):
        self.n = n
        self.k = k
        self.aggregate = 0
        self.host = host
        self.port = port
        self.responses = 0
        self.respset = set()
        self.resplist = []
        self.lef_client_ids = []
        self.ready_client_ids = set()
        self.app = Flask(__name__)
        self.socketClientio = socketIO_client.SocketIO("192.168.1.223", 2018, LoggingNamespace)
        self.socketServerio = flask_socketio.SocketIO(self.app)
        self.register_Server_handles()
        self.register_Client_handles()
        self.recovers = 0
        self.clientsVanderData = dict()
        self.clientsVanderCheck = dict()
        self.clientId = dict()
        self.numkeys = 0
        self.randReed = 35
        self.clientReadyNum = 0
        self.clientOnline = 0
        self.revGradFlag = False
        self.pos = 0
        self.w1FragNum = 0
        self.w2FragNum = 0
        self.gradResponse = 0
        self.gradW1Response = 0
        self.gradW2Response = 0
        self.gradW3Response = 0
        self.gradB1Response = 0
        self.gradB2Response = 0
        self.gradB3Response = 0
        self.everySendFrag = 15  # 15*512*8
        self.w1 = dict()
        self.w2 = dict()
        self.w3 = dict()
        self.b1 = dict()
        self.b2 = dict()
        self.b3 = dict()
        self.encodeTimeStart = 0
        self.encodeTimeEnd = 0
        self.decodeTimeStart = 0
        self.decodeTimeEnd = 0


    def getRandomList(self, data, num):
        finalY = []
        if data == 0:
            for i in range(1, num // 2 + 1):
                finalY.append(i)
                finalY.append(-i)
            if num % 2 != 0:
                finalY.append(0)
        else:
            if data > 0:
                sign = 1
            else:
                sign = -1

            currentData = data * sign
            y0 = np.random.randint(currentData, size=num - 1)
            ratio = sum(y0) / currentData
            # print("y0:%d, ratio：%d", y0, ratio)

            y1 = y0 // ratio
            y1 = y1.tolist()
            y1.append(currentData - sum(y1))
            finalY = [x * sign for x in y1]

        return finalY

    def getMixData(self, mixData, mixNum):
        mixArray = []


        p1 = self.getRandomList(mixData, mixNum)
        p2 = self.getRandomList(mixData, mixNum)
        p2.reverse()

        for i in range(len(p1)):
            mixArray.append(p1[i] - p2[i])

        return mixArray

    def recoverMixData(self, absendId, sid, msg):
        lossData = []
        vanderData = []
        # print('recoverData:', msg)
        lossData.append(msg)
        lossData.extend(self.clientsVanderData[absendId])
        pos = self.clientId[sid]
        # print('absentId:', absendId)
        # print('pos:', pos)
        # print('clientsVanderCheck:', self.clientsVanderCheck)
        vanderData.append(self.clientsVanderCheck[absendId][pos])
        vanderData.extend(
            self.clientsVanderCheck[absendId][len(self.clientId):len(self.clientId) + len(self.clientId) - 1])
        # print('lossData:', lossData)
        # print('vanderData:', vanderData)

        recover_data = np.linalg.inv(vanderData).dot(lossData)
        # print('recover_data:', recover_data)

        return sum(recover_data)

    def sendGrad(self):
        self.curFragSize = np.array(self.aggregate[0]).shape
        self.iterSendNum = self.curFragSize[0] // self.everySendFrag

        startIndex = 0
        endIndex = self.everySendFrag
        sendFrag = self.aggregate[0][startIndex:endIndex]

        msg = {
            'W1': np.array(sendFrag).tolist(),
        }
        print('iterTime:', self.iterSendNum)

        print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)
        for rid in self.lef_client_ids:
            self.socketServerio.emit('gradW1Frag', msg, room=rid)

    def register_Client_handles(self):
        def on_serverToPoxyGrad(*args):
            print('poxy receives grads from server and distriutes grad to clients')
            msg = args[0]

            currentWeight = []
            currentWeight.append(np.array(msg['W1']))
            currentWeight.append(np.array(msg['b1']))
            currentWeight.append(np.array(msg['W2']))
            currentWeight.append(np.array(msg['b2']))
            currentWeight.append(np.array(msg['W3']))
            currentWeight.append(np.array(msg['b3']))

            self.aggregate = currentWeight                      #can't send bytes larger than 64k

            self.sendGrad()                                     #need to check the accuracy

        self.socketClientio.on('serverToPoxyGrad', on_serverToPoxyGrad)

    def register_Server_handles(self):
        @self.socketServerio.on("wakeup")
        def handle_wakeup():
            print("Recieved wakeup from", request.sid)

            self.numkeys += 1

            if self.numkeys == self.n:
                ready_clients = list(self.ready_client_ids)
                self.lef_client_ids = copy.deepcopy(ready_clients)

                mix = self.getMixData(self.randReed, self.n)
                print('send clientsId to clients', self.n, mix)

                pos = 0
                self.encodeTimeStart = time.clock()
                for rid in self.lef_client_ids:
                    msg = {
                        'mixData': mix[pos],
                        'clientIds': ready_clients
                    }
                    pos += 1
                    emit('poxyToClientId', msg, room=rid)
                self.numkeys = 0

        @self.socketServerio.on("clientIdAndRsFinish")
        def handle_clientIdFinish():
            print(request.sid, 'received clientId')
            self.clientReadyNum += 1

            if (self.clientReadyNum == self.n):
                self.encodeTimeEnd = time.clock()
                print('encode finalTime:', self.encodeTimeEnd - self.encodeTimeStart)
                self.socketClientio.emit('poxyReady')      #clients are ready, let server sends grad to poxys
                self.clientReadyNum = 0

        @self.socketServerio.on("clientNewIdAndRsFinish")
        def handle_clientNewIdFinish():
            print(request.sid, 'received new clientId and get rs share')
            self.clientReadyNum += 1
            if (self.clientReadyNum == len(self.resplist)):
                for rid in self.resplist:                        #ask left clients to send their grad
                    print('ask grad from client:', rid)
                    emit('absentAskGrad', room=rid)

                self.clientReadyNum = 0

        @self.socketServerio.on("connect")
        def handle_connect():
            print(request.sid, " Connected")
            self.ready_client_ids.add(request.sid)
            self.respset.add(request.sid)
            print('Connected devices:', self.ready_client_ids)

        @self.socketServerio.on('disconnect')
        def handle_disconnect():
            print(request.sid, " Disconnected")
            if request.sid in self.ready_client_ids:
                self.ready_client_ids.remove(request.sid)
            print(self.ready_client_ids)

        @self.socketServerio.on('absentShare')      #need only one share can recover mixData
        def on_othersSecret(*args):
            msg = args[0]
            print('receive absentMixs of ', request.sid, msg, self.respset)    #need modify

            decodeStart = time.clock()
            for absendId in self.respset:
                recoverMix = self.recoverMixData(absendId, request.sid, msg[absendId])                    #recover mixData
                print('recoverMix', recoverMix)
                self.recovers += 1
                self.aggregate += recoverMix
                print('recover mix of', absendId)

            if self.recovers == len(self.respset):
                self.decodeTimeEnd = time.clock()
                print('decodeTimeEnd:', self.decodeTimeEnd)
                print('decode comm time:', decodeStart - self.decodeTimeStart)
                print('decode time:', self.decodeTimeEnd - decodeStart)
                print('decode finalTime:', self.decodeTimeEnd - self.decodeTimeStart)

                if self.k == 1:
                    mix = 0
                else:
                    mix = self.getMixData(self.randReed, self.k)
                print('send new clientsId to clients:', self.k, mix)     #after recover data, emit new clientIds

                pos = 0
                for rid in self.resplist:
                    if len(self.resplist) == 1:
                        msg = {
                            'mixData': mix,
                            'clientIds': list(self.resplist)
                        }
                    else:
                        msg = {
                            'mixData': mix[pos],
                            'clientIds': list(self.resplist)
                        }
                        pos += 1

                    emit('poxyToClientNewId', msg, room=rid)

        @self.socketServerio.on('simulateAbsentClientToPoxyGrad')
        def on_simulateAbsentClientToPoxyGrad_agg(*args):
            if self.n != self.k:
                if self.responses < self.k:
                    print('poxy receives grad from', request.sid, ':simulate absent')
                    self.responses += 1
                    self.respset.remove(request.sid)
                    self.resplist.append(request.sid)
                else:
                    print(request.sid, ':absent')
                    self.socketServerio.emit('late', {
                        'msg': "Hey I'm server"
                    }, room=request.sid)
                    self.responses += 1

                if self.responses == self.k:
                    print("k WIGHTS RECIEVED. BEGINNING AGGREGATION PROCESS.", self.resplist, self.ready_client_ids)
                    respsetList = list(self.respset)
                    self.lef_client_ids = copy.deepcopy(self.resplist)

                    msg = {
                        'absentClientIds': respsetList
                    }
                    print('require share for', self.resplist[0])
                    self.decodeTimeStart = time.clock()
                    print('decode startTime:', self.decodeTimeStart)
                    emit('poxyRequireShare', msg, room=self.resplist[0])    #only need to ask one client to send share to poxy to recover mix data
            else:
                self.clientOnline += 1
                if self.clientOnline == self.n:
                    print('no clients dropout')
                    self.lef_client_ids = copy.deepcopy(self.ready_client_ids)
                    self.aggregate = 0
                    for rid in self.lef_client_ids:                             #ask left clients to send their grad
                        print('ask grad from client:', rid)
                        emit('absentAskGrad', room=rid)
                    self.clientOnline = 0

        @self.socketServerio.on('gradFrag')
        def handle_gradFrag(*args):
            print('rcv gradFrag request from:', request.sid)
            self.gradResponse += 1
            if self.gradResponse == len(self.lef_client_ids):
                if self.pos == 0:
                    self.w1FragNum += 1
                    if self.w1FragNum < self.iterSendNum:
                        startIndex = self.everySendFrag * self.w1FragNum
                        endIndex = self.everySendFrag * (self.w1FragNum + 1)
                        print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)  # -----W1----
                        sendFrag = self.aggregate[0][startIndex:endIndex]
                        msg = {
                            'W1': sendFrag.tolist(),
                        }
                        for rid in self.lef_client_ids:
                            emit('gradW1Frag', msg, room=rid)
                    elif self.w1FragNum == self.iterSendNum:
                        startIndex = self.everySendFrag * self.w1FragNum
                        endIndex = self.curFragSize[0]
                        print('send W1-', self.w1FragNum, '-', startIndex, '-', endIndex)
                        sendFrag = self.aggregate[0][startIndex:endIndex]
                        msg = {
                            'W1': sendFrag.tolist(),
                        }
                        for rid in self.lef_client_ids:
                            emit('gradW1Frag', msg, room=rid)
                    elif self.w1FragNum > self.iterSendNum:
                        print('send b1')  # -----b1----
                        msg = {
                            'b1': self.aggregate[1].tolist(),
                        }
                        for rid in self.lef_client_ids:
                            emit('gradB1Frag', msg, room=rid)
                        self.pos += 1
                elif self.pos == 1:
                    self.curFragSize = np.array(self.aggregate[2]).shape  # -----w2----
                    self.iterSendNum = self.curFragSize[0] // self.everySendFrag
                    startIndex = 0
                    endIndex = self.everySendFrag
                    print('itertime:', self.iterSendNum)
                    print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                    sendFrag = self.aggregate[2][startIndex:endIndex]
                    msg = {
                        'W2': sendFrag.tolist(),
                    }
                    for rid in self.lef_client_ids:
                        emit('gradW2Frag', msg, room=rid)
                    self.pos += 1
                elif self.pos == 2:
                    self.w2FragNum += 1
                    if self.w2FragNum < self.iterSendNum:  # -----w2----
                        startIndex = self.everySendFrag * self.w2FragNum
                        endIndex = self.everySendFrag * (self.w2FragNum + 1)
                        sendFrag = self.aggregate[2][startIndex:endIndex]
                        msg = {
                            'W2': sendFrag.tolist(),
                        }
                        print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                        for rid in self.lef_client_ids:
                            emit('gradW2Frag', msg, room=rid)
                    elif self.w2FragNum == self.iterSendNum:
                        startIndex = self.everySendFrag * self.w2FragNum
                        endIndex = self.curFragSize[0]
                        sendFrag = self.aggregate[2][startIndex:endIndex]
                        msg = {
                            'W2': sendFrag.tolist(),
                        }
                        print('send W2-', self.w2FragNum, '-', startIndex, '-', endIndex)
                        for rid in self.lef_client_ids:
                            emit('gradW2Frag', msg, room=rid)
                    elif self.w2FragNum > self.iterSendNum:  # -----b2----
                        msg = {
                            'b2': self.aggregate[3].tolist(),
                        }
                        print('send b2')
                        for rid in self.lef_client_ids:
                            emit('gradB2Frag', msg, room=rid)
                        self.pos += 1
                elif self.pos == 3:  # -----W3----
                    msg = {
                        'W3': self.aggregate[4].tolist(),
                    }
                    print('send W3')
                    for rid in self.lef_client_ids:
                        emit('gradW3Frag', msg, room=rid)
                    self.pos += 1
                elif self.pos == 4:
                    msg = {
                        'b3': self.aggregate[5].tolist(),  # -----b3----
                    }
                    print('send b3')
                    for rid in self.lef_client_ids:
                        emit('gradB3Frag', msg, room=rid)
                    self.pos += 1
                elif self.pos == 5:
                    self.pos = 0
                    self.w1FragNum = 0
                    self.w2FragNum = 0
                    print('send gradFinish')
                    for rid in self.lef_client_ids:
                        self.w1[rid] = []
                        self.w2[rid] = []
                        self.w3[rid] = []
                        self.b1[rid] = []
                        self.b2[rid] = []
                        self.b3[rid] = []
                        emit('gradFinish', room=rid)
                    self.gradResponse = 0
                    self.aggregate = 0
                self.gradResponse = 0

        @self.socketServerio.on('gradW1Frag')
        def handle_gradW1Frag(*args):
            self.gradW1Response += 1
            msg = args[0]
            self.w1[request.sid].extend(np.array(msg['W1']))
            print('rcv w1 frag from client', request.sid, 'w1 shape:', np.array(self.w1[request.sid]).shape)
            if self.gradW1Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradW1Response = 0

        @self.socketServerio.on('gradW2Frag')
        def handle_gradW2Frag(*args):
            self.gradW2Response += 1
            msg = args[0]
            self.w2[request.sid].extend(np.array(msg['W2']))
            print('rcv w2 frag from client', request.sid, 'w2 shape:', np.array(self.w2[request.sid]).shape)
            if self.gradW2Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradW2Response = 0

        @self.socketServerio.on('gradW3Frag')
        def handle_gradW3Frag(*args):
            print('rcv w3 frag from client', request.sid)
            self.gradW3Response += 1
            msg = args[0]
            self.w3[request.sid] = np.array(msg['W3'])
            if self.gradW3Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradW3Response = 0

        @self.socketServerio.on('gradB1Frag')
        def handle_gradB1Frag(*args):
            print('rcv b1 frag from client', request.sid)
            self.gradB1Response += 1
            msg = args[0]
            self.b1[request.sid] = np.array(msg['b1'])
            if self.gradB1Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradB1Response = 0

        @self.socketServerio.on('gradB2Frag')
        def handle_gradB2Frag(*args):
            print('rcv b2 frag from client', request.sid)
            self.gradB2Response += 1
            msg = args[0]
            self.b2[request.sid] = np.array(msg['b2'])
            if self.gradB2Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradB2Response = 0

        @self.socketServerio.on('gradB3Frag')
        def on_gradB3Frag(*args):
            print('rcv b3 frag from client', request.sid)
            self.gradB3Response += 1
            msg = args[0]
            self.b3[request.sid] = np.array(msg['b3'])
            if self.gradB3Response == len(self.lef_client_ids):
                for rid in self.lef_client_ids:
                    emit('gradClientFrag', room=rid)
                self.gradB3Response = 0

        @self.socketServerio.on('gradFinish')
        def handle_gradFinish(*args):
            print('weights finish', request.sid)
            self.gradResponse += 1
            if self.gradResponse == len(self.lef_client_ids):
                self.gradResponse = 0

                # print('aggerate:', self.aggregate)

                for rid in self.lef_client_ids:
                    currentWeight = []
                    currentWeight.append(np.array(self.w1[rid]))
                    currentWeight.append(np.array(self.b1[rid]))
                    currentWeight.append(np.array(self.w2[rid]))
                    currentWeight.append(np.array(self.b2[rid]))
                    currentWeight.append(np.array(self.w3[rid]))
                    currentWeight.append(np.array(self.b3[rid]))
                    self.aggregate += np.array(currentWeight)

                self.aggregate = self.aggregate / len(self.lef_client_ids)

                msg = {
                    "W1": self.aggregate[0].tolist(),
                    "b1": self.aggregate[1].tolist(),
                    "W2": self.aggregate[2].tolist(),
                    "b2": self.aggregate[3].tolist(),
                    "W3": self.aggregate[4].tolist(),
                    "b3": self.aggregate[5].tolist()
                }

                self.socketClientio.emit('poxyToServerGrad', msg)      #send grad to server

                self.responses = 0
                self.aggregate = 0

        @self.socketServerio.on('clientToPoxyRsCode')
        def on_clienttopoxy_vander(*args):
            print('receiver vander and mixShres from clients')
            msg = args[0]   #   msg = {
                                    #     'reedSolomnData': reedSolomnData.tolist(),
                                    #     'vanderData': vanderData.tolist(),
                                    #     'vanderCheckArray': vanderCheckArray.tolist(),
                                    # }

            self.clientsVanderData[request.sid] = msg['vanderData']
            self.clientsVanderCheck[request.sid] = msg['vanderCheckArray']
            print('send these shares to other clients')

            reedSolomnData = np.array(msg['reedSolomnData'])

            print('reedSolomnData:', reedSolomnData)

            pos = 0
            for sid in self.lef_client_ids:
                self.clientId[sid] = pos

                msg = {
                    'sid': request.sid,
                    'value': reedSolomnData[pos]
                }
                emit('othersClientShare', msg, room=sid)
                pos += 1


    def _receive_events_thread(self):
        self.socketClientio.wait()

    def start(self):
        self.socketClientio.emit("wakeup")
        self.receive_events_thread = threading.Thread(target=self._receive_events_thread)
        self.receive_events_thread.daemon = True
        self.receive_events_thread.start()

        self.socketServerio.async_mode = threading
        self.socketServerio.run(self.app, host=self.host, port=self.port)


if __name__=="__main__":
    server = secaggserver("127.0.0.1", 2020, 5, 1)
    print("listening on 127.0.0.1:2020")
    server.start()