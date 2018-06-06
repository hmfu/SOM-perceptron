import argparse
import tensorflow as tf
import numpy as np
import random

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

class DataFormater():
    def __init__(self):
        self.ptArr = None
        self.classArr = None
        self.dataNum = None
        self.sameClassArr = None
        self.diffClassArr = None

    def loadPtData(self, fileName):
        ptList = []
        with open(fileName, 'r') as inFile:
            for line in inFile:
                ptList += [[float(x) for x in line[:-1].split('\t')]]
        self.ptArr = np.array(ptList)

    def loadClassData(self, fileName):
        classList = []
        with open(fileName, 'r') as inFile:
            for line in inFile:
                classList += [[int(x) for x in line[:-1].split('\t')]]
        self.classArr = np.array(classList)
        self.dataNum = len(self.classArr)
        

        nodeSetList = []
        for nodeInd in range(self.dataNum):
            if not nodeSetList or (True not in [nodeInd in theSet for theSet in nodeSetList]):
                nodeSetList += [set([ind for ind in range(self.dataNum) if self.classArr[nodeInd, ind] == 1])]
        print (nodeSetList, [len(theSet) for theSet in nodeSetList])
        for theSet in nodeSetList:
            for node in theSet:
                for node2 in range(self.dataNum):
                    if (node2 in theSet and self.classArr[node, node2] == 0) or (node2 not in theSet and self.classArr[node, node2] == 1):
                        print ('the relation between node %d and node %d is wrong!' % (node, node2))
        print ('check done!')


    def buildDiffSameClassArr(self):
        sameClassList = []
        diffClassList = []
        for i in range(self.dataNum):
            for j in range(i, self.dataNum):
                if self.classArr[i, j] == 1:
                    sameClassList += [[i, j]]
                else:
                    diffClassList += [[i, j]]
        self.sameClassArr = np.array(sameClassList)
        self.diffClassArr = np.array(diffClassList)

class SOM_perceptron():
    def __init__(self, ptArr, sameClassArr, diffClassArr, wInitStddev, bInitStddev):
        self.ptArr = ptArr
        self.sameClassArr = sameClassArr
        self.diffClassArr = diffClassArr
        self.wInitStddev = wInitStddev
        self.bInitStddev = bInitStddev
        self.weightList = None
        self.biasList = None
        self.inputPH = None
        self.hiddenPHList = None
        self.lossPHList = None
        self.outputPH = None
        self.ptNum = len(ptArr)
        self.optimizer = None

    def perceptronLayer(self, inPH, outDim):
        inDim = tf.cast(inPH.shape[1], tf.int32)
        w = tf.Variable(tf.random_normal(shape = [inDim, outDim], stddev = self.wInitStddev))
        b = tf.Variable(tf.random_normal(shape = [outDim], stddev = self.bInitStddev))
        self.weightList += [w]
        self.biasList += [b]
        return tf.sigmoid(tf.matmul(inPH, w) + b)

    def buildModel(self, L, nList, EtaAtt, EtaRep):

        self.L = L
        
        if L != len(nList):
            raise Exception('nList length should be equal to L, got %d and %d' % (len(nList), L))

        self.inputPH = tf.placeholder('float32', shape = [None, 2], name = 'inputPH')
        
        self.hiddenPH = self.inputPH
        self.hiddenPHList = []
        self.weightList = []
        self.biasList = []
        for layerInd in range(L):
            self.hiddenPH = self.perceptronLayer(self.hiddenPH, nList[layerInd])
            self.hiddenPHList += [self.hiddenPH]
        self.outputPH = self.hiddenPH

        self.lossPHList = []
        for layerInd in range(L):
            row0 = tf.gather(self.hiddenPHList[layerInd], tf.constant([0]))
            row1 = tf.gather(self.hiddenPHList[layerInd], tf.constant([1]))
            row2 = tf.gather(self.hiddenPHList[layerInd], tf.constant([2]))
            row3 = tf.gather(self.hiddenPHList[layerInd], tf.constant([3]))
            self.lossPHList += [tf.constant(EtaAtt / 2) * tf.reduce_sum(tf.square(row0 - row1)) - tf.constant(EtaRep / 2) * tf.reduce_sum(tf.square(row2 - row3))]

    def trainModel(self, epochNum, evalPerEpochNum, useRandSampEpochNum):
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            for layerInd in range(self.L):
                print ('\nuse random sample epoch number: ', useRandSampEpochNum)
                print ('total epoch num: ', epochNum)
                print ('training layer %d...' % (layerInd))
                inPHValue = self.ptArr if layerInd == 0 else sess.run(self.hiddenPHList[layerInd-1], feed_dict = {self.inputPH: self.ptArr})
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(self.lossPHList[layerInd], var_list = [self.weightList[layerInd], self.biasList[layerInd]])
                for epochInd in range(epochNum):
                    if epochInd % evalPerEpochNum == evalPerEpochNum - 1:
                        print ('\ntraining epoch %d...' % (epochInd))
                        print ('evaluating...')
                        self.evaluate(sess)
                    useRandSamp = 1 if epochInd < useRandSampEpochNum else 0
                    self.trainLayer(sess, layerInd, inPHValue, useRandSamp)
        
    def trainLayer(self, sess, layerInd, inPHValue, useRandSamp):
        inPH = self.inputPH if layerInd == 0 else self.hiddenPHList[layerInd - 1]
        outPHValue = sess.run(self.hiddenPHList[layerInd], feed_dict = {inPH: inPHValue})
        
        if useRandSamp:
            sameClassMaxDistPair = np.array([inPHValue[random.randint(0, self.ptNum-1)] for _ in range(2)])
            diffClassMinDistPair = self.getDiffClassMinDistPair(inPHValue, outPHValue)
        else:
            sameClassMaxDistPair = self.getSameClassMaxDistPair(inPHValue, outPHValue)
            diffClassMinDistPair = self.getDiffClassMinDistPair(inPHValue, outPHValue)
        
        trainFeedDict = {inPH: np.concatenate([sameClassMaxDistPair, diffClassMinDistPair], axis = 0)}
        sess.run(self.optimizer, feed_dict = trainFeedDict)

    def getSameClassMaxDistPair(self, inPHValue, outPHValue):
        firstPt = np.array([outPHValue[ind] for ind in self.sameClassArr[:, 0]])
        secondPt = np.array([outPHValue[ind] for ind in self.sameClassArr[:, 1]])
        maxDistInd = np.argmax(np.sum(np.square(firstPt - secondPt), axis = 1))
        return np.array([inPHValue[ind] for ind in self.sameClassArr[maxDistInd]])

    def getDiffClassMinDistPair(self, inPHValue, outPHValue):
        firstPt = np.array([outPHValue[ind] for ind in self.diffClassArr[:, 0]])
        secondPt = np.array([outPHValue[ind] for ind in self.diffClassArr[:, 1]])
        minDistInd = np.argmin(np.sum(np.square(firstPt - secondPt), axis = 1))
        return np.array([inPHValue[ind] for ind in self.diffClassArr[minDistInd]])

    def evaluate(self, sess):
        PHValueList = [sess.run(ph, feed_dict = {self.inputPH: self.ptArr}) for ph in self.hiddenPHList]
        sameDistList = []
        diffDistList = []
        for layerInd in range(self.L):
            valueArr = PHValueList[layerInd]
            maxSame = max([np.linalg.norm(valueArr[pair[0]] - valueArr[pair[1]]) for pair in self.sameClassArr])
            minDiff = min([np.linalg.norm(valueArr[pair[0]] - valueArr[pair[1]]) for pair in self.diffClassArr])
            sameDistList += [maxSame]
            diffDistList += [minDiff]

        print ('        maxSame \tminDiff')
        for ind in range(self.L):
            maxSame, minDiff = sameDistList[ind], diffDistList[ind]
            print ('L %d    ' % (ind), maxSame, '\t', minDiff, '\t', maxSame / minDiff)

if __name__ == '__main__':
    df = DataFormater()
    df.loadPtData('hw2pt.dat')
    df.loadClassData('hw2class.dat')
    df.buildDiffSameClassArr()

    som = SOM_perceptron(df.ptArr, df.sameClassArr, df.diffClassArr, wInitStddev = 1.0, bInitStddev = 0.1)
    som.buildModel(L = 5, nList = [5, 5, 5, 5, 5], EtaAtt = 0.0001, EtaRep = 1.0)
    som.trainModel(epochNum = 15000, evalPerEpochNum = 5000, useRandSampEpochNum = 0)
