# coding: utf-8

"""

利用训练好的模型构造整个测试模型，这里分别提取了训练阶段多个CNN的不同训练结果

"""
import numpy as np
import caffe


class CNN(object):
    def __init__(self, net, model):
        self.net = net
        self.model = model
        try:
            self.cnn = caffe.Net(str(net), str(model), caffe.TEST)
        except:
            print ("Can not open %s, %s"%(net, model))

    def forward(self, data, layer='fc2'):
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2*i], x[2*i+1]]) for i in range(len(x)/2)])
        result = t(result)
        return result

# global cnns
cnn = dict(level1=None, level2=None, level3=None)
m1 = '_iter_1000000.caffemodel'
m2 = '_iter_100000.caffemodel'
m3 = '_iter_100000.caffemodel'

def getCNNs(level=1):
    types = ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2']
    if level == 1:
        if cnn['level1'] is None:
            F = CNN('/home/pli/Face_Landmark/prototxt/1_F_deploy.prototxt', '/home/pli/Face_Landmark//model/1_F/%s'%(m1))
            EN = CNN('/home/pli/Face_Landmark/prototxt/1_EN_deploy.prototxt', '/home/pli/Face_Landmark/model/1_EN/%s'%(m2))
            NM = CNN('/home/pli/Face_Landmark/prototxt/1_NM_deploy.prototxt', '/home/pli/Face_Landmark/model/1_NM/%s'%(m2))
            cnn['level1'] = [F, EN, NM]
        return cnn['level1']
    elif level == 2:
        if cnn['level2'] is None:
            cnn['level2'] = []
            for t in types:
                cnn['level2'].append(CNN('/home/pli/Face_Landmark/prototxt/2_%s_deploy.prototxt'%t,
                                         '/home/pli/Face_Landmark/model/2_%s/%s'%(t,m2)))
        return cnn['level2']
    else:
        if cnn['level3'] is None:
            cnn['level3'] = []
            for t in types:
                cnn['level3'].append(CNN('/home/pli/Face_Landmark/prototxt/3_%s_deploy.prototxt'%t,
                                         '/home/pli/Face_Landmark/model/3_%s/%s'%(t,m3)))
        return cnn['level3']
