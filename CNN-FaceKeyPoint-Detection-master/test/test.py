# coding: utf-8
"""

利用训练好的模型对整个测试集进行测试

"""

import time
from functools import partial
import cv2
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils import getDataFromTxt, logger


plot_point = np.zeros([3466,5,2])

TXT = '/home/pli/Face_Landmark/dataset/train/testImageList.txt'

template = '''################## Summary #####################
Test Number: %d
Time Consume: %.03f s
FPS: %.03f
LEVEL - %d
Mean Error:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
Failure:
    Left Eye       = %f
    Right Eye      = %f
    Nose           = %f
    Left Mouth     = %f
    Right Mouth    = %f
'''

def evaluateError(landmarkGt, landmarkP, bbox):
    e = np.zeros(5)
    for i in range(5):
        e[i] = norm(landmarkGt[i] - landmarkP[i])
    e = e / bbox.w
    print ('landmarkGt')
    print (landmarkGt)
    print ('landmarkP')
    print (landmarkP)
    print ('error', e)
    return e


def E(level=1):
    if level == 0:
        from level import level1 as P
        P = partial(P, FOnly=True)
    elif level == 1:
        from level import level1 as P
    elif level == 2:
        from level import level2 as P
    else:
        from level import level3 as P

    data = getDataFromTxt(TXT)
    error = np.zeros((len(data), 5))
    for i in range(len(data)):
        imgPath, bbox, landmarkGt = data[i]
        img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)

        landmarkP = P(img, bbox)
        plot_point[i] = landmarkP

        # real landmark
        landmarkP = bbox.reprojectLandmark(landmarkP)
        landmarkGt = bbox.reprojectLandmark(landmarkGt)
        error[i] = evaluateError(landmarkGt, landmarkP, bbox)
    return error

def plotError(e, name):
    # config global plot
    plt.rc('font', size=16)
    plt.rcParams["savefig.dpi"] = 240

    fig = plt.figure(figsize=(20, 15))
    binwidth = 0.001
    yCut = np.linspace(0, 70, 100)
    xCut = np.ones(100)*0.05
    # left eye
    ax = fig.add_subplot(321)
    data = e[:, 0]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left eye')
    # right eye
    ax = fig.add_subplot(322)
    data = e[:, 1]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right eye')
    # nose
    ax = fig.add_subplot(323)
    data = e[:, 2]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('nose')
    # left mouth
    ax = fig.add_subplot(325)
    data = e[:, 3]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('left mouth')
    # right mouth
    ax = fig.add_subplot(326)
    data = e[:, 4]
    ax.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), normed=1)
    ax.plot(xCut, yCut, 'r', linewidth=2)
    ax.set_title('right mouth')

    fig.suptitle('%s'%name)
    fig.savefig('/home/pli/Face_Landmark/log/%s.png'%name)


nameMapper = ['F_test', 'level1_test', 'level2_test', 'level3_test']

if __name__ == '__main__':

    level = 3

    t = time.clock()
    error = E(level)
    t = time.clock() - t

    N = len(error)
    fps = N / t
    errorMean = error.mean(0)
    # failure
    failure = np.zeros(5)
    threshold = 0.05
    for i in range(5):
        failure[i] = float(sum(error[:, i] > threshold)) / N
    # log string
    s = template % (N, t, fps, level, errorMean[0], errorMean[1], errorMean[2], \
        errorMean[3], errorMean[4], failure[0], failure[1], failure[2], \
        failure[3], failure[4])
    print (s)

    logfile = '/home/pli/Face_Landmark/log/{0}.log'.format(nameMapper[level])
    with open(logfile, 'w') as fd:
        fd.write(s)

    # plot error hist
    plotError(error, nameMapper[level])
