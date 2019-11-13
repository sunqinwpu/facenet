from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import os
from matplotlib import pyplot

# 加载图片


def loadImage(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    return pixels


# 提取一个人脸
def extractFace(filename, requiredSize=(160, 160)):
    image = loadImage(filename)
    detector = MTCNN()
    results = detector.detect_faces(image)
    print(results)
    if (len(results) == 0):
        return None
    x1, y1, width, height = results[0].get('box')
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1+width, y1+width
    face = image[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(requiredSize)
    faceArray = np.asarray(image)
    return faceArray


# 提取文件夹下的人脸和标识
def loadFaces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = directory + filename
        face = extractFace(path)
        faces.append(face)
    return faces


# 提取整个文件夹下的数据集
def loadDataset(directory):
    x,  y = list(),  list()
    for subdir in os.listdir(directory):
        path = directory + '/' + subdir + '/'
        print('pre load path:{}'.format(path))
        if not os.path.isdir(path):
            continue
        print('load path:{}'.format(path))
        faces = loadFaces(path)
        labels = [subdir for i in range(len(faces))]
        x.extend(faces)
        y.extend(labels)
    return np.asarray(x),  np.asarray(y)


def getEmbedding(model,  facePixels):
    facePixels = facePixels.astype('float32')
    mean,  std = facePixels.mean(),  facePixels.std()
    facePixels = (facePixels - mean) / std
    samples = np.expand_dims(facePixels,  axis=0)
    yPre = model.predict(samples)
    return yPre[0]


if __name__ == '__main__':
    model = load_model('facenet_keras.h5')

    datasetFile = '5-celebrity-faces-dataset.npz'
    if os.path.exists(datasetFile):
        print('load loadDataset from datasetFile :{}'.format(datasetFile))
        data = np.load('5-celebrity-faces-dataset.npz')
        trainX, trainY, testX, testY = data['arr_0'],  data['arr_1'],  data['arr_2'],  data['arr_3']
    else:
        trainX, trainY = loadDataset('5-celebrity-faces-dataset/train')
        testX, testY = loadDataset('5-celebrity-faces-dataset/val')
        np.savez_compressed(datasetFile, trainX, trainY, testX, testY)

    embeddingsFile = '5-celebrity-faces-embeddings.npz'
    if os.path.exists(embeddingsFile):
        print('load data embeddings from embeddingsFile:{}'.format(embeddingsFile))
        data = np.load(embeddingsFile)
        newTrainX, newTrainY, newTestX, newTestY = data['arr_0'],  data['arr_1'],  data['arr_2'],  data['arr_3']
    else:
        # extract embeddings
        newTrainX = list()
        for facePixels in trainX:
            embeddings = getEmbedding(model, facePixels)
            newTrainX.append(embeddings)
        newTrainX = np.asarray(newTrainX)
        newTrainY = trainY
        print("newTrainX shape:".format(newTrainX.shape))

        newTestX = list()
        for facePixels in testX:
            embeddings = getEmbedding(model, facePixels)
            newTestX.append(embeddings)
        newTestX = np.asarray(newTestX)
        newTestY = testY
        print("newTestX shape:".format(newTestX.shape))
        np.savez_compressed(embeddingsFile, newTrainX, newTrainY, newTestX, newTestY)

    # classfication model
    inputEncoder = Normalizer(norm='l2')
    newTrainX = inputEncoder.transform(newTrainX)
    newTestX = inputEncoder.transform(newTestX)

    outEncoder = LabelEncoder()
    outEncoder.fit(newTrainY)
    newTrainY = outEncoder.transform(newTrainY)
    newTestY = outEncoder.transform(newTestY)

    classModel = SVC(kernel='linear', probability=True)
    classModel.fit(newTrainX, newTrainY)

    yPreTrain = classModel.predict(newTrainX)
    yPreTest = classModel.predict(newTestX)

    scoreTrain = accuracy_score(newTrainY, yPreTrain)
    scoreTest = accuracy_score(newTestY, yPreTest)

    print('Accuracy: train=%.3f, test=%.3f' % (scoreTrain*100,  scoreTest*100))

    testFace = extractFace('test.jpg')
    testEmbedding = getEmbedding(model, testFace)
    testEmbeddings = list()
    testEmbeddings.append(testEmbedding)
    testEmbeddings = np.asarray(testEmbeddings)
    testEmbeddings = inputEncoder.transform(testEmbeddings)
    print("testEmbeddings:")
    print(testEmbeddings)
    testPreClass = classModel.predict(testEmbeddings)
    print("testPreClass:")
    print(testPreClass)
    testPreClassProb = classModel.predict_proba(testEmbeddings)

    testPreClassIndex = testPreClass[0]
    testPreClassProb = testPreClassProb[0, testPreClassIndex] * 100
    testPreNames = outEncoder.inverse_transform(testPreClass)
    print('predict test.jpg:{}'.format(testPreNames[0]))
    print('predict test.jpg probability:{}'.format(testPreClassProb))
    pyplot.imshow(testFace)
    title = '%s (%3f)' % (testPreNames[0], testPreClassProb)
    pyplot.title(title)
    pyplot.show()
