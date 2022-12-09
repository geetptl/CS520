import random
import time

import numpy as np

import cnn
import naiveBayes
import perceptron
import samples

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def loadDataSet(dataset):
    if (dataset == "faces"):
        rawTrainingData = samples.loadDataFile("facedata/facedatatrain", 451, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", 451)
        rawValidationData = samples.loadDataFile("facedata/facedatavalidation", 301, FACE_DATUM_WIDTH,
                                                 FACE_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("facedata/facedatavalidationlabels", 301)
        rawTestData = samples.loadDataFile("facedata/facedatatest", 150, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", 150)
    else:
        rawTrainingData = samples.loadDataFile("digitdata/trainingimages", 5000, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", 5000)
        rawValidationData = samples.loadDataFile("digitdata/validationimages", 1000, DIGIT_DATUM_WIDTH,
                                                 DIGIT_DATUM_HEIGHT)
        validationLabels = samples.loadLabelsFile("digitdata/validationlabels", 1000)
        rawTestData = samples.loadDataFile("digitdata/testimages", 1000, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
        testLabels = samples.loadLabelsFile("digitdata/testlabels", 1000)
    return rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels


def getLegalLabels(dataset):
    if (dataset == 'digits'):
        return list(range(10))
    else:
        return list(range(2))


def loadClassifier(classifier, labels):
    if classifier == 'nb':
        return naiveBayes.NaiveBayesClassifier(labels)
    elif classifier == 'perceptron':
        return perceptron.PerceptronClassifier(labels, 3)
    elif classifier == 'cnn':
        return cnn.ConvolutionalNeuralNetworkClassifier(3)


def extractFeatures(rawTrainingData, classifierName, dataset):
    if classifierName == 'nb':
        if dataset == 'faces':
            return list(map(naiveBayes.extractFaceFeatures, rawTrainingData))
        else:
            return list(map(naiveBayes.extractDigitFeatures, rawTrainingData))
    elif classifierName == 'perceptron':
        if dataset == 'faces':
            return list(map(perceptron.extractFaceFeatures, rawTrainingData))
        else:
            return list(map(perceptron.extractDigitFeatures, rawTrainingData))
    elif classifierName == 'cnn':
        if dataset == 'faces':
            return list(map(cnn.extractFaceFeatures, rawTrainingData))
        else:
            return list(map(cnn.extractDigitFeatures, rawTrainingData))


if __name__ == '__main__':
    datasets = ['digits', 'faces']
    classifiers = ['nb', 'perceptron', 'cnn']
    accuracy = {}
    runs = 10
    for dataset, data in zip(datasets, list(map(loadDataSet, datasets))):
        dataset_ = {}
        rawTrainingData, trainingLabels, rawValidationData, validationLabels, rawTestData, testLabels = data
        legalLabels = getLegalLabels(dataset)
        for classifierName, classifier in zip(classifiers,
                                              list(map(loadClassifier, classifiers, [legalLabels] * len(classifiers)))):
            classifier_ = {}
            trainingData = extractFeatures(rawTrainingData, classifierName, dataset)
            validationData = extractFeatures(rawValidationData, classifierName, dataset)
            testData = extractFeatures(rawTestData, classifierName, dataset)
            for datasize in range(10, 110, 10):
                print("Training on {}% data".format(datasize))
                trainingDataHolder = list(zip(trainingData, trainingLabels))
                runsAccuracy = []
                runtime = []
                for i in range(runs):
                    print("Run", i)
                    random.shuffle(trainingDataHolder)
                    trainingData_ = [i for i, j in trainingDataHolder[:int(datasize * len(trainingData) / 100)]]
                    trainingLabels_ = [j for i, j in trainingDataHolder[:int(datasize * len(trainingData) / 100)]]
                    print("Training on size", len(trainingData_))
                    start_time = time.process_time_ns()
                    classifier.train(trainingData_, trainingLabels_, validationData, validationLabels)
                    runtime.append(1e6 * (time.process_time_ns() - start_time))
                    print("Validating...")
                    guesses = classifier.classify(validationData)
                    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
                    print(str(correct),
                          ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (
                                  100.0 * correct / len(validationLabels)))
                    print("Testing...")
                    guesses = classifier.classify(testData)
                    correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)
                    print(str(correct),
                          ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (
                                  100.0 * correct / len(testLabels)))
                    runsAccuracy.append(100.0 * correct / len(testLabels))
                stats_ = {"mean": np.mean(runsAccuracy), "median": np.median(runsAccuracy),
                          "std": np.std(runsAccuracy), "average_runtime": np.mean(runtime)}
                print(stats_)
                classifier_[datasize] = stats_
            dataset_[classifierName] = classifier_
        accuracy[dataset] = dataset_
    print("============Final============")
    print(accuracy)
