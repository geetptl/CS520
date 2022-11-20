# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import math

import numpy as np
import matplotlib.pyplot as plt

import classificationMethod
import util


def showDatum(data, filename):
    viewer = np.zeros(shape=(max([u for u, _ in data.keys()]) + 1, max([u for _, u in data.keys()]) + 1))
    for feature, value in data.items():
        viewer[feature] = value
    viewer = viewer / np.max(viewer)
    viewer = np.pad(np.rot90(np.fliplr(np.kron(viewer, np.ones((20, 20))))), ((1, 1), (1, 1)), 'constant',
                    constant_values=1)
    plt.imsave(filename, viewer, cmap='gray', vmin=0, vmax=1)


class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels):
        self.features = None
        self.posteriors = None
        self.conditionals = None
        self.priors = None
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1  # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False  # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in trainingData for f in list(datum.keys())]))

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        priors = util.Counter()
        for p in trainingLabels:
            priors.incrementAll([p], 1)
        priors.normalize()
        self.priors = priors

        bestAccuracy = -math.inf
        bestConditionals = None
        for k_ in kgrid:
            print("k :", k_)
            conditionals_ = util.Counter()
            for data, label in zip(trainingData, trainingLabels):
                labelConditional = util.Counter()
                for feature, value in data.items():
                    labelConditional.incrementAll([feature], value + k_)
                labelConditional.normalize()
                conditionals_[label] = labelConditional
            self.conditionals = conditionals_
            for label, data in conditionals_.items():
                showDatum(data, "out/training/k_{}_label_{}.png".format(k_, label))
            guesses = self.classify(validationData)
            correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            accuracy = correct * 100 / len(validationLabels)
            print("Accuracy :", accuracy)
            if accuracy > bestAccuracy:
                self.k = k_
                bestConditionals = conditionals_
        self.conditionals = bestConditionals
        print("Best k :", self.k)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = []  # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            label = posterior.argMax()
            guesses.append(label)
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()
        for label in self.legalLabels:
            c = 0
            for feature, p in self.conditionals[label]:
                if datum[feature] > 0:
                    c += math.log(p)
            logJoint[label] = math.log(self.priors[label]) + c
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresOdds
