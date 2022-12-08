# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import numpy as np

import classificationMethod
import util

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


def extractDigitFeatures(datum):
    features = util.Counter()
    FEATURE_WIDTH = 2
    FEATURE_HEIGHT = 2
    for i, j in [(i, j) for i in range(0, DIGIT_DATUM_WIDTH, FEATURE_WIDTH) for j in
                 range(0, DIGIT_DATUM_HEIGHT, FEATURE_HEIGHT)]:
        features[(i // FEATURE_WIDTH, j // FEATURE_HEIGHT)] = sum(
            [datum.getPixel(u, v) for u in range(i, i + FEATURE_WIDTH)
             for v in range(j, j + FEATURE_HEIGHT)])
    return features


def extractFaceFeatures(datum):
    """
    Your feature extraction playground for faces.
    It is your choice to modify this.
    """
    features = util.Counter()
    FEATURE_WIDTH = 5
    FEATURE_HEIGHT = 5
    for i, j in [(i, j) for i in range(0, FACE_DATUM_WIDTH, FEATURE_WIDTH) for j in
                 range(0, FACE_DATUM_HEIGHT, FEATURE_HEIGHT)]:
        features[(i // FEATURE_WIDTH, j // FEATURE_HEIGHT)] = sum(
            [datum.getPixel(u, v) for u in range(i, i + FEATURE_WIDTH)
             for v in range(j, j + FEATURE_HEIGHT)])
    return features


class PerceptronClassifier(classificationMethod.ClassificationMethod):
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.features = None
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = []
        # for label in legalLabels:
        #     self.weights[label] = util.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        # self.weights == weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = list(trainingData[0].keys())  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        self.weights = np.random.rand(len(self.legalLabels), len(self.features))
        NormalizedAccuracy = 0
        for iteration in range(self.max_iterations):
            correctPred = 0
            print("Starting iteration ", iteration, "...")
            i = 0
            for data, label in zip(trainingData, trainingLabels):
                maxLabelVal = 0
                predicted_value = 0
                for y in range(len(self.legalLabels)):
                    stepVal = np.dot(np.array(list(data.values())), self.weights[y])
                    if (stepVal > maxLabelVal):
                        maxLabelVal = stepVal
                        predicted_value = y
                if (predicted_value != label):
                    self.weights[label] = self.weights[label] + np.array(list(data.values()))
                    self.weights[predicted_value] = (self.weights[predicted_value]) - np.array(list(data.values()))
                else:
                    correctPred = correctPred + 1
            accuracy = correctPred * 100 / len(trainingData)
            NormalizedAccuracy = NormalizedAccuracy + accuracy
        # print("Accuracy:", NormalizedAccuracy / self.max_iterations)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = np.matmul(self.weights[l], np.array(list(datum.values())))
            label = vectors.argMax()
            guesses.append(label)
        return guesses

    def stepFunction(self, x):
        return np.where(x >= 0, 1, 0)

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

        return featuresWeights
