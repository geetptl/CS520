import classificationMethod
import util

import warnings

import pandas as pd
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

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


class ConvolutionalNeuralNetworkClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, max_iterations):
        self.type = "MLP"
        self.modelMLP = None
        self.features = None
        self.max_iterations = max_iterations

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        print(len(trainingData))
        df = pd.DataFrame()
        vd = pd.DataFrame()
        df = df.append(trainingData, ignore_index=True)
        vd = vd.append(validationData, ignore_index=True)
        """ model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(16,(3,3),activation = "relu" , input_shape = (28,28)))
        model.add(tf.keras.layers.MaxPooling2D(2,2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(10,activation = "softmax"))
        print(len(trainingLabels))
        model.summary()
        adam=Adam(learning_rate=0.1) """
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])
        # value = model.fit(df, np.array(trainingLabels), validation_data=(vd,np.array(validationLabels)),epochs=self.max_iterations)
        # print("Accuracy: &2.f%%" %(model.evaluate(vd, validationLabels)[1]*100))
        self.modelMLP = MLPClassifier(hidden_layer_sizes=(1000,), activation='tanh', alpha=0.01, random_state=1,
                                      warm_start=True, learning_rate='adaptive', max_iter=200).fit(df, trainingLabels)
        # print(self.modelMLP.score(vd, validationLabels))
        predicted_y = self.modelMLP.predict(vd)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """

        dataNew = pd.DataFrame()
        dataNew = dataNew.append(data, ignore_index=True)
        predicted_y = self.modelMLP.predict(dataNew)
        return predicted_y
