import dataClassifier

if __name__ == '__main__':
    accuracy = {}
    for dataset in ['digits', 'faces']:
        dataset_ = {}
        for classifier in ['nb']:
            classifier_ = {}
            for datasize in range(20, 110, 10):
                opts = "-d {} -c {} -t {} -f -a".format(dataset, classifier, datasize)
                args, options = dataClassifier.readCommand(opts.split(" "))
                classifier_[datasize] = dataClassifier.runClassifier(args, options)
            dataset_[classifier] = classifier_
        accuracy[dataset] = dataset_
    print(accuracy)
