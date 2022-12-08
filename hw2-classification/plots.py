import matplotlib.pyplot as plt

data = {'digits': {'nb': {10: {'mean': 72.37000000000002, 'median': 72.45, 'std': 1.3342038824707405},
                          20: {'mean': 73.61, 'median': 73.75, 'std': 1.1085576214162256},
                          30: {'mean': 74.47, 'median': 74.65, 'std': 0.591692487699479},
                          40: {'mean': 74.34, 'median': 74.15, 'std': 0.5589275444992865},
                          50: {'mean': 74.66, 'median': 74.8, 'std': 0.5782732917920372},
                          60: {'mean': 74.57, 'median': 74.6, 'std': 0.5967411499134271},
                          70: {'mean': 74.78, 'median': 74.7, 'std': 0.4975942121849881},
                          80: {'mean': 74.55, 'median': 74.5, 'std': 0.36945906403822193},
                          90: {'mean': 74.58000000000001, 'median': 74.5, 'std': 0.4093897898091764},
                          100: {'mean': 74.9, 'median': 74.9, 'std': 0.0}},
                   'perceptron': {10: {'mean': 68.72999999999999, 'median': 69.4, 'std': 3.9117898716572195},
                                  20: {'mean': 74.89000000000001, 'median': 75.5, 'std': 2.9357963144605246},
                                  30: {'mean': 75.30999999999999, 'median': 76.1, 'std': 2.552430214521055},
                                  40: {'mean': 76.33000000000001, 'median': 76.95, 'std': 3.3124160366717232},
                                  50: {'mean': 77.49, 'median': 78.55000000000001, 'std': 2.610919378303357},
                                  60: {'mean': 78.57, 'median': 78.6, 'std': 1.3364505228402586},
                                  70: {'mean': 79.02000000000001, 'median': 79.95, 'std': 4.196617685708336},
                                  80: {'mean': 78.67000000000002, 'median': 78.55, 'std': 1.369707998078425},
                                  90: {'mean': 78.22999999999999, 'median': 78.0, 'std': 1.6905916124244778},
                                  100: {'mean': 78.38, 'median': 77.85, 'std': 2.485880125830688}}}, 'faces': {
    'nb': {10: {'mean': 76.66666666666667, 'median': 77.33333333333334, 'std': 3.0840089349920996},
           20: {'mean': 79.73333333333332, 'median': 79.66666666666666, 'std': 2.2744962812309306},
           30: {'mean': 82.66666666666667, 'median': 83.0, 'std': 3.141125063837268},
           40: {'mean': 82.13333333333334, 'median': 82.66666666666667, 'std': 1.5719768163402137},
           50: {'mean': 82.86666666666666, 'median': 82.33333333333334, 'std': 2.1509687739868912},
           60: {'mean': 84.13333333333334, 'median': 84.0, 'std': 1.8571184369578835},
           70: {'mean': 83.20000000000002, 'median': 83.33333333333333, 'std': 1.571976816340213},
           80: {'mean': 83.86666666666667, 'median': 83.66666666666666, 'std': 1.8086213288334048},
           90: {'mean': 84.2, 'median': 85.33333333333333, 'std': 2.2320892057044284},
           100: {'mean': 84.66666666666666, 'median': 84.66666666666667, 'std': 1.4210854715202004e-14}},
    'perceptron': {10: {'mean': 74.46666666666667, 'median': 78.0, 'std': 8.516911281549069},
                   20: {'mean': 81.73333333333332, 'median': 84.0, 'std': 6.016274225428523},
                   30: {'mean': 79.46666666666667, 'median': 80.0, 'std': 6.173779681488122},
                   40: {'mean': 81.0, 'median': 82.0, 'std': 7.075780286771676},
                   50: {'mean': 81.6, 'median': 83.0, 'std': 5.251031644670726},
                   60: {'mean': 81.2, 'median': 81.33333333333334, 'std': 4.833218389437828},
                   70: {'mean': 84.06666666666666, 'median': 84.66666666666667, 'std': 2.0099751242241792},
                   80: {'mean': 82.86666666666666, 'median': 84.66666666666667, 'std': 4.000555516980665},
                   90: {'mean': 82.66666666666667, 'median': 82.0, 'std': 3.614784456460257},
                   100: {'mean': 83.26666666666667, 'median': 85.33333333333333, 'std': 4.992883824894073}}}}

if __name__ == '__main__':
    digits_nb_mean = [d1['mean'] for p, d1 in data['digits']['nb'].items()]
    digits_nb_std = [d1['std'] for p, d1 in data['digits']['nb'].items()]
    digits_perceptron_mean = [d1['mean'] for p, d1 in data['digits']['perceptron'].items()]
    digits_perceptron_std = [d1['std'] for p, d1 in data['digits']['perceptron'].items()]
    print(digits_nb_mean)
    print(digits_nb_std)
    print(digits_perceptron_mean)
    print(digits_perceptron_std)

    faces_nb_mean = [d1['mean'] for p, d1 in data['faces']['nb'].items()]
    faces_nb_std = [d1['std'] for p, d1 in data['faces']['nb'].items()]
    faces_perceptron_mean = [d1['mean'] for p, d1 in data['faces']['perceptron'].items()]
    faces_perceptron_std = [d1['std'] for p, d1 in data['faces']['perceptron'].items()]
    print(faces_nb_mean)
    print(faces_nb_std)
    print(faces_perceptron_mean)
    print(faces_perceptron_std)

    plt.xlabel('Data size (%)')
    plt.ylabel('Test accuracy (%)')
    plt.plot(list(range(10, 101, 10)), digits_nb_mean, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), digits_perceptron_mean, label='Perceptron')
    plt.legend(loc="upper left")
    plt.savefig("out/digits_learning_rate.svg", bbox_inches='tight', format='svg')

    plt.clf()
    plt.xlabel('Data size (%)')
    plt.ylabel('Test accuracy (%)')
    plt.plot(list(range(10, 101, 10)), faces_nb_mean, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), faces_perceptron_mean, label='Perceptron')
    plt.legend(loc="upper left")
    plt.savefig("out/faces_learning_rate.svg", bbox_inches='tight', format='svg')

    plt.xlabel('Data size (%)')
    plt.ylabel('Standard deviation')
    plt.plot(list(range(10, 101, 10)), digits_nb_std, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), digits_perceptron_std, label='Perceptron')
    plt.legend(loc="upper left")
    plt.savefig("out/digits_std.svg", bbox_inches='tight', format='svg')

    plt.clf()
    plt.xlabel('Data size (%)')
    plt.ylabel('Standard deviation')
    plt.plot(list(range(10, 101, 10)), faces_nb_std, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), faces_perceptron_std, label='Perceptron')
    plt.legend(loc="upper left")
    plt.savefig("out/faces_std.svg", bbox_inches='tight', format='svg')
