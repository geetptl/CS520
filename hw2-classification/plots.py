import matplotlib.pyplot as plt

data = {'digits': {
    'nb': {10: {'mean': 72.75, 'median': 72.85, 'std': 1.4685026387446483, 'average_runtime': 4999.615822499999},
           20: {'mean': 74.06, 'median': 73.6, 'std': 0.9286549412995105, 'average_runtime': 5746.6824516},
           30: {'mean': 74.23, 'median': 74.45, 'std': 0.8718371407550836, 'average_runtime': 6540.6013148},
           40: {'mean': 74.46, 'median': 74.35, 'std': 0.8404760555780267, 'average_runtime': 7300.0992761},
           50: {'mean': 74.82000000000001, 'median': 75.05, 'std': 0.6289674077406561,
                'average_runtime': 8139.952517499999},
           60: {'mean': 74.83999999999999, 'median': 74.85, 'std': 0.5023942674832206,
                'average_runtime': 8893.791643300001},
           70: {'mean': 74.64, 'median': 74.4, 'std': 0.6102458520956943, 'average_runtime': 9646.538788599999},
           80: {'mean': 74.84, 'median': 74.80000000000001, 'std': 0.36660605559646653,
                'average_runtime': 10438.1625491},
           90: {'mean': 74.43, 'median': 74.55000000000001, 'std': 0.46914816422959527,
                'average_runtime': 11262.0307224},
           100: {'mean': 74.9, 'median': 74.9, 'std': 0.0, 'average_runtime': 11985.0969979}}, 'perceptron': {
        10: {'mean': 70.60000000000001, 'median': 70.55, 'std': 2.914446774260941, 'average_runtime': 165.398969},
        20: {'mean': 71.38000000000001, 'median': 72.2, 'std': 4.7385229766246795,
             'average_runtime': 325.92722919999994},
        30: {'mean': 74.49, 'median': 74.19999999999999, 'std': 3.510113958264033, 'average_runtime': 487.2611151},
        40: {'mean': 76.77, 'median': 76.5, 'std': 2.34266941756621, 'average_runtime': 649.3453283},
        50: {'mean': 76.50999999999999, 'median': 76.4, 'std': 2.0141747689810843, 'average_runtime': 787.7739005},
        60: {'mean': 78.66999999999999, 'median': 79.05, 'std': 1.9214838016491333,
             'average_runtime': 951.0130746000001},
        70: {'mean': 77.17, 'median': 78.05000000000001, 'std': 3.5642811336930227,
             'average_runtime': 1080.0789787999997},
        80: {'mean': 77.32, 'median': 77.0, 'std': 3.8154423072561325, 'average_runtime': 1230.2530795999999},
        90: {'mean': 77.81, 'median': 77.75, 'std': 1.8774717041809195, 'average_runtime': 1380.6245783},
        100: {'mean': 78.0, 'median': 79.45, 'std': 3.8691084244306198, 'average_runtime': 1532.7866778}}, 'mlp': {
        10: {'mean': 82.73000000000002, 'median': 82.55, 'std': 1.2434226956268748,
             'average_runtime': 37276.452271500006},
        20: {'mean': 86.0, 'median': 86.4, 'std': 0.912140340079308, 'average_runtime': 63995.22092199999},
        30: {'mean': 87.68, 'median': 87.4, 'std': 0.932523458149982, 'average_runtime': 96221.2141686},
        40: {'mean': 89.00999999999999, 'median': 89.05, 'std': 0.4908156476723198,
             'average_runtime': 121452.92237120001},
        50: {'mean': 89.61999999999999, 'median': 89.75, 'std': 0.44226688774992057, 'average_runtime': 146638.3061421},
        60: {'mean': 90.8, 'median': 90.75, 'std': 0.4219004621945807, 'average_runtime': 179139.06127959996},
        70: {'mean': 91.09, 'median': 90.9, 'std': 0.6040695324215586, 'average_runtime': 211367.6290536},
        80: {'mean': 91.77000000000001, 'median': 91.80000000000001, 'std': 0.5514526271584902,
             'average_runtime': 235350.2569791},
        90: {'mean': 92.39999999999999, 'median': 92.4, 'std': 0.3224903099319426,
             'average_runtime': 241991.83123999997},
        100: {'mean': 92.86999999999999, 'median': 92.9, 'std': 0.23259406699226085,
              'average_runtime': 269337.7604554}}}, 'faces': {'nb': {
    10: {'mean': 78.06666666666668, 'median': 77.66666666666666, 'std': 2.2597935008904377,
         'average_runtime': 307.7677154},
    20: {'mean': 80.2, 'median': 79.66666666666666, 'std': 2.6507860469428057, 'average_runtime': 364.04964859999995},
    30: {'mean': 82.46666666666665, 'median': 82.66666666666667, 'std': 1.7139946842909914,
         'average_runtime': 429.6126564},
    40: {'mean': 81.53333333333333, 'median': 81.66666666666666, 'std': 1.4621141466307521,
         'average_runtime': 495.8993838},
    50: {'mean': 83.46666666666667, 'median': 83.0, 'std': 1.5719768163402132, 'average_runtime': 569.1966669999999},
    60: {'mean': 83.66666666666667, 'median': 84.0, 'std': 1.2382783747337818, 'average_runtime': 622.4625003},
    70: {'mean': 83.33333333333333, 'median': 83.33333333333333, 'std': 0.9888264649460878,
         'average_runtime': 681.4010626},
    80: {'mean': 83.60000000000001, 'median': 83.66666666666666, 'std': 1.6653327995729046,
         'average_runtime': 736.9696965},
    90: {'mean': 84.86666666666667, 'median': 85.33333333333333, 'std': 1.5506271132817364,
         'average_runtime': 788.9674749},
    100: {'mean': 84.66666666666666, 'median': 84.66666666666667, 'std': 1.4210854715202004e-14,
          'average_runtime': 849.7635757}}, 'perceptron': {
    10: {'mean': 75.13333333333334, 'median': 78.0, 'std': 8.284121357553053, 'average_runtime': 3.2443701999999996},
    20: {'mean': 78.93333333333332, 'median': 82.0, 'std': 8.155161964465615, 'average_runtime': 6.1698957},
    30: {'mean': 80.26666666666667, 'median': 80.66666666666666, 'std': 3.2138588781850537,
         'average_runtime': 9.052759},
    40: {'mean': 81.33333333333333, 'median': 82.0, 'std': 4.58015040995138, 'average_runtime': 12.0477585},
    50: {'mean': 82.93333333333334, 'median': 83.33333333333334, 'std': 2.9992591677872005,
         'average_runtime': 14.936179900000003},
    60: {'mean': 80.8, 'median': 81.0, 'std': 4.3594087264724815, 'average_runtime': 17.739544499999997},
    70: {'mean': 83.06666666666668, 'median': 84.0, 'std': 3.6539172282785928, 'average_runtime': 20.639914599999997},
    80: {'mean': 82.33333333333334, 'median': 82.66666666666666, 'std': 3.8093452339097746,
         'average_runtime': 23.6716513},
    90: {'mean': 84.33333333333333, 'median': 84.66666666666667, 'std': 2.4267032964268402,
         'average_runtime': 26.172939199999995},
    100: {'mean': 84.0, 'median': 84.33333333333334, 'std': 2.3094010767585056, 'average_runtime': 29.1016526}},
                                                              'mlp': {10: {'mean': 78.53333333333333,
                                                                           'median': 79.66666666666667,
                                                                           'std': 4.796294866294653,
                                                                           'average_runtime': 4141.092853200001},
                                                                      20: {'mean': 82.33333333333334,
                                                                           'median': 82.33333333333334,
                                                                           'std': 3.309917756615041,
                                                                           'average_runtime': 6648.722892099999},
                                                                      30: {'mean': 87.26666666666668,
                                                                           'median': 87.33333333333333,
                                                                           'std': 1.7499206331208919,
                                                                           'average_runtime': 9081.956591600001},
                                                                      40: {'mean': 86.46666666666667, 'median': 87.0,
                                                                           'std': 1.978776277287444,
                                                                           'average_runtime': 13003.307816699999},
                                                                      50: {'mean': 87.53333333333333,
                                                                           'median': 87.33333333333333,
                                                                           'std': 1.4621141466307561,
                                                                           'average_runtime': 18357.5035244},
                                                                      60: {'mean': 87.93333333333332,
                                                                           'median': 87.66666666666666,
                                                                           'std': 2.1176506899024785,
                                                                           'average_runtime': 19324.731858499996},
                                                                      70: {'mean': 87.8, 'median': 88.0,
                                                                           'std': 1.3999999999999995,
                                                                           'average_runtime': 22447.725187099997},
                                                                      80: {'mean': 88.0, 'median': 88.0,
                                                                           'std': 1.1155467020454326,
                                                                           'average_runtime': 25164.983646999997},
                                                                      90: {'mean': 88.0, 'median': 88.0,
                                                                           'std': 0.8432740427115671,
                                                                           'average_runtime': 40648.8287072},
                                                                      100: {'mean': 87.8, 'median': 88.0,
                                                                            'std': 0.9451631252505223,
                                                                            'average_runtime': 31642.654354899998}}}}

if __name__ == '__main__':
    digits_nb_mean = [d1['mean'] for p, d1 in data['digits']['nb'].items()]
    digits_nb_std = [d1['std'] for p, d1 in data['digits']['nb'].items()]
    digits_nb_runtime = [d1['average_runtime'] for p, d1 in data['digits']['nb'].items()]
    digits_perceptron_mean = [d1['mean'] for p, d1 in data['digits']['perceptron'].items()]
    digits_perceptron_std = [d1['std'] for p, d1 in data['digits']['perceptron'].items()]
    digits_perceptron_runtime = [d1['average_runtime'] for p, d1 in data['digits']['perceptron'].items()]
    digits_mlp_mean = [d1['mean'] for p, d1 in data['digits']['mlp'].items()]
    digits_mlp_std = [d1['std'] for p, d1 in data['digits']['mlp'].items()]
    digits_mlp_runtime = [d1['average_runtime'] for p, d1 in data['digits']['mlp'].items()]

    faces_nb_mean = [d1['mean'] for p, d1 in data['faces']['nb'].items()]
    faces_nb_std = [d1['std'] for p, d1 in data['faces']['nb'].items()]
    faces_nb_runtime = [d1['average_runtime'] for p, d1 in data['faces']['nb'].items()]
    faces_perceptron_mean = [d1['mean'] for p, d1 in data['faces']['perceptron'].items()]
    faces_perceptron_std = [d1['std'] for p, d1 in data['faces']['perceptron'].items()]
    faces_perceptron_runtime = [d1['average_runtime'] for p, d1 in data['faces']['perceptron'].items()]
    faces_mlp_mean = [d1['mean'] for p, d1 in data['faces']['mlp'].items()]
    faces_mlp_std = [d1['std'] for p, d1 in data['faces']['mlp'].items()]
    faces_mlp_runtime = [d1['average_runtime'] for p, d1 in data['faces']['mlp'].items()]

    plt.clf()
    ax = plt.gca()
    ax.set_ylim([50, 100])
    plt.xlabel('Data size (%)')
    plt.ylabel('Test accuracy (%)')
    plt.plot(list(range(10, 101, 10)), digits_nb_mean, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), digits_perceptron_mean, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), digits_mlp_mean, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/digits_learning_rate.svg", bbox_inches='tight', format='svg')

    plt.clf()
    ax = plt.gca()
    ax.set_ylim([50, 100])
    plt.xlabel('Data size (%)')
    plt.ylabel('Test accuracy (%)')
    plt.plot(list(range(10, 101, 10)), faces_nb_mean, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), faces_perceptron_mean, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), faces_mlp_mean, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/faces_learning_rate.svg", bbox_inches='tight', format='svg')

    plt.clf()
    ax = plt.gca()
    ax.set_ylim([0, 8])
    plt.xlabel('Data size (%)')
    plt.ylabel('Standard deviation')
    plt.plot(list(range(10, 101, 10)), digits_nb_std, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), digits_perceptron_std, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), digits_mlp_std, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/digits_std.svg", bbox_inches='tight', format='svg')

    plt.clf()
    ax = plt.gca()
    ax.set_ylim([0, 8])
    plt.xlabel('Data size (%)')
    plt.ylabel('Standard deviation')
    plt.plot(list(range(10, 101, 10)), faces_nb_std, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), faces_perceptron_std, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), digits_mlp_std, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/faces_std.svg", bbox_inches='tight', format='svg')

    plt.clf()
    plt.xlabel('Data size (%)')
    plt.ylabel('Training time (ms)')
    plt.plot(list(range(10, 101, 10)), digits_nb_runtime, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), digits_perceptron_runtime, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), digits_mlp_runtime, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/digits_runtime.svg", bbox_inches='tight', format='svg')

    plt.clf()
    plt.xlabel('Data size (%)')
    plt.ylabel('Training time (ms)')
    plt.plot(list(range(10, 101, 10)), faces_nb_runtime, label='Naive Bayes')
    plt.plot(list(range(10, 101, 10)), faces_perceptron_runtime, label='Perceptron')
    plt.plot(list(range(10, 101, 10)), faces_mlp_runtime, label='MLP')
    plt.legend(loc="upper left")
    plt.savefig("out/faces_runtime.svg", bbox_inches='tight', format='svg')
