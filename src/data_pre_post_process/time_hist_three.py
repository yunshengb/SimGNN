import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

args1 = {'astar': {'color': 'grey'},
         'beam5': {'color': 'deeppink'},
         'beam10': {'color': 'b'},
         'beam20': {'color': 'forestgreen'},
         'beam40': {'color': 'darkorange'},
         'beam80': {'color': 'cyan'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'darkcyan'},
         'graph2vec': {'color': 'darkcyan'},
         'siamese': {'color': 'red'},
         'transductive': {'color': 'red'}}


args2 = {'astar': {'marker': '*', 'facecolors': 'none', 'edgecolors': 'grey'},
         'beam5': {'marker': '|', 'facecolors': 'deeppink'},
         'beam10': {'marker': '_', 'facecolors': 'b'},
         'beam20': {'marker': 'D', 'facecolors': 'none',
                    'edgecolors': 'forestgreen'},
         'beam40': {'marker': '^', 'facecolors': 'none',
                    'edgecolors': 'darkorange'},
         'beam80': {'marker': 's', 'facecolors': 'none', 'edgecolors': 'cyan'},
         'hungarian': {'marker': 'X', 'facecolors': 'none',
                       'edgecolors': 'deepskyblue'},
         'vj': {'marker': 'h', 'facecolors': 'none',
                'edgecolors': 'darkcyan'},
         'graph2vec': {'marker': 'h', 'facecolors': 'none',
                       'edgecolors': 'darkcyan'},
         'siamese': {'marker': 'P',
                     'facecolors': 'none', 'edgecolors': 'red'},
         'transductive': {'marker': 'P',
                          'facecolors': 'none', 'edgecolors': 'red'}
         }


TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'movie': '#ff6666',
    'tvSeries': '#ff6666',
    'actor': 'lightskyblue',
    'actress': '#ffb3e6',
    'director': 'yellowgreen',
    'composer': '#c2c2f0',
    'producer': '#ffcc99',
    'cinematographer': 'gold'}


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30, }


if __name__ == '__main__':
    # aids  Beam80  79.604    Beam1 14.222  Beam2
    # linux Beam80  30.788    Beam1 11.358  Beam2
    # imdb  Beam80  139.266   Beam1 0       Beam2
    # name_list = ['A*', 'Beam80', 'Beam\n1', 'Hung\narian', 'VJ', 'Avg',
    #              'GCN\nMean\nPool', 'GCN\nMax\nPool', 'Att\nDeg',
    #              'Att\nCout', 'Att\nTrans\nCout', 'SimGNN']
    name_list = ['A*', 'Beam', 'Hung\narian', 'VJ', 'SimpleMean',
                 'HierarchicalMean', 'HierarchicalMax', 'AttDegree',
                 'AttGlobalContext', 'AttLearnableGC', 'SimGNN']
    num_list_aids = [5540.527, 19.026, 5.726, 8.801, 1.728, 2.139, 2.131, 1.586, 1.678, 1.681, 2.549]
    num_list_linux = [534.505, 7.923, 3.684, 4.735, 1.444, 2.185, 2.166, 1.929, 1.964, 2.084, 2.517]
    num_list_imdb = [0, 139.266, 120.349, 135.264, 1.371, 2.764, 2.865, 2.069, 2.227, 2.335, 2.997]
    color_list = ['grey', 'deeppink', 'blue', 'forestgreen', 'darkorange',
         'cyan','deepskyblue', 'darkcyan', 'darkcyan', 'red']
    index = np.arange(len(name_list))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(20, 12))
    plt.bar(index, num_list_aids, width=bar_width, align="center",
            color='white', label='AIDS', edgecolor='k', hatch="///", lw=3)
    # for a, b in zip(name_list, num_list_aids):
    #     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)

    plt.bar(index + bar_width, num_list_linux, width=bar_width, align="center",
            color='white', label='LINUX', edgecolor='k', hatch="\\\\", lw=3)
    # for a, b in zip(name_list, num_list_linux):
    #     plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)

    plt.bar(index + 2 * bar_width, num_list_imdb, width=bar_width, align="center",
            color='white', label='IMDB', edgecolor='k', hatch='', lw=3)
    # for a, b in zip(name_list, num_list_imdb):
    #     if b != 0:
    #         plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)
    plt.xticks(index + 0.25, ('A*', 'Beam', 'Hung\narian', 'VJ', 'Simple\nMean',
                 'Hierar\nchical\nMean', 'Hierar\nchical\nMax', 'Att\nDegree',
                 'Att\nGlobal\nContext', 'Att\nLearn-\nableGC', 'SimGNN'), fontsize=28)
    plt.yticks(fontsize=30)
    plt.ylabel("time(msec)", font2)
    plt.yscale("log")
    plt.legend(loc='best', prop={'size': 30})
    plt.grid(which=u'major', axis=u'y', linestyle='--')
    plt.tight_layout()
    plt.savefig("time.png")
    plt.savefig("time.eps")
    plt.close()
    # plt.show()






