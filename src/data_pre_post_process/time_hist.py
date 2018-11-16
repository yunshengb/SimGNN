import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


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
         'size': 25, }


if __name__ == '__main__':
    name_list = ['A*', 'Beam80', 'Hung\narian', 'VJ', 'Avg',
                 'GCN\nMean\nPool', 'GCN\nMax\nPool', 'Att\nDeg',
                 'Att\nCout', 'Att\nTrans\nCout', 'Att\nTrans\nCont+PNC']
    num_list_aids = [5540.527, 79.604, 5.726, 8.801, 1.728, 2.139, 2.131, 1.886, 1.681, 1.678, 2.549]
    num_list_linux = [534.505, 30.788, 3.684, 10.242, 0, 2.385, 2.166, 1.964, 1.929, 0, 2.084]
    num_list_imdb = [139.266, 120.349, 135.264, 1.371, 2.069, 0, 1.371, 0, 1.542, 0]
    color_list = ['grey', 'deeppink', 'blue', 'forestgreen', 'darkorange',
         'cyan','deepskyblue', 'darkcyan', 'darkcyan', 'red']
    plt.figure(figsize=(16, 9))
    plt.bar(left=name_list, height=num_list_aids, width=0.5, align="center", color='r')
    for a, b in zip(name_list, num_list_aids):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=30)
    plt.ylabel("time(msec)", font2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("time_1.png")
    plt.savefig("time_1.eps")
    plt.close()

    plt.figure(figsize=(16, 9))
    plt.bar(left=name_list, height=num_list_linux, width=0.5, align="center", color='b')
    for a, b in zip(name_list, num_list_linux):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=30)
    plt.ylabel("time(msec)", font2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("time_2.png")
    plt.savefig("time_2.eps")
    plt.close()

    name_list = ['Beam', 'Hung\narian', 'VJ', 'Avg',
                 'GCN\nMean\nPool', 'GCN\nMax\nPool', 'Att\nDeg',
                 'Att\nCout', 'Att\nTrans\nCout', 'Att\nTrans\nCont+PNC']
    plt.figure(figsize=(16, 9))
    plt.bar(left=name_list, height=num_list_imdb, width=0.5, align="center", color='g')
    for a, b in zip(name_list, num_list_imdb):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=30)
    plt.ylabel("time(msec)", font2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("time_3.png")
    plt.savefig("time_3.eps")
    plt.close()
    # plt.show()






