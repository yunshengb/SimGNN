import seaborn as sns
import os
import networkx as nx
import matplotlib.pyplot as plt


font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18, }


def StatData():
    aids_count = 0
    aids_nodes = {}
    linux_count = 0
    linux_nodes = {}
    IMDBMulti_count = 0
    IMDBMulti_nodes = {}

    for filename in os.listdir("../../data/AIDS700nef/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/AIDS700nef/train/' + filename
            G = nx.read_gexf(f)
            aids_nodes[filename] = len(G)
            aids_count = aids_count + 1

    for filename in os.listdir("../../data/AIDS700nef/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/AIDS700nef/test/' + filename
            G = nx.read_gexf(f)
            aids_nodes[filename] = len(G)
            aids_count = aids_count + 1

    for filename in os.listdir("../../data/linux/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/linux/train/' + filename
            G = nx.read_gexf(f)
            linux_nodes[filename] = len(G)
            linux_count = linux_count + 1

    for filename in os.listdir("../../data/linux/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/linux/test/' + filename
            G = nx.read_gexf(f)
            linux_nodes[filename] = len(G)
            linux_count = linux_count + 1

    for filename in os.listdir("../../data/IMDBMulti/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDBMulti/train/' + filename
            G = nx.read_gexf(f)
            IMDBMulti_nodes[filename] = len(G)
            IMDBMulti_count = IMDBMulti_count + 1

    for filename in os.listdir("../../data/IMDBMulti/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDBMulti/test/' + filename
            G = nx.read_gexf(f)
            IMDBMulti_nodes[filename] = len(G)
            IMDBMulti_count = IMDBMulti_count + 1

    return aids_count, aids_nodes, linux_count, linux_nodes, IMDBMulti_count, IMDBMulti_nodes


if __name__ == '__main__':
    aids_count, aids_nodes, linux_count, linux_nodes, IMDBMulti_count, IMDBMulti_nodes = StatData()

    aids_list = list(aids_nodes.values())
    linux_list = list(linux_nodes.values())
    IMDBMulti_list = list(IMDBMulti_nodes.values())

    max_aids = max(aids_list)
    max_linux = max(linux_list)
    max_IMDBMulti = max(IMDBMulti_list)

    aids_size = list(range(0, max_aids + 1))
    linux_size = list(range(0, max_linux + 1))
    IMDBMulti_size = list(range(0, max_IMDBMulti + 1))

    aids_freq = [0] * (max_aids + 1)
    linux_freq = [0] * (max_linux + 1)
    IMDBMulti_freq = [0] * (max_IMDBMulti + 1)

    for num in aids_list:
        aids_freq[num] = aids_freq[num] + 1

    for num in linux_list:
        linux_freq[num] = linux_freq[num] + 1

    for num in IMDBMulti_list:
        IMDBMulti_freq[num] = IMDBMulti_freq[num] + 1

    plt.figure(figsize=(8, 6))
    plt.plot(aids_size, aids_freq, color='red', marker='o',
             markerfacecolor='white', label="AIDS", linewidth=2,
             markersize=10)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    plt.xscale('log')
    # plt.title('Train')
    plt.legend(loc='best')
    # plt.grid(linestyle='-', color='grey')

    plt.plot(linux_size, linux_freq, color='blue', marker='x',
             markerfacecolor='white', label="LINUX", linewidth=2,
             markersize=10)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    plt.xscale('log')
    # plt.title('Train')
    plt.legend(loc='best')
    # plt.grid(linestyle='-', color='grey')

    plt.plot(IMDBMulti_size, IMDBMulti_freq, color='green', marker='*',
             markerfacecolor='white', label="IMDB", linewidth=2,
             markersize=10)
    plt.xlabel("#nodes", font2)
    plt.ylabel("#graphs", font2)
    plt.xscale('log')
    # plt.title('Train')
    plt.legend(loc='best', prop={'size': 20})
    plt.grid(which=u'both',  axis=u'x', linestyle='--', color='grey')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig("line_chart.png")
    plt.savefig("line_chart.eps")
    plt.show()
    # sns.pointplot(x="sepal_length", y="species", data=iris)

