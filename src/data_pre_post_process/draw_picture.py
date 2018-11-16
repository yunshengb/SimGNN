import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib as mpl
import numpy as np
from bs4 import BeautifulSoup as Soup
sys.path.append('../')
from utils import load_data

# colors = ['#ff6666', '#ffcc99', '#99ff99', '#66b3ff', 'coral', 'lightskyblue', 'gold', 'yellowgreen', 'deeppink', 'brown', \
#           'royalblue', 'darkorange']

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

colors = ['#ff6666', 'lightskyblue', 'yellowgreen', 'yellow']
# colors = ['#66b3ff', '#ff6666', 'yellowgreen', 'gold', '#c2c2f0', '#ffb3e6', '#ffcc99', '#99ff99']

def StatisticData():
    aids_test_count = 0
    aids_train_count = 0
    aids_test_nodes = {}
    aids_train_nodes = {}
    linux_test_count = 0
    linux_train_count = 0
    linux_test_nodes = {}
    linux_train_nodes = {}
    IMDB1kCoarse_test_count = 0
    IMDB1kCoarse_train_count = 0
    IMDB1kCoarse_test_nodes = {}
    IMDB1kCoarse_train_nodes = {}
    IMDB1kFine_test_count = 0
    IMDB1kFine_train_count = 0
    IMDB1kFine_test_nodes = {}
    IMDB1kFine_train_nodes = {}

    for filename in os.listdir("../../data/AIDS700nef/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/AIDS700nef/train/' + filename
            G = nx.read_gexf(f)
            aids_train_nodes[filename] = len(G)
            aids_train_count = aids_train_count + 1

    for filename in os.listdir("../../data/AIDS700nef/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/AIDS700nef/test/' + filename
            G = nx.read_gexf(f)
            aids_test_nodes[filename] = len(G)
            aids_test_count = aids_test_count + 1

    for filename in os.listdir("../../data/linux/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/linux/train/' + filename
            G = nx.read_gexf(f)
            linux_train_nodes[filename] = len(G)
            linux_train_count = linux_train_count + 1

    for filename in os.listdir("../../data/linux/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/linux/test/' + filename
            G = nx.read_gexf(f)
            linux_test_nodes[filename] = len(G)
            linux_test_count = linux_test_count + 1

    for filename in os.listdir("../../data/IMDB1kCoarse/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDB1kCoarse/train/' + filename
            G = nx.read_gexf(f)
            IMDB1kCoarse_train_nodes[filename] = len(G)
            IMDB1kCoarse_train_count = IMDB1kCoarse_train_count + 1

    for filename in os.listdir("../../data/IMDB1kCoarse/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDB1kCoarse/test/' + filename
            G = nx.read_gexf(f)
            IMDB1kCoarse_test_nodes[filename] = len(G)
            IMDB1kCoarse_test_count = IMDB1kCoarse_test_count + 1

    for filename in os.listdir("../../data/IMDB1kFine/train/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDB1kFine/train/' + filename
            G = nx.read_gexf(f)
            IMDB1kFine_train_nodes[filename] = len(G)
            IMDB1kFine_train_count = IMDB1kFine_train_count + 1

    for filename in os.listdir("../../data/IMDB1kFine/test/"):
        if filename.endswith(".gexf"):
            f = '../../data/IMDB1kFine/test/' + filename
            G = nx.read_gexf(f)
            IMDB1kFine_test_nodes[filename] = len(G)
            IMDB1kFine_test_count = IMDB1kCoarse_test_count + 1

    return aids_train_count, aids_train_nodes, aids_test_count, aids_test_nodes, \
           linux_train_count, linux_train_nodes, linux_test_count, linux_test_nodes, \
           IMDB1kCoarse_train_count, IMDB1kCoarse_train_nodes, IMDB1kCoarse_test_count, IMDB1kCoarse_test_nodes, \
           IMDB1kFine_train_count, IMDB1kFine_train_nodes, IMDB1kFine_test_count, IMDB1kFine_test_nodes

if __name__ == '__main__':
    aids_train_count, aids_train_nodes, aids_test_count, aids_test_nodes, \
    linux_train_count, linux_train_nodes, linux_test_count, linux_test_nodes, \
    IMDB1kCoarse_train_count, IMDB1kCoarse_train_nodes, IMDB1kCoarse_test_count, IMDB1kCoarse_test_nodes, \
    IMDB1kFine_train_count, IMDB1kFine_train_nodes, IMDB1kFine_test_count, IMDB1kFine_test_nodes = StatisticData()

    aids_train_list = list(aids_train_nodes.values())
    aids_test_list = list(aids_test_nodes.values())
    linux_train_list = list(linux_train_nodes.values())
    linux_test_list = list(linux_test_nodes.values())
    IMDB1kCoarse_train_list = list(IMDB1kCoarse_train_nodes.values())
    IMDB1kCoarse_test_list = list(IMDB1kCoarse_test_nodes.values())
    IMDB1kFine_train_list = list(IMDB1kFine_train_nodes.values())
    IMDB1kFine_test_list = list(IMDB1kFine_test_nodes.values())

    max_aids_train = max(aids_train_list)
    max_linux_train = max(linux_train_list)
    max_aids_test = max(aids_test_list)
    max_linux_test = max(linux_test_list)
    max_IMDB1kCoarse_train = max(IMDB1kCoarse_train_list)
    max_IMDB1kFine_train = max(IMDB1kFine_train_list)
    max_IMDB1kCoarse_test = max(IMDB1kCoarse_test_list)
    max_IMDB1kFine_test = max(IMDB1kFine_test_list)

    # min_aids_train = min(aids_train_list)
    # min_aids_test = min(aids_test_list)

    aids_train_size = list(range(0, max_aids_train + 1))
    aids_test_size = list(range(0, max_aids_test + 1))
    linux_train_size = list(range(0, max_linux_train + 1))
    linux_test_size = list(range(0, max_linux_test + 1))
    IMDB1kCoarse_train_size = list(range(0, max_IMDB1kCoarse_train + 1))
    IMDB1kCoarse_test_size = list(range(0, max_IMDB1kCoarse_test + 1))
    IMDB1kFine_train_size = list(range(0, max_IMDB1kFine_train + 1))
    IMDB1kFine_test_size = list(range(0, max_IMDB1kFine_test + 1))

    aids_train_freq = [0] * (max_aids_train + 1)
    linux_train_freq = [0] * (max_linux_train + 1)
    IMDB1kCoarse_train_freq = [0] * (max_IMDB1kCoarse_train + 1)
    IMDB1kFine_train_freq = [0] * (max_IMDB1kFine_train + 1)

    for num in aids_train_list:
        aids_train_freq[num] = aids_train_freq[num] + 1

    for num in linux_train_list:
        linux_train_freq[num] = linux_train_freq[num] + 1

    for num in IMDB1kCoarse_train_list:
        IMDB1kCoarse_train_freq[num] = IMDB1kCoarse_train_freq[num] + 1

    for num in IMDB1kFine_train_list:
        IMDB1kFine_train_freq[num] = IMDB1kFine_train_freq[num] + 1

    aids_test_freq = [0] * (max_aids_test + 1)
    linux_test_freq = [0] * (max_linux_test + 1)
    IMDB1kCoarse_test_freq = [0] * (max_IMDB1kCoarse_test + 1)
    IMDB1kFine_test_freq = [0] * (max_IMDB1kFine_test + 1)

    for num in aids_test_list:
        aids_test_freq[num] = aids_test_freq[num] + 1

    for num in linux_test_list:
        linux_test_freq[num] = linux_test_freq[num] + 1

    for num in IMDB1kCoarse_test_list:
        IMDB1kCoarse_test_freq[num] = IMDB1kCoarse_test_freq[num] + 1

    for num in IMDB1kFine_test_list:
        IMDB1kFine_test_freq[num] = IMDB1kFine_test_freq[num] + 1

    # draw line chart
    plt.plot(aids_train_size, aids_train_freq, 'ro-', markerfacecolor='white', label="aids_train", linewidth=0.5, markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    plt.plot(aids_test_size, aids_test_freq, 'ro--', markerfacecolor='white', label="aids_test", linewidth=0.5, markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    plt.plot(linux_train_size, linux_train_freq, 'bo-', markerfacecolor='white', label="linux_train", linewidth=0.5,
             markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    # print("linux_test:", linux_test_size, linux_test_freq)
    plt.plot(linux_test_size, linux_test_freq, 'bo--', markerfacecolor='white', label="linux_test", linewidth=0.5,
             markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    plt.plot(IMDB1kCoarse_train_size, IMDB1kCoarse_train_freq, 'go-', markerfacecolor='white', label="IMDB_train", linewidth=0.5,
             markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    plt.plot(IMDB1kCoarse_test_size, IMDB1kCoarse_test_freq, 'go--', markerfacecolor='white', label="IMDB_test", linewidth=0.5,
             markersize=5)
    plt.xlabel("#nodes")
    plt.ylabel("#graphs")
    # plt.title('Train')
    plt.legend(loc='best')
    plt.grid(linestyle='-', color='grey')

    # plt.plot(IMDB1kFine_train_size, IMDB1kFine_train_freq, 'mo-', markerfacecolor='white', label="IMDB1kFine_train", linewidth=0.5,
    #          markersize=5)
    # plt.xlabel("#nodes")
    # plt.ylabel("#graphs")
    # # plt.title('Train')
    # plt.legend(loc='best')
    # plt.grid(linestyle='-', color='grey')
    #
    # plt.plot(IMDB1kFine_test_size, IMDB1kFine_test_freq, 'mo--', markerfacecolor='white', label="IMDB1kFine_test", linewidth=0.5,
    #          markersize=5)
    # plt.xlabel("#nodes")
    # plt.ylabel("#graphs")
    # # plt.title('Train')
    # plt.legend(loc='best')
    # plt.grid(linestyle='-', color='grey')
    plt.savefig("line_chart.eps")
    plt.show()

    node_label_dict = {}
    edge_label_dict = {}
    count = 0

    for filename in os.listdir("../../data/AIDS700nef/test/"):
        if filename.endswith(".gexf"):
            f = "../../data/AIDS700nef/test/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                # print(node_attrs)
                node_label = node_attrs['value']
                # print(node_label)
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            for edge in soup.findAll('edge'):
                edge_attvalues = edge.findAll('attvalues')[0]
                edge_attvalue = edge_attvalues.findAll('attvalue')[0]
                edge_attrs = dict(edge_attvalue.attrs)
                edge_label = edge_attrs['value']
                if edge_label in edge_label_dict:
                    edge_label_dict[edge_label] += 1
                else:
                    edge_label_dict[edge_label] = 1
            count += 1

    for filename in os.listdir("../../data/AIDS700nef/train/"):
        if filename.endswith(".gexf"):
            f = "../../data/AIDS700nef/train/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                node_label = node_attrs['value']
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            for edge in soup.findAll('edge'):
                edge_attvalues = edge.findAll('attvalues')[0]
                edge_attvalue = edge_attvalues.findAll('attvalue')[0]
                edge_attrs = dict(edge_attvalue.attrs)
                edge_label = edge_attrs['value']
                if edge_label in edge_label_dict:
                    edge_label_dict[edge_label] += 1
                else:
                    edge_label_dict[edge_label] = 1
            count += 1
    print(node_label_dict)
    node_label_list = sorted(node_label_dict.items(), key=lambda item: item[1], reverse=True)
    print(node_label_list)
    # print([(k, node_label_dict[k]) for k in sorted(node_label_dict.keys())])
    print(len([key for key in node_label_dict.keys()]))
    labels = []
    sizes = []
    for i in range(len(node_label_list)):
        labels.append(node_label_list[i][0])
        sizes.append(node_label_list[i][1])
    # for label in node_label_dict.keys():
    #     labels.append(label)
    # for size in node_label_dict.values():
    #     sizes.append(size)
    explode = [0] * len(labels)
    count_other = 0
    for i in sizes[:]:
        if i < 500:
            count_other = count_other + i
            index_del = sizes.index(i)
            sizes.remove(i)
            del explode[index_del]
            del labels[index_del]
    sizes.append(count_other)
    explode.append(0)
    labels.append('Other')
    # plt.axis(aspect=1)
    # draw pie chart
    plt.axes(aspect='equal')
    patches, texts, autotexts = plt.pie(x=sizes, colors=colors[0:len(sizes)], explode=explode, labels=labels, \
            pctdistance=0.6, autopct='%1.1f%%', textprops={'fontsize': 15, 'color': 'k'}, \
            center=(0.0, 0.0), radius=1.4, shadow=False, labeldistance=1.1)
    proptease = fm.FontProperties()
    proptease.set_size('xx-large')
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)
    plt.savefig("pie_chart_1.eps", format='eps', dpi=1000)
    plt.show()

    node_label_dict = {}
    count = 0

    # draw pie chart
    for filename in os.listdir("../../data/IMDB1kCoarse/test/"):
        if filename.endswith(".gexf"):
            f = "../../data/IMDB1kCoarse/test/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                node_label = node_attrs['value']
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            count += 1

    for filename in os.listdir("../../data/IMDB1kCoarse/train/"):
        if filename.endswith(".gexf"):
            f = "../../data/IMDB1kCoarse/train/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                node_label = node_attrs['value']
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            count += 1
    print(node_label_dict)
    node_label_list = sorted(node_label_dict.items(), key=lambda item: item[1], reverse=True)
    print(node_label_list)
    # print([(k, node_label_dict[k]) for k in sorted(node_label_dict.keys())])
    print(len([key for key in node_label_dict.keys()]))
    labels = []
    sizes = []
    for i in range(len(node_label_list)):
        labels.append(node_label_list[i][0])
        sizes.append(node_label_list[i][1])
    # for label in node_label_dict.keys():
    #     labels.append(label)
    # for size in node_label_dict.values():
    #     sizes.append(size)

    # print(labels)
    # print(sizes)
    explode = [0] * len(labels)
    count_other = 0
    for i in sizes[:]:
        if i < 200:
            count_other = count_other + i
            index_del = sizes.index(i)
            sizes.remove(i)
            del explode[index_del]
            del labels[index_del]
    sizes.append(count_other)
    explode.append(0)
    labels.append('Other')
    plt.axes(aspect='equal')
    plt.pie(x=sizes, colors=colors[0:len(sizes)], explode=explode, labels=labels, \
            pctdistance=0.6, autopct='%3.1f%%', textprops={'fontsize': 15, 'color':'k'},\
            center=(0.0, 0.0), radius=1.2, shadow=False, labeldistance=1.1)
    plt.savefig("pie_chart_2.eps", format='eps', dpi=1000)
    plt.show()

    node_label_dict = {}
    count = 0

    # draw pie chart
    for filename in os.listdir("../../data/IMDB1kFine/test/"):
        if filename.endswith(".gexf"):
            f = "../../data/IMDB1kFine/test/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                node_label = node_attrs['value']
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            count += 1

    for filename in os.listdir("../../data/IMDB1kFine/train/"):
        if filename.endswith(".gexf"):
            f = "../../data/IMDB1kFine/train/" + filename
            handler = open(f).read()
            soup = Soup(handler, "xml")
            for node in soup.findAll('node'):
                node_attvalues = node.findAll('attvalues')[0]
                node_attvalue = node_attvalues.findAll('attvalue')[0]
                node_attrs = dict(node_attvalue.attrs)
                node_label = node_attrs['value']
                if node_label in node_label_dict:
                    node_label_dict[node_label] += 1
                else:
                    node_label_dict[node_label] = 1
            count += 1
    print(node_label_dict)
    labels = []
    sizes = []
    for label in node_label_dict.keys():
        labels.append(label)
    for size in node_label_dict.values():
        sizes.append(size)
    # print(labels)
    # print(sizes)
    explode = [0] * len(labels)
    count_other = 0

    for i in sizes[:]:
        if i < 22:
            count_other = count_other + i
            index_del = sizes.index(i)
            sizes.remove(i)
            del explode[index_del]
            del labels[index_del]
    sizes.append(count_other)
    explode.append(0)
    labels.append('Other')
    plt.pie(x=sizes, colors=colors[0:len(sizes)], explode=explode, labels=labels, \
            pctdistance=1.5, shadow=False, radius=0.8, textprops={'fontsize': 0})
    plt.legend(loc='upper right')
    plt.savefig("pie_chart_3.png")
    plt.show()
