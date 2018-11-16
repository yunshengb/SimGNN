import csv
import matplotlib
matplotlib.use('Agg')
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import sys
sys.path.append('../')
from utils import load_data, load_as_dict, get_result_path, \
    create_dir_if_not_exists, get_norm_str
from results import load_result
from exp import get_text_label
from vis import sorted_dict, info_dict_preprocess, draw_extra, list_safe_get, calc_subplot_size


TRUE_MODEL = 'astar'


def draw_graph(g, info_dict):
    if g is None:
        return
    pos = sorted_dict(graphviz_layout(g))
    # print(info_dict['node_label_name'])
    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        print("node_labels:", node_labels)
        color_values = [info_dict['draw_node_color_map'].get(node_label, 'yellow')
                        for node_label in node_labels.values()]
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = ['#ff6666'] * len(pos.keys())
    # print(node_labels)
    # for key, value in node_labels.items():
    #     if len(value) > 6:
    #         value = value[0:6]  # shorten the label
    #     node_labels[key] = value
    if info_dict['node_label_name'] is not None:
        nx.draw_networkx(g, pos, nodelist=pos.keys(),
                        node_color=color_values, with_labels=True,
                        node_size=info_dict['draw_node_size'], labels=node_labels,
                        font_size=info_dict['draw_node_label_font_size'])
    else:
        nx.draw_networkx(g, pos, nodelist=pos.keys(), node_color=color_values,
                         with_labels=True, node_size=info_dict['draw_node_size'],
                         labels=node_labels)

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def vis(q=None, gs=None, info_dict=None):
    plt.figure()
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph(q, info_dict)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph(gs[i], info_dict)
        draw_extra(i, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_path = info_dict['plot_save_path']
    if save_path is None or save_path == "":
        plt.show()
    else:
        sp = info_dict['plot_save_path']
        print('Saving query vis plot to {}'.format(sp))
        plt.savefig(sp, dpi=info_dict['plot_dpi'])

def read_csv(query_num):
    filename = "/home/songbian/Documents/fork/GraphEmbedding/" \
               "result/ged_imdb1kfine_astar_2018-07-24T11_51_06_qilin_15cpus.csv"
    with open(filename) as f:
        reader = csv.reader(f)
        head_row = next(reader)
        ged = []
        for row in reader:
            if int(row[0]) == query_num:
                ged.append(int(row[8]))
        ged_sort_id = sorted(range(len(ged)), key=lambda k: ged[k])
    return ged_sort_id


def get_node_types(g):
    rtn = set()
    for node in g.nodes():
        rtn.add(g.node[node]['name'])
    return rtn


if __name__ == '__main__':
    dataset = 'imdb1kfine'
    model = 'astar'
    concise = True
    ext = 'png'
    norms = [True, False]
    dir = get_result_path() + '/{}/att_vis/{}'.format(dataset, model)
    create_dir_if_not_exists(dir)
    info_dict = {
        # draw node config
        'draw_node_size': 150 if dataset != 'linux' else 10,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'name',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': {'C': '#ff6666',
                                'O': 'lightskyblue',
                                'N': 'yellowgreen',
                                'movie': '#ff6666',
                                'tvSeries': '#ff6666',
                                'actor': 'lightskyblue',
                                'actress': '#ffb3e6',
                                'director': 'yellowgreen',
                                'composer': '#c2c2f0',
                                'producer': '#ffcc99',
                                'cinematographer': 'gold'},
        # draw edge config
        'draw_edge_label_enable': False,
        'edge_label_name': 'valence',
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 8,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6 if concise else 1,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path': ''
    }
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    # r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    # tr = load_result(dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        for i in range(len(row_graphs)):
            q = test_data.graphs[i]
            ids = read_csv(i)
            gids = np.concatenate([ids[:6], ids[-1:]])
            gs = [train_data.graphs[i] for i in gids]
            q_names = get_node_types(q)
            for g in gs:
                print(i, len(q_names.intersection(get_node_types(g))), g.graph['gid'])
            info_dict['plot_save_path'] = '{}/query_vis_{}_{}_{}{}.{}'.format(
                   dir, dataset, model, i, get_norm_str(norm), ext)
            vis(q, gs, info_dict)

