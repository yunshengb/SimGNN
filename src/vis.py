import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import OrderedDict
import math
from colour import Color


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
    save_figs(info_dict)


def draw_graph(g, info_dict):
    if g is None:
        return
    pos = sorted_dict(graphviz_layout(g))
    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        color_values = [info_dict['draw_node_color_map'].get(node_label, 'yellow')
                        for node_label in node_labels.values()]
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = ['#ff6666'] * len(pos.keys())
    for key, value in node_labels.items():
        if len(value) > 6:
            value = value[0:6]  # shorten the label
        node_labels[key] = value
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


def vis_attention(q=None, gs=None, info_dict=None, weight=None, weight_max=0, weight_min=0):
    plt.figure(figsize=(12, 8))
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph_attention(q, info_dict, weight[0], weight_max, weight_min)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph_attention(gs[i], info_dict, weight[i + 1], weight_max, weight_min)
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
    save_figs(info_dict)


def vis_small(q=None, gs=None, info_dict=None):
    plt.figure(figsize=(8, 3))
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size_small(graph_num)

    # draw query graph
    info_dict['each_graph_text_font_size'] = 9
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph_small(q, info_dict)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    info_dict['each_graph_text_font_size'] = 12
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph_small(gs[i], info_dict)
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
    save_figs(info_dict)


def draw_graph_attention(g, info_dict, weight, weight_max, weight_min):
    if g is None:
        return
    pos = sorted_dict(graphviz_layout(g))
    weight_array = []
    for i in range(len(weight)):
        weight_array.append(weight[i][0])
    for i in range(len(weight_array)):
        if weight_array[i] > weight_max:
            weight_array[i] = weight_max
        if weight_array[i] < weight_min:
            weight_array[i] = weight_min
        weight_array[i] = (weight_array[i] - weight_min) / (weight_max - weight_min)

    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        # from light to dark
        # color_values = sorted(range(len(weight_array)), key=lambda k: weight_array[k])
        red = Color("red")
        white = Color("white")
        color_list = list(white.range_to(red, 10))
        # print(len(color_list))
        # print(color_list)
        # print(color_list[1])
        color_values = []
        for i in range(len(pos.keys())):
            # print(int(weight_array[i] * 10))
            color_values.append(color_list[int(weight_array[i] * 10)].hex_l)
        # for i in range(len(pos.keys())):
        #     if len(str(hex(int(255 - 255 * weight_array[i])))) <= 3:
        #         if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
        #             color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
        #                                 str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
        #         else:
        #             color_values.append("#" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
        #                                 str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
        #     else:
        #         if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
        #             color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] +
        #                                 str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
        #         else:
        #             color_values.append("#" + str(hex(int(255 - 255 * weight_array[i])))[2:] +
        #                                 str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = []
        for i in range(len(pos.keys())):
            if len(str(hex(int(255 - 255 * weight_array[i])))) <= 3:
                if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
                    color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
                else:
                    color_values.append("#" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] + "0" +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
            else:
                if len(str(hex(int(80 + 255 - 255 * weight_array[i])))) < 3:
                    color_values.append("#0" + str(hex(int(80 + 255 - 255 * weight_array[i])))[2:] +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
                else:
                    color_values.append("#" + str(hex(int(255 - 255 * weight_array[i])))[2:] +
                                        str(hex(int(255 - 255 * weight_array[i])))[2:] + "ff")
    for key, value in node_labels.items():
        if len(value) > 6:
            value = value[0:6]  # shorten the label
        node_labels[key] = value
    if info_dict['node_label_name'] is not None:
        # cmap=plt.cm.Blues or cmap=plt.cm.PuBu or cmap=plt.Reds it depends on you
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


def draw_graph_small(g, info_dict):
    if g is None:
        return

    pos = sorted_dict(graphviz_layout(g))

    if info_dict['node_label_name'] is not None:
        node_labels = sorted_dict(nx.get_node_attributes(g, info_dict['node_label_name']))
        color_values = [info_dict['draw_node_color_map'].get(node_label, 'yellow')
                        for node_label in node_labels.values()]
    else:
        node_labels = {}
        for i in range(len(pos.keys())):
            node_labels[str(i)] = ''
        node_labels = sorted_dict(node_labels)
        color_values = ['lightskyblue'] * len(pos.keys())

    for key, value in node_labels.items():
        node_labels[key] = ''
    # print(pos)
    if info_dict['node_label_name'] is not None:
        # cmap=plt.cm.Blues or cmap=plt.cm.PuBu or cmap=plt.Reds it depends on you
        nx.draw_networkx(g, pos, nodelist=pos.keys(),
                         node_color=color_values, with_labels=True,
                         node_size=info_dict['draw_node_size'], labels=node_labels,
                         linewidths=5.0)
    else:
        nx.draw_networkx(g, pos, nodelist=pos.keys(), node_color=color_values,
                         with_labels=True, node_size=info_dict['draw_node_size'],
                         labels=node_labels, linewidths=5.0)

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def calc_subplot_size(area):
    h = int(math.sqrt(area))
    while area % h != 0:
        area += 1
    w = area / h
    return [h, w]


def calc_subplot_size_small(area):
    w = int(area)
    return [2, int(w/2)]


def list_safe_get(l, index, default):
    try:
        return l[index]
    except IndexError:
        return default


def draw_extra(i, ax, info_dict, text):
    left = list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
    bottom = list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)
    # print(left, bottom)
    ax.title.set_position([left, bottom])
    ax.set_title(text, fontsize=info_dict['each_graph_text_font_size'])
    plt.axis('off')


def info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('node_label_name', '')
    info_dict.setdefault('draw_node_label_font_size', 6)

    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('edge_label_name', '')
    info_dict.setdefault('draw_edge_label_font_size', 6)

    info_dict.setdefault('each_graph_text_font_size', "")
    info_dict.setdefault('each_graph_text_pos', [0.5, 0.8])

    info_dict.setdefault('plot_dpi', 200)
    info_dict.setdefault('plot_save_path', "")

    info_dict.setdefault('top_space', 0.08)
    info_dict.setdefault('bottom_space', 0)
    info_dict.setdefault('hbetween_space', 0.5)
    info_dict.setdefault('wbetween_space', 0.01)


def sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn


def save_figs(info_dict):
    save_path = info_dict['plot_save_path_eps']
    if save_path is None or save_path == "":
        # print('plt.show')
        plt.show()
    else:
        sp = info_dict['plot_save_path_png']
        # print('Saving query vis plot to {}'.format(sp))
        plt.savefig(sp, dpi=info_dict['plot_dpi'])
        sp = info_dict['plot_save_path_eps']
        # print('Saving query vis plot to {}'.format(sp))
        plt.savefig(sp, dpi=info_dict['plot_dpi'])
    plt.close()