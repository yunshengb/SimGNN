import sys
sys.path.append('../')
from utils import load_as_dict, load_data, get_norm_str, get_result_path
from results import load_result
from vis import vis_attention

TRUE_MODEL = 'astar'

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


if __name__ == '__main__':
    plot_what = 'att_vis'
    concise = True
    dataset = 'aids700nef'
    model = 'astar'
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what, model)
    weight_data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/"
                               "model/Siamese/logs/"
                               "siamese_classification_aids700nef_2018-07-28T10:09:33/"
                               "test_info.pickle")
    weight = weight_data['atts']
    info_dict = {
        # draw node config
        'draw_node_size': 800 if dataset != 'linux' else 20,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 16,
        'draw_node_color_map': TYPE_COLOR_MAP,
        # draw edge config
        'draw_edge_label_enable': False,
        'edge_label_name': 'valence',
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 25,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20 if concise else 0.26,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6 if concise else 1,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    weight_max_array = []
    weight_min_array = []
    for i in range(len(weight)):
        weight_min_array.append(min(weight[i]))
        weight_max_array.append(max(weight[i]))
    weight_max = max(weight_max_array)
    weight_min = min(weight_min_array)
    print(weight_max)
    print(weight_min)
    # print("max:", weight_max)
    # print("min:", weight_min)
    # weight_max = 0.85
    # weight_min = 0.70
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    tr = load_result(dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    ids = r.sort_id_mat(True)
    q = train_data.graphs[ids[28][0]]
    gs = [test_data.graphs[47],
          train_data.graphs[ids[56][3]], train_data.graphs[ids[61][4]],
          test_data.graphs[13], test_data.graphs[67], train_data.graphs[ids[67][0]],
          test_data.graphs[99]]
    weight_query = [weight[ids[28][0]], weight[len(col_graphs) + 47],
                    weight[ids[56][3]], weight[ids[61][4]], weight[len(col_graphs) + 13],
                    weight[len(col_graphs) + 67], weight[ids[67][0]],
                    weight[len(col_graphs) + 99]]
    info_dict['each_graph_text_list'] = ['(a)', '(b)', '(c)', '(d)',
                                         '(e)', '(f)', '(g)', '(h)']
    info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
        dir, plot_what, dataset, model, 'hard', get_norm_str(True), 'png')
    info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
        dir, plot_what, dataset, model, 'hard', get_norm_str(True), 'eps')

    vis_attention(q, gs, info_dict, weight_query, weight_max, weight_min)



