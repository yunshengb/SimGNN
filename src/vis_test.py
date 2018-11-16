from vis import vis
from utils import load_data, get_root_path

info_dict = {
    # draw node config
    'draw_node_size': 10,
    'draw_node_label_enable': True,
    'node_label_name': 'type',
    'draw_node_label_font_size': 8,
    'draw_node_color_map': {'C': 'red',
           'O': 'blue',
           'N': 'green'},
   
    # draw edge config
    'draw_edge_label_enable': True,
    'edge_label_name': 'valence',
    'draw_edge_label_font_size': 6,

    # graph text info config
    'each_graph_text_list': ["testa\ntestb\ntestc", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"],
    'each_graph_text_font_size': 8,
    'each_graph_text_pos': [0.5, 1.05],  # [left, bottom], value range: [0, 1]

    # graph padding: value range: [0, 1]
    'top_space': 0.1, # out of whole graph
    'bottom_space': 0,
 
    'hbetween_space': 0.4, # out of the subgraph
    'wbetween_space': 0.01,

    # plot config
    'plot_dpi': 200,
    'plot_save_path': get_root_path() + '/temp/test_vis.png'
}



test_data = load_data('aids10k', train=False)
train_data = load_data('aids10k', train=True)
q = test_data.graphs[0]

gs = []
for i in range(1, 5):
    print(i)
    gs.append(train_data.graphs[i])

vis(q, gs, info_dict)

'''
TODO:
1. node color [done]
2. support graph-level text [done]
3. plt.save in vis [done]
4. edge color
'''
