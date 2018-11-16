#!/usr/bin/env python3
from utils import get_result_path, create_dir_if_not_exists, \
    load_data, get_ts, exec_turnoff_print, prompt, prompt_get_computer_name, \
    check_nx_version, prompt_get_cpu, format_float, get_norm_str, load_as_dict
from metrics import Metric, prec_at_ks, mean_reciprocal_rank, \
    mean_squared_error, average_time, accuracy, kendalls_tau, spearmans_rho
from distance import ged
from similarity import create_sim_kernel
from results import load_results_as_dict, load_result
from dist_calculator import get_train_train_dist_mat
from classification import get_classification_labels_from_dist_mat
import networkx as nx
from random import choice
from sklearn.manifold import TSNE

check_nx_version()
import multiprocessing as mp
import numpy as np
from pandas import read_csv
# Comment out for qilin. The following packages are not installed on the server.
# Note: qilin can only afford 10 cpus if astar is used,
#       because astar needs more memory allocation.
import matplotlib

matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
import matplotlib.pyplot as plt
from vis import vis, vis_attention, vis_small

# End of commenting out for qilin.
import seaborn as sns

BASELINE_MODELS = ['beam1', 'beam2', 'beam5', 'beam10', 'beam20', 'beam40',
                   'beam80',
                   'hungarian', 'vj']
TRUE_MODEL = 'astar'

""" Plotting. """
args1 = {'astar': {'color': 'grey'},
         'beam1': {'color': 'yellowgreen'},
         'beam2': {'color': 'gold'},
         'beam5': {'color': 'deeppink'},
         'beam10': {'color': 'blue'},
         'beam20': {'color': 'forestgreen'},
         'beam40': {'color': 'darkorange'},
         'beam80': {'color': 'cyan'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'darkcyan'},
         'graph2vec': {'color': 'darkcyan'},
         'siamese': {'color': 'red'},
         'transductive': {'color': 'red'}}
args2 = {'astar': {'marker': '*', 'facecolors': 'none', 'edgecolors': 'grey'},
         'beam1': {'marker': 'H', 'facecolors': 'none', 'edgecolors': 'yellowgreen'},
         'beam2': {'marker': '^', 'facecolors': 'none', 'edgecolors': 'gold'},
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


def getKey(item):
    return item[1]


def exp1():
    """ Run baselines on real datasets. Take a while. """
    dataset = prompt('Which dataset?')
    row_train = prompt('Train or test for row graphs? (1/0)') == '1'
    row_graphs = load_data(dataset, train=row_train).graphs
    col_train = prompt('Train or test for col graphs? (1/0)') == '1'
    col_graphs = load_data(dataset, train=col_train).graphs
    model = prompt('Which model?')
    exec_turnoff_print()
    num_cpu = prompt_get_cpu()
    real_dataset_run_helper(dataset, model, row_graphs, col_graphs, num_cpu)


def real_dataset_run_helper(dataset, model, row_graphs, col_graphs, num_cpu):
    m = len(row_graphs)
    n = len(col_graphs)
    ged_mat = np.zeros((m, n))
    time_mat = np.zeros((m, n))
    outdir = '{}/{}'.format(get_result_path(), dataset)
    create_dir_if_not_exists(outdir + '/csv')
    create_dir_if_not_exists(outdir + '/ged')
    create_dir_if_not_exists(outdir + '/time')
    computer_name = prompt_get_computer_name()
    exsiting_csv = prompt('File path to exsiting csv files?')
    exsiting_entries = load_from_exsiting_csv(exsiting_csv)
    is_symmetric = prompt('Is the ged matrix symmetric? (1/0)') == '1'
    if is_symmetric:
        assert (m == n)
    csv_fn = '{}/csv/ged_{}_{}_{}_{}_{}cpus.csv'.format(
        outdir, dataset, model, get_ts(), computer_name, num_cpu)
    file = open(csv_fn, 'w')
    print('Saving to {}'.format(csv_fn))
    print_and_log('i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,ged,lcnt,time(msec)', file)
    # Multiprocessing.
    pool = mp.Pool(processes=num_cpu)
    # print('Using {} CPUs'.format(cpu_count()))
    # Submit to pool workers.
    results = [[None] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i, j) in exsiting_entries:
                continue
            if is_symmetric and (j, i) in exsiting_entries:
                continue
            g1 = row_graphs[i]
            g2 = col_graphs[j]
            results[i][j] = pool.apply_async(
                ged, args=(g1, g2, model, True, True,))
        print_progress(i, j, m, n, 'submit: {} {} {} cpus;'.
                       format(model, computer_name, num_cpu))
    # Retrieve results from pool workers.
    for i in range(m):
        for j in range(n):
            print_progress(i, j, m, n, 'work: {} {} {} cpus;'.
                           format(model, computer_name, num_cpu))
            g1 = row_graphs[i]
            g2 = col_graphs[j]
            if results[i][j] is None:
                tmp = exsiting_entries.get((i, j))
                if tmp:
                    i_gid, j_gid, i_node, j_node, d, lcnt, t = tmp
                else:
                    assert (is_symmetric)
                    j_gid, i_gid, j_node, i_node, d, lcnt, t = exsiting_entries[(j, i)]
                assert (g1.graph['gid'] == i_gid)
                assert (g2.graph['gid'] == j_gid)
                assert (g1.number_of_nodes() == i_node)
                assert (g2.number_of_nodes() == j_node)
            else:
                d, lcnt, g1_a, g2_a, t = results[i][j].get()
                i_gid, j_gid, i_node, j_node = \
                    g1.graph['gid'], g2.graph['gid'], \
                    g1.number_of_nodes(), g2.number_of_nodes()
                assert (g1.number_of_nodes() == g1_a.number_of_nodes())
                assert (g2.number_of_nodes() == g2_a.number_of_nodes())
                exsiting_entries[(i, j)] = (i_gid, j_gid, i_node, j_node, d, lcnt, t)
            s = '{},{},{},{},{},{},{},{},{},{},{:.2f}'.format(
                i, j, g1.graph['gid'], g2.graph['gid'],
                g1.number_of_nodes(), g2.number_of_nodes(),
                g1.number_of_edges(), g2.number_of_edges(),
                d, lcnt, t)
            print_and_log(s, file)
            ged_mat[i][j] = d
            time_mat[i][j] = t
    file.close()
    save_as_np(outdir, ged_mat, time_mat, get_ts(),
               dataset, row_graphs, col_graphs, model, computer_name, num_cpu)


def post_real_dataset_run_convert_csv_to_np():
    """ Use in case only csv is generated,
        and numpy matrices need to be saved. """
    dataset = 'imdb1kcoarse'
    model = 'astar'
    row_graphs = load_data(dataset, False).graphs
    col_graphs = load_data(dataset, True).graphs
    num_cpu = 15
    computer_name = 'feilong'
    ts = '2018-07-17T13:56:32'
    outdir = '{}/{}'.format(get_result_path(), dataset)
    csv_fn = '{}/csv/ged_{}_{}_{}_{}_{}cpus.csv'.format(
        outdir, dataset, model, ts, computer_name, num_cpu)
    data = read_csv(csv_fn)
    m = len(row_graphs)
    n = len(col_graphs)
    ged_mat = np.zeros((m, n))
    time_mat = np.zeros((m, n))
    cnt = 0
    for _, row in data.iterrows():
        i = int(row['i'])
        j = int(row['j'])
        d = int(row['ged'])
        t = float(row['time(msec)'])
        ged_mat[i][j] = d
        time_mat[i][j] = t
        cnt += 1
    print(cnt)
    assert (cnt == m * n)
    save_as_np(outdir, ged_mat, time_mat, ts,
               dataset, row_graphs, col_graphs, model, computer_name, num_cpu)


def load_from_exsiting_csv(csv_fn):
    rtn = {}
    if csv_fn:
        data = read_csv(csv_fn)
        for _, row in data.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            i_gid = int(row['i_gid'])
            j_gid = int(row['j_gid'])
            i_node = int(row['i_node'])
            j_node = int(row['j_node'])
            d = int(row['ged'])
            lcnt = int(row['lcnt'])
            t = float(row['time(msec)'])
            rtn[(i, j)] = (i_gid, j_gid, i_node, j_node, d, lcnt, t)
    print('Loaded {} entries from {}'.format(len(rtn), csv_fn))
    return rtn


def save_as_np(outdir, ged_mat, time_mat, ts,
               dataset, row_graphs, col_graphs, model, computer_name, num_cpu):
    s = '{}_{}_{}_{}_{}cpus'.format(
        dataset,
        model, ts, computer_name, num_cpu)
    np.save('{}/ged/ged_ged_mat_{}'.format(outdir, s), ged_mat)
    np.save('{}/time/ged_time_mat_{}'.format(outdir, s), time_mat)


def print_progress(i, j, m, n, label):
    cur = i * n + j
    tot = m * n
    print('----- {} progress: {}/{}={:.1%}'.format(label, cur, tot, cur / tot))


def print_and_log(s, file):
    print(s)
    file.write(s + '\n')
    file.flush()


def exp2():
    """ Plot ged and time. """
    dataset = 'aids50'
    models = BASELINE_MODELS
    rs = load_results_as_dict(
        dataset, models,
        row_graphs=load_data(dataset, train=False).graphs,
        col_graphs=load_data(dataset, train=True).graphs)
    metrics = [Metric('ged', 'ged'), Metric('time', 'time (msec)')]
    for metric in metrics:
        plot_ged_time_helper(dataset, models, metric, rs)


def plot_ged_time_helper(dataset, models, metric, rs):
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)

    plt.figure(0)
    plt.figure(figsize=(16, 10))

    xs = get_test_graph_sizes(dataset)
    so = np.argsort(xs)
    xs.sort()
    for model in models:
        mat = rs[model].mat(metric.name, norm=True)
        print('plotting for {}'.format(model))
        ys = np.mean(mat, 1)[so]
        plt.plot(xs, ys, **get_plotting_arg(args1, model))
        plt.scatter(xs, ys, s=200, label=model, **get_plotting_arg(args2, model))
    plt.xlabel('query graph size')
    ax = plt.gca()
    ax.set_xticks(xs)
    plt.ylabel('average {}'.format(metric.ylabel))
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = get_result_path() + '/{}/{}/ged_{}_mat_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_plotting_arg(args, model):
    for k, v in args.items():
        if k == model:
            return v
    for k, v in args.items():
        if k in model:
            return v
    raise RuntimeError('Unknown model {} in atgs {}'.format(model, args))


def get_test_graph_sizes(dataset):
    test_data = load_data(dataset, train=False)
    return [g.number_of_nodes() for g in test_data.graphs]


def exp3():
    dataset = 'aids700nef'
    models = BASELINE_MODELS + [TRUE_MODEL]
    metric = 'prec@k'
    norms = [True, False]
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    rs = load_results_as_dict(
        dataset, models, row_graphs=row_graphs, col_graphs=col_graphs)
    from utils import load_as_dict
    d = load_as_dict(
        '/home/yba/Documents/GraphEmbedding/model/Siamese/logs/siamese_regression_aids700nef_2018-08-01T02:29:11/test_info.pickle')
    pred_r = load_result(dataset, 'siamese_test', sim_mat=d['sim_mat'], time_mat=[],
                         row_graphs=row_graphs, col_graphs=col_graphs)
    models += ['siamese_test']
    rs['siamese_test'] = pred_r
    true_result = rs[TRUE_MODEL]
    plot_preck(dataset, models, rs, true_result, metric, norms)


def plot_preck(dataset, models, rs, true_result, metric, norms,
               plot_results=True, eps_dir=None):
    """ Plot prec@k. """
    create_dir_if_not_exists('{}/{}/{}'.format(
        get_result_path(), dataset, metric))
    rtn = {}
    for norm in norms:
        _, n = true_result.m_n()
        # ks = []
        # k = 1
        # while k < n:
        #     ks.append(k)
        #     k *= 2
        # plot_preck_helper(
        #     dataset, models, rs, true_result, metric, norm, ks,
        #     True, plot_results)
        ks = range(1, n)
        d = plot_preck_helper(
            dataset, models, rs, true_result, metric, norm, ks,
            False, plot_results, eps_dir)
        rtn.update(d)
    return rtn


def plot_preck_helper(dataset, models, rs, true_result, metric, norm, ks,
                      logscale, plot_results, eps_dir):
    print_ids = []
    numbers = {}
    assert (metric[0:6] == 'prec@k')
    if len(metric) > 6:
        rm = float(metric.split('_')[1])
    else:
        rm = 0
    for model in models:
        precs = prec_at_ks(true_result, rs[model], norm, ks, rm, print_ids)
        numbers[model] = {'ks': ks, 'precs': precs}
    rtn = {'preck{}_{}'.format(get_norm_str(norm), rm): numbers}
    if not plot_results:
        return rtn
    plt.figure(figsize=(16, 10))
    for model in models:
        ks = numbers[model]['ks']
        inters = numbers[model]['precs']
        if logscale:
            pltfunc = plt.semilogx
        else:
            pltfunc = plt.plot
        pltfunc(ks, inters, **get_plotting_arg(args1, model))
        plt.scatter(ks, inters, s=200, label=shorten_name(model),
                    **get_plotting_arg(args2, model))
    plt.xlabel('k')
    # ax = plt.gca()
    # ax.set_xticks(ks)
    plt.ylabel(metric)
    plt.ylim([-0.06, 1.06])
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    kss = 'k_{}_{}'.format(min(ks), max(ks))
    bfn = 'ged_{}_{}_{}_{}{}_{}'.format(
        metric, dataset, '_'.join(models), kss, get_norm_str(norm), rm)
    dir = '{}/{}/{}'.format(get_result_path(), dataset, metric)
    create_dir_if_not_exists(dir)
    sp = dir + '/' + bfn + '.png'
    plt.savefig(sp)
    print('Saved to {}'.format(sp))
    if eps_dir:
        sp = eps_dir + '/' + bfn + '.png'
        plt.savefig(sp)
        print('Saved to {}'.format(sp))
        sp = eps_dir + '/' + bfn + '.eps'
        plt.savefig(sp)
        print('Saved to {}'.format(sp))
    return rtn


def exp4():
    exec_turnoff_print()
    dataset = 'linux'
    models = BASELINE_MODELS
    metric = 'spearmans_rho'
    norms = [True, False]
    sim_kernel = 'inverse'
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    yeta = 1.0
    thresh_poss = [2.1, 7]
    thresh_negs = [2.1, 7]
    see_acc_thresh_only = False
    rs = load_results_as_dict(
        dataset, models, row_graphs=row_graphs, col_graphs=col_graphs)
    fp = '/home/yba/Documents/GraphEmbedding/model/Siamese/logs/siamese_regression_linux_2018-08-07T14:13:01_anpm(very good)/test_info.pickle'
    if fp:
        name, siamese_result = create_siamese_result_from_test_info_pickle(
            fp, dataset, row_graphs, col_graphs)
        rs[name] = siamese_result
        models = [name]
    true_result = load_result(
        dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    # for i in range(len(norms)):
    #     p1trtr, p2trtr, p1tetr, p2tetr = get_acc_threshold_percentage(
    #         thresh_poss[i], thresh_negs[i], dataset, 'ged', 'astar', norms[i],
    #         True, true_result)
    #     print('{}{} train_train {:.2%} {:.2%}'.format(
    #         dataset, get_norm_str(norms[i]), p1trtr, p2trtr))
    #     print('{}{} test_train {:.2%} {:.2%}'.format(
    #         dataset, get_norm_str(norms[i]), p1tetr, p2tetr))
    if see_acc_thresh_only:
        return
    plot_single_number_metric(
        dataset, models, rs, true_result, metric, norms, sim_kernel, yeta,
        thresh_poss, thresh_negs)


def create_siamese_result_from_test_info_pickle(fp, dataset, row_gs, col_gs):
    name = 'siamese_test'
    d = load_as_dict(fp)
    return name, load_result(dataset, name, sim_mat=d['sim_mat'],
                             row_graphs=row_gs, col_graphs=col_gs,
                             time_mat=[])


def plot_single_number_metric(dataset, models, rs, true_result, metric, norms,
                              sim_kernel=None, yeta=None, scale=None,
                              thresh_poss=None, thresh_negs=None,
                              thresh_poss_sim=None, thresh_negs_sim=None,
                              plot_results=True,
                              eps_dir=None):
    """ Plot mrr or mse. """
    create_dir_if_not_exists('{}/{}/{}'.format(
        get_result_path(), dataset, metric))
    rtn = {}
    if norms and thresh_poss and thresh_negs:
        assert (len(norms) == len(thresh_poss) == len(thresh_negs))
    for i, norm in enumerate(norms):
        thresh_pos = thresh_poss[i] if thresh_poss else None
        thresh_neg = thresh_negs[i] if thresh_negs else None
        thresh_pos_sim = thresh_poss_sim[i] if thresh_poss_sim else None
        thresh_neg_sim = thresh_negs_sim[i] if thresh_negs_sim else None
        d = plot_single_number_metric_helper(
            dataset, models, rs, true_result, metric, norm, sim_kernel, yeta,
            scale, thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim,
            plot_results, eps_dir)
        rtn.update(d)
    return rtn


def plot_single_number_metric_helper(dataset, models, rs, true_result, metric, norm,
                                     sim_kernel, yeta, scale, thresh_pos, thresh_neg,
                                     thresh_pos_sim, thresh_neg_sim,
                                     plot_results, eps_dir):
    print_ids = []
    rtn = {}
    val_list = []
    for model in models:
        if metric == 'mrr':
            val = mean_reciprocal_rank(
                true_result, rs[model], norm, print_ids)
        elif metric == 'mse':
            val = mean_squared_error(
                true_result, rs[model], sim_kernel, yeta, scale, norm)
        elif metric == 'time':
            val = average_time(rs[model])
        elif 'acc' in metric:
            val = accuracy(
                true_result, rs[model], thresh_pos, thresh_neg,
                thresh_pos_sim, thresh_neg_sim, norm)
            pos_acc, neg_acc, acc = val
            if metric == 'pos_acc':
                val = pos_acc
            elif metric == 'neg_acc':
                val = neg_acc
            elif metric == 'acc':
                val = acc  # only the overall acc
            else:
                assert (metric == 'accall')
        elif metric == 'kendalls_tau':
            val = kendalls_tau(true_result, rs[model], norm)
        elif metric == 'spearmans_rho':
            val = spearmans_rho(true_result, rs[model], norm)
        else:
            raise RuntimeError('Unknown {}'.format(metric))
        # print('{} {}: {}'.format(metric, model, mrr_mse_time))
        rtn[model] = val
        val_list.append(val)
    rtn = {'{}{}'.format(metric, get_norm_str(norm)): rtn}
    if not plot_results:
        return rtn
    plt.figure(figsize=(16, 10))
    ind = np.arange(len(val_list))  # the x locations for the groups
    val_lists = proc_val_list(val_list)
    width = 0.35  # the width of the bars
    if len(val_lists) > 1:
        width = width - 0.02 * len(val_lists)
    for i, val_list in enumerate(val_lists):
        bars = plt.bar(ind + i * width, val_list, width)
        for i, bar in enumerate(bars):
            bar.set_color(get_plotting_arg(args1, models[i])['color'])
        autolabel(bars)
    plt.xlabel('model')
    plt.xticks(ind + (width / 2) * (len(val_lists) - 1), shorten_names(models))
    if metric == 'time':
        ylabel = 'time (msec)'
        norm = None
    elif metric == 'pos_acc':
        ylabel = 'pos_recall'
    elif metric == 'neg_acc':
        ylabel = 'neg_recall'
    elif metric == 'kendalls_tau':
        ylabel = 'Kendall\'s $\\tau$'
    elif metric == 'spearmans_rho':
        ylabel = 'Spearman\'s $\\rho$'
    else:
        ylabel = metric
    plt.ylabel(ylabel)
    if metric == 'time':
        plt.yscale('log')
    if 'acc' in metric:
        p1trtr, p2trtr = get_acc_threshold_percentage(
            thresh_pos, thresh_neg, dataset, 'ged', 'astar', norm,
            plot_labels_map=False)
        metric_addi_info = '_{}({:.2%})_{}({:.2%})'.format(
            thresh_pos, p1trtr, thresh_neg, p2trtr)
    else:
        metric_addi_info = ''
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    bfn = 'ged_{}{}_{}_{}{}'.format(
        metric, metric_addi_info,
        dataset, '_'.join(models),
        get_norm_str(norm))
    sp = get_result_path() + '/{}/{}/'.format(dataset, metric) + bfn + '.png'
    plt.savefig(sp)
    print('Saved to {}'.format(sp))
    if eps_dir:
        sp = eps_dir + '/' + bfn + '.png'
        plt.savefig(sp)
        print('Saved to {}'.format(sp))
        sp = eps_dir + '/' + bfn + '.eps'
        plt.savefig(sp)
        print('Saved to {}'.format(sp))
    return rtn


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 2., 1.005 * height,
            format_float(height), ha='center', va='bottom')


def shorten_names(models):
    return [shorten_name(model) for model in models]


def shorten_name(model):
    return '\n'.join(model.split('_'))


def get_acc_threshold_percentage(thresh_pos, thresh_neg, dataset, dist_metric,
                                 dist_algo, norm, plot_labels_map=True,
                                 true_result=None):
    trtr_dist_mat = get_train_train_dist_mat(dataset, dist_metric, dist_algo, norm)
    rtn = []
    # Plot the label heatmap.
    if plot_labels_map:
        for (gs1_str, gs2_str), dm in \
                [(('train', 'train'), trtr_dist_mat),
                 (('test', 'train'), true_result.dist_mat(norm))]:
            plot_heatmap(gs1_str, gs2_str, dm, thresh_pos, thresh_neg,
                         dataset, dist_metric, norm)
            for x in get_percentage(dm, thresh_pos, thresh_neg):
                rtn.append(x)
    else:
        for x in get_percentage(trtr_dist_mat, thresh_pos, thresh_neg):
            rtn.append(x)
    return tuple(rtn)


def get_percentage(dist_mat, thresh_pos, thresh_neg):
    dists = dist_mat.flatten()
    dists = np.sort(dists)
    a = np.searchsorted(dists, thresh_pos, side='right')  # tie inclusive
    b = np.searchsorted(dists, thresh_neg, side='right')
    return a / len(dists), 1 - b / len(dists)


def plot_heatmap(gs1_str, gs2_str, dist_mat, thresh_pos, thresh_neg,
                 dataset, dist_metric, norm):
    m, n = dist_mat.shape
    label_mat, num_poses, num_negs, _, _ = \
        get_classification_labels_from_dist_mat(
            dist_mat, thresh_pos, thresh_neg)
    title = '{} pos pairs ({:.2%})\n{} neg pairs ({:.2%})'.format(
        num_poses, num_poses / (m * n), num_negs, num_negs / (m * n))
    sorted_label_mat = np.sort(label_mat, axis=1)[:, ::-1]
    mat_str = '{}({})_{}({})_{}_{}'.format(
        gs1_str, m, gs2_str, n, thresh_pos, thresh_neg)
    fn = '{}_acc_{}_labels_heatmap_{}{}'.format(dist_metric, mat_str,
                                                dataset, get_norm_str(norm))
    dir = '{}/{}/classif_labels'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    plot_heatmap_helper(sorted_label_mat, title, '{}/{}.png'.format(dir, fn),
                        cmap='bwr')
    sorted_dist_mat = np.sort(dist_mat, axis=1)
    mat_str = '{}({})_{}({})'.format(
        gs1_str, m, gs2_str, n)
    fn = '{}_acc_{}_dist_heatmap_{}{}'.format(dist_metric, mat_str,
                                              dataset, get_norm_str(norm))
    plot_heatmap_helper(sorted_dist_mat, '', '{}/{}.png'.format(dir, fn),
                        cmap='tab20')


def plot_heatmap_helper(mat, title, filepath, cmap):
    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(mat, aspect='auto', cmap=plt.get_cmap(cmap))
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)
    ax.set_title(title, fontsize=17)
    print('Saved to {}'.format(filepath))
    plt.savefig(filepath)


def proc_val_list(val_list):
    assert (val_list)
    if type(val_list[0]) is tuple:
        rtn = [[] for _ in range(len(val_list[0]))]
        for val in val_list:
            for i, v in enumerate(val):
                rtn[i].append(v)
        return rtn
    else:
        return [val_list]


def exp5():
    """ Query visualization. """
    dataset = 'imdbmulti'
    model = 'astar'
    concise = True
    norms = [True, False]
    dir = get_result_path() + '/{}/query_vis/{}'.format(dataset, model)
    create_dir_if_not_exists(dir)
    info_dict = {
        # draw node config
        'draw_node_size': 150 if dataset != 'linux' else 10,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
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
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    tr = load_result(dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        ids = r.sort_id_mat(norm)
        m, n = r.m_n()
        num_vis = 10
        for i in range(num_vis):
            q = test_data.graphs[i]
            gids = np.concatenate([ids[i][:3], [ids[i][int(n / 2)]], ids[i][-3:]])
            gs = [train_data.graphs[j] for j in gids]
            info_dict['each_graph_text_list'] = \
                [get_text_label(dataset, r, tr, i, i, q, model, norm, True, concise)] + \
                [get_text_label(dataset, r, tr, i, j,
                                train_data.graphs[j], model, norm, False, concise) \
                 for j in gids]
            # print(info_dict['each_graph_text_list'])
            info_dict['plot_save_path_png'] = '{}/query_vis_{}_{}_{}{}.{}'.format(
                dir, dataset, model, i, get_norm_str(norm), 'png')
            info_dict['plot_save_path_eps'] = '{}/query_vis_{}_{}_{}{}.{}'.format(
                dir, dataset, model, i, get_norm_str(norm), 'eps')
            vis(q, gs, info_dict)


def get_text_label(dataset, r, tr, qid, gid, g, model, norm, is_query, concise):
    rtn = ''
    if is_query:
        # rtn += '\n\n'
        pass
    else:
        trank = tr.ranking(qid, gid, norm)
        if r.model_ == tr.model_:
            rtn += 'rank: {}\n'.format(trank)
        else:
            pass
            # ged_str = get_ged_select_norm_str(tr, qid, gid, norm)
            # rtn = 'true ged: {}\ntrue rank: {}\n'.format(
            #     ged_str, trank)
    # rtn += 'gid: {}{}'.format(g.graph['gid'], get_graph_stats_text(g, concise))
    if is_query:
        # rtn += '\nquery {}\nmodel: {}'.format(dataset, model)
        rtn += 'query: {}\n'.format(dataset)
    else:
        pass
        # ged_sim_str, ged_sim = r.dist_sim(qid, gid, norm)
        # if ged_sim_str == 'ged':
        #     ged_str = get_ged_select_norm_str(r, qid, gid, norm)
        #     rtn += '\n {}: {}\n'.format(ged_sim_str, ged_str)
        # else:
        #     rtn += '\n {}: {:.2f}\n'.format(ged_sim_str, ged_sim)
        # t = r.time(qid, gid)
        # if t:
        #     rtn += 'time: {:.2f} msec'.format(t)
        # else:
        #     rtn += 'time: -'
    return rtn


def get_ged_select_norm_str(r, qid, gid, norm):
    ged = r.dist_sim(qid, gid, norm=False)[1]
    norm_ged = r.dist_sim(qid, gid, norm=True)[1]
    if norm:
        return '{:.2f}({})'.format(norm_ged, int(ged))
    else:
        return '{}({:.2f})'.format(int(ged), norm_ged)
    # if norm:
    #     return '{:.2f} ({})'.format(norm_ged, ged)
    # else:
    #     return '{} ({:.2f})'.format(ged, norm_ged)


def get_graph_stats_text(g, concise):
    return '' if concise else '\n#nodes: {}\n#edges: {}\ndensity: {:.2f}'.format(
        g.number_of_nodes(), g.number_of_edges(), nx.density(g))


def exp6():
    """ Check similarity kernel. """
    dataset = 'linux'
    model = 'astar'
    sim_kernel_name = 'inverse'
    norms = [True, False]
    yetas1 = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    yetas2 = np.arange(0.1, 1.1, 0.1)
    yetas3 = get_gaussian_yetas(0.0001, 0.001)
    scales1 = [0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 1.6, 1.8, 2, 3, 4]
    # scales2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    result = load_result(
        dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        if sim_kernel_name == 'gaussian':
            print(yetas2)
            for yetas in [yetas1, sorted(yetas2), sorted(yetas3)]:
                sim_kernels = []
                for yeta in yetas:
                    sim_kernels.append(create_sim_kernel(sim_kernel_name, yeta=yeta))
                plot_sim_kernel(dataset, result, sim_kernels, norm)
        elif sim_kernel_name == 'inverse' or sim_kernel_name == 'exp':
            for scales in [scales1]:
                sim_kernels = []
                for scale in scales:
                    sim_kernels.append(create_sim_kernel(sim_kernel_name, scale=scale))
                plot_sim_kernel(dataset, result, sim_kernels, norm)


def get_gaussian_yetas(middle_yeta, delta_yeta):
    yetas2 = [middle_yeta]
    for i in range(1, 6):
        yetas2.append(middle_yeta + i * delta_yeta)
        yetas2.append(middle_yeta - i * delta_yeta)
    return yetas2


def plot_sim_kernel(dataset, result, sim_kernels, norm):
    dir = '{}/{}/sim'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    m, n = result.m_n()
    ged_mat = result.dist_sim_mat(norm=norm)
    plt.figure(figsize=(16, 10))
    m = 20
    n = 20
    for sim_kernel in sim_kernels:
        for i in range(m):
            for j in range(n):
                d = ged_mat[i][j]
                plt.scatter(ged_mat[i][j], sim_kernel.dist_to_sim_np(d), s=100)
        # Plot the function.
        xs, ys = get_sim_kernel_points(ged_mat, sim_kernel)
        plt.plot(xs, ys, label=sim_kernel.name())
    plt.xlabel('GED')
    plt.ylabel('Similarity')
    plt.ylim([-0.06, 1.06])
    plt.legend(loc='best')
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = '{}/sim_{}_{}{}.png'.format(
        dir, dataset, '_'.join(
            [sk.shortname() for sk in sim_kernels]), get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_sim_kernel_points(ged_mat, sim_kernel):
    xs = []
    i = 0
    while i < np.amax(ged_mat) * 1.05:
        xs.append(i)
        i += 0.2
    ys = [sim_kernel.dist_to_sim_np(x) for x in xs]
    return xs, ys


def exp7():
    """ Check symmetry of GED. """
    exec_turnoff_print()
    train_data = load_data('aids50nef', train=True)
    for i in range(10000):
        g1 = choice(train_data.graphs)
        g2 = choice(train_data.graphs)
        if g1.number_of_nodes() <= 10 and g2.number_of_nodes() <= 10:
            print(g1.number_of_nodes(), g2.number_of_nodes(),
                  g1.number_of_edges(), g2.number_of_edges())
            algo = 'astar'
            print(algo)
            d = ged(g1, g2, algo)
            print(g1.graph['gid'], g2.graph['gid'], d)
            d = ged(g2, g1, algo)
            print(g2.graph['gid'], g1.graph['gid'], d)
            algo = 'beam80'
            print(algo)
            d = ged(g1, g2, algo)
            print(g1.graph['gid'], g2.graph['gid'], d)
            d = ged(g2, g1, algo)
            print(g2.graph['gid'], g1.graph['gid'], d)
            print()


def exp8():
    # data = load_as_dict(
    # '/home/yba/Documents/GraphEmbedding/model/Siamese/logs'
    # '/siamese_classification_aids80nef_2018-07-24T22:32:17/'
    # 'test_info.pickle')
    data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/"
                        "data/2018-08-03T01:36:37_test_info.pickle")
    embs = data['embs']
    dataset = 'aids700nef'
    thresh_pos = 0.95
    thresh_neg = 0.95
    thresh_pos_sim = 0.5
    thresh_neg_sim = 0.5
    norm = True
    row_graphs = load_data(dataset, train=False).graphs
    col_graphs = load_data(dataset, train=True).graphs
    true_result = load_result(
        dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    visualize_embeddings_binary(dataset, embs, true_result, thresh_pos, thresh_neg,
                                thresh_pos_sim, thresh_neg_sim, norm)


def visualize_embeddings_binary(
        dataset, orig_embs, true_result,
        thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim,
        norm, eps_dir=None):
    label_mat, _, _ = true_result.classification_mat(
        thresh_pos, thresh_neg, thresh_pos_sim, thresh_neg_sim, norm)
    tsne = TSNE(n_components=2)
    embs = tsne.fit_transform(orig_embs)
    dir = '{}/{}/emb_vis_binary'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    m = np.shape(label_mat)[0]
    n = np.shape(label_mat)[1]
    plt_cnt = 0
    print('TSNE embeddings: {} --> {} to plot'.format(
        orig_embs.shape, embs.shape))
    for j in range(m):
        axis_x_red = []
        axis_y_red = []
        axis_x_blue = []
        axis_y_blue = []
        axis_x_query = []
        axis_y_query = []
        for i in range(n):
            if label_mat[j][i] == -1:
                axis_x_blue.append(embs[i, 0])
                axis_y_blue.append(embs[i, 1])
            else:
                axis_x_red.append(embs[i, 0])
                axis_y_red.append(embs[i, 1])
        axis_x_query.append(embs[n + j, 0])
        axis_y_query.append(embs[n + j, 1])

        plt.figure()
        plt.scatter(axis_x_red, axis_y_red, s=15, c='red', marker='o', alpha=0.6)
        plt.scatter(axis_x_blue, axis_y_blue, s=15, facecolors='none',
                    edgecolor='blue', marker='s', alpha=0.6)
        plt.scatter(axis_x_query, axis_y_query, s=400, c='limegreen', marker='P', alpha=0.6)
        plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        plt.savefig(dir + '/' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
        if eps_dir:
            plt.savefig(eps_dir + '/' + str(j) + '.png',
                        bbox_inches='tight', pad_inches=0)
            plt.savefig(eps_dir + '/' + str(j) + '.eps',
                        bbox_inches='tight', pad_inches=0)
        plt_cnt += 1
        plt.close()
    print('Saved {} embedding visualization plots'.format(plt_cnt))


# visualize embeddings gradually changing color
def exp9():
    data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/model/"
                        "Siamese/logs/"
                        "siamese_regression_aids700nef_2018-08-01T11:52:11(cur_best)/"
                        "test_info.pickle")
    embs = data['embs']
    dataset = 'aids700nef'
    model = 'astar'
    # pred_r = load_result(
    #     dataset, 'siamese', sim_mat=data['sim_mat'], time_mat=data['time_li'])
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    visualize_embeddings_gradual(dataset, embs)


def visualize_embeddings_gradual(dataset, orig_embs, eps_dir=None):
    model = 'astar'
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    tsne = TSNE(n_components=2)
    embs = tsne.fit_transform(orig_embs)
    dir = '{}/{}/emb_vis_gradual'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    m = np.shape(r.sort_id_mat_)[0]
    n = np.shape(r.sort_id_mat_)[1]
    sim_matrix = r.dist_norm_mat_
    print('sim_matrix:', sim_matrix)
    plt_cnt = 0
    print('TSNE embeddings: {} --> {} to plot'.format(
        orig_embs.shape, embs.shape))
    for i in range(m):
        axis_x_red = []
        axis_y_red = []
        axis_x_blue = []
        axis_y_blue = []
        axis_x_query = []
        axis_y_query = []
        red_number = []
        blue_number = []
        for j in range(n):
            if r.dist_norm_mat_[i][j] < 0.6:
                red_number.append((j, r.dist_norm_mat_[i][j]))
            else:
                blue_number.append((j, r.dist_norm_mat_[i][j]))
        sorted(red_number, key=lambda x:x[1])
        sorted(blue_number, key=lambda x:x[1], reverse=True)
        for j in range(len(red_number)):
            axis_x_red.append(embs[red_number[j][0], 0])
            axis_y_red.append(embs[red_number[j][0], 1])
        for j in range(len(blue_number)):
            axis_x_blue.append(embs[blue_number[j][0], 0])
            axis_y_blue.append(embs[blue_number[j][0], 1])
        axis_x_query.append(embs[i + n, 0])
        axis_y_query.append(embs[i + n, 1])

        plt.figure()
        plt.scatter(axis_x_blue, axis_y_blue, s=15,
                    c=sorted(range(len(axis_x_blue)), reverse=False),
                    marker='s', alpha=0.6, cmap=plt.cm.get_cmap("Blues"))
        plt.scatter(axis_x_red, axis_y_red, s=30,
                    c=sorted(range(len(axis_x_red)), reverse=False),
                    marker='o', alpha=0.6, cmap=plt.cm.get_cmap("Reds"))
        plt.scatter(axis_x_query, axis_y_query, s=400, c='limegreen', marker='P', alpha=0.6)
        plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        plt.tight_layout()
        print(dir + '/' + str(i) + '.png')
        plt.savefig(dir + '/' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(dir + '/' + str(i) + '.eps', bbox_inches='tight', pad_inches=0)
        if eps_dir:
            plt.savefig(eps_dir + '/' + str(i) + '.png',
                        bbox_inches='tight', pad_inches=0)
            plt.savefig(eps_dir + '/' + str(i) + '.eps',
                        bbox_inches='tight', pad_inches=0)
        plt_cnt += 1
        plt.close()
    print('Saved {} embedding visualization plots'.format(plt_cnt))


def exp10():
    """ground_truth"""
    dataset = 'aids700nef'
    model = 'astar'
    groundtruth(dataset, model)


def groundtruth(dataset, model, eps_dir=None):
    plot_what = 'att_vis_true'
    # weight_data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/model/"
    #                            "Siamese/logs/"
    #                            "siamese_regression_aids700nef_2018-08-01T11:52:11(cur_best)/"
    #                            "test_info.pickle")
    concise = True
    ext = 'eps'
    norms = [True]
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what, model)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    info_dict = {
        # draw node config
        'draw_node_size': 20 if dataset != 'linux' else 20,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
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
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        ids = r.sort_id_mat(norm)
        m, n = r.m_n()
        for i in range(len(row_graphs)):
            q = test_data.graphs[i]
            gids = np.concatenate([ids[i][:5], [ids[i][int(n / 2)]], ids[i][-1:]])
            gs = [train_data.graphs[j] for j in gids]
            info_dict['each_graph_text_list'] = \
                [get_text_label_small(r, i, i, norm, True, dataset)] + \
                [get_text_label_small(r, i, j, norm, False, dataset) \
                 for j in gids]
            info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            if eps_dir:
                info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
                info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
            vis_small(q, gs, info_dict)


def get_text_label_small(r, qid, gid, norm, is_query, dataset, gids_groundtruth):
    rtn = ''
    # if is_query:
    #     rtn += '\n\n'
    if is_query:
        if dataset == 'imdbmulti':
            rtn += 'nGED by Beam-\nHungarian-VJ'
        else:
            rtn += 'nGED by\nA*'
    else:
        ged_sim_str, ged_sim = r.dist_sim(qid, gid, norm)
        if ged_sim_str == 'ged':
            ged_str = get_ged_select_norm_str(r, qid, gid, norm)
            if gid != gids_groundtruth[6]:
                rtn += '\n {}'.format(ged_str[:4])
            else:
                rtn += '\n ...   {}   ...'.format(ged_str[:4])
        else:
            rtn += '\n {.2f}'.format(ged_sim)
    return rtn


def exp11():
    """ranking"""
    dataset = 'aids700nef'
    model = 'astar'
    pickle_path = "/home/songbian/Documents/fork/GraphEmbedding/model/Siamese/logs/" \
                  "siamese_regression_aids700nef_2018-08-01T11:52:11(cur_best)/test_info.pickle"
    emb_data = load_as_dict(pickle_path)
    pred_r = load_result(
        dataset, 'siamese', sim_mat=emb_data['sim_mat'], time_mat=emb_data['time_li'])
    ranking(dataset, model, pred_r)


def ranking(dataset, model, pred_r, eps_dir=None):
    plot_what = 'att_vis_pred'
    # weight_data = load_as_dict(pickle_path)
    concise = True
    ext = 'eps'
    norms = [True]
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what, model)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    info_dict = {
        # draw node config
        'draw_node_size': 20 if dataset != 'linux' else 20,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 20,
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
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    for norm in norms:
        ids = pred_r.sort_id_mat_
        for i in range(len(row_graphs)):
            q = test_data.graphs[i]
            gids = np.concatenate([ids[i][:5], [ids[i][int(len(col_graphs) / 2)]], ids[i][-1:]])
            gs = [train_data.graphs[j] for j in gids]
            text = ['\n\n']
            for j in gids:
                rtn = '\n {}: {:.2f}'.format('sim', pred_r.sim_mat_[i][j])
                text.append(rtn)
            info_dict['each_graph_text_list'] = text
            info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            if eps_dir:
                info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
                info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            vis_small(q, gs, info_dict)


def exp12():
    dataset = 'aids700nef'
    model = 'astar'
    weight_data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/"
                               "model/Siamese/logs/"
                               "siamese_classification_aids700nef_2018-07-28T10:09:33/"
                               "test_info.pickle")
    weight = weight_data['atts']
    draw_attention(dataset, model, weight)


def draw_attention(dataset, model, weight, eps_dir=None):
    """visualize_attentions"""
    plot_what = 'att_vis'
    concise = True
    norms = [True]
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what, model)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    info_dict = {
        # draw node config
        'draw_node_size': 150 if dataset != 'linux' else 20,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
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
    # weight_min = 0.7
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    tr = load_result(dataset, TRUE_MODEL, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        ids = r.sort_id_mat(norm)
        m, n = r.m_n()
        for i in range(len(row_graphs)):
            q = test_data.graphs[i]
            gids = ids[i][:5]
            gs = [train_data.graphs[j] for j in gids]
            weight_query = []
            weight_query.append(weight[len(col_graphs) + i])
            for j in gids:
                weight_query.append(weight[j])
            info_dict['each_graph_text_list'] = \
                [get_text_label(dataset, r, tr, i, i, q, model, norm, True, concise)] + \
                [get_text_label(dataset, r, tr, i, j,
                                train_data.graphs[j], model, norm, False, concise) \
                 for j in gids]
            info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
            if eps_dir:
                info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
                info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
            vis_attention(q, gs, info_dict, weight_query, weight_max, weight_min)


def draw_emb_hist_heat(dataset, nel, norm, max_nodes=10, apply_sigmoid=True, eps_dir=None):
    # nel: node embeddings list
    create_dir_if_not_exists('{}/{}'.format(eps_dir, 'heatmap'))
    create_dir_if_not_exists('{}/{}'.format(eps_dir, 'histogram'))
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    true_r = load_result(dataset, TRUE_MODEL,
                         row_graphs=row_graphs, col_graphs=col_graphs)
    for i in range(len(nel)):
        for j in range(len(nel[i])):
            if len(nel[i]) < max_nodes:
                nel[i] = np.pad(
                    nel[i], ((0, max_nodes - len(nel[i])), (0, 0)),
                    'constant')
    ids = true_r.sort_id_mat(norm)
    plt_cnt = 0
    for i in range(len(row_graphs)):
        gids = np.concatenate([ids[i][:1], ids[i][-1:]])
        for j in gids:
            _, d = true_r.dist_sim(i, j, norm)
            result = np.dot(nel[i], nel[j].T)
            if apply_sigmoid:
                result = sigmoid(result) - 0.5  # sigmoid - 0.5
            assert (result.shape == (max_nodes, max_nodes))
            plt.figure()
            sns_plot = sns.heatmap(result, fmt='d', cmap='Blues')
            fig = sns_plot.get_figure()
            fig.savefig('{}/{}/{}_{}_{}.png'.format(
                eps_dir, 'heatmap', i, j, d))
            fig.savefig('{}/{}/{}_{}_{}.eps'.format(
                eps_dir, 'heatmap', i, j, d))
            plt.close()
            plt_cnt += 2
            result_array = []
            for m in range(len(result)):
                for n in range(len(result[m])):
                    result_array.append(result[m][n])
            plt.figure()
            plt.xlim(-1, 1)
            plt.ylim(0, 100)
            sns_plot = sns.distplot(result_array, bins=16, color='r',
                                    kde=False, rug=False, hist=True)
            fig = sns_plot.get_figure()
            fig.savefig('{}/{}/{}_{}_{}.png'.format(
                eps_dir, 'histogram', i, j, d))
            fig.savefig('{}/{}/{}_{}_{}.eps'.format(
                eps_dir, 'histogram', i, j, d))
            plt_cnt += 2
            plt.close()
    print('Saved {} node embeddings mne plots'.format(plt_cnt))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def exp13():
    dataset = 'imdbmulti'
    model = 'astar'
    data = load_as_dict("/home/songbian/Documents/fork/GraphEmbedding/"
                        "data/imdbmulti_test_info.pickle")
    pred_r = load_result(
        dataset, 'siamese', sim_mat=data['sim_mat'], time_mat=data['time_li'])
    comb_gt_rk(dataset, model, pred_r)


def comb_gt_rk(dataset, model, pred_r, eps_dir=None):
    plot_what = 'att_vis_gt_rk'
    concise = True
    norms = [True]
    dir = get_result_path() + '/{}/{}/{}'.format(dataset, plot_what, model)
    create_dir_if_not_exists(dir)
    if eps_dir:
        create_dir_if_not_exists(eps_dir)
    info_dict = {
        # draw node config
        'draw_node_size': 20 if dataset != 'linux' else 20,
        'draw_node_label_enable': True,
        'node_label_name': None if dataset == 'linux' else 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': TYPE_COLOR_MAP,
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 10,
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
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    row_graphs = test_data.graphs
    col_graphs = train_data.graphs
    r = load_result(dataset, model, row_graphs=row_graphs, col_graphs=col_graphs)
    for norm in norms:
        ids_groundtruth = r.sort_id_mat(norm)
        ids_rank = pred_r.sort_id_mat_
        for i in range(len(row_graphs)):
            q = test_data.graphs[i]
            gids_groundtruth = np.array([ids_groundtruth[i][1],
                                         ids_groundtruth[i][2],
                                         ids_groundtruth[i][3],
                                         ids_groundtruth[i][4],
                                         ids_groundtruth[i][7],
                                         ids_groundtruth[i][int(len(col_graphs) / 2)],
                                         ids_groundtruth[i][-1:]])
            if dataset == 'aids700nef':
                gids_rank = np.array([ids_rank[i][0], ids_rank[i][1],
                                  ids_rank[i][2], ids_rank[i][4],
                                  ids_rank[i][3], ids_rank[i][5],
                                  ids_rank[i][int(len(col_graphs) / 2)],
                                  ids_rank[i][-1:]])
            else:
                gids_rank = np.concatenate([ids_rank[i][:6],
                                           [ids_rank[i][int(len(col_graphs) / 2)]],
                                            ids_rank[i][-1:]])
            gs_rank = [test_data.graphs[i]]
            gs_rank = gs_rank + [train_data.graphs[j] for j in gids_rank]
            gs_groundtruth = [test_data.graphs[i]]
            gs_groundtruth = gs_groundtruth + [train_data.graphs[j] for j in gids_groundtruth]
            gs = gs_groundtruth + gs_rank
            # text = ['\n\n']
            gids_groundtruth = np.array([ids_groundtruth[i][0],
                                         ids_groundtruth[i][1],
                                         ids_groundtruth[i][2],
                                         ids_groundtruth[i][3],
                                         ids_groundtruth[i][4],
                                         ids_groundtruth[i][7],
                                         ids_groundtruth[i][int(len(col_graphs) / 2)],
                                         ids_groundtruth[i][-1:]])
            text = []
            text = text + [get_text_label_small(r, i, i, norm, True, dataset, gids_groundtruth)] + \
                   [get_text_label_small(r, i, j, norm, False, dataset, gids_groundtruth) for j in gids_groundtruth]
            text.append("Rank by\nSimGNN")
            for j in range(len(gids_rank)):
                if j == len(gids_rank) - 2:
                    rtn = '\n ...   {}   ...'.format(int(len(col_graphs) / 2))
                elif j == len(gids_rank) - 1:
                    rtn = '\n {}'.format(int(len(col_graphs)))
                else:
                    rtn = '\n {}'.format(str(j+1))
                # rtn = '\n {}: {:.2f}'.format('sim', pred_r.sim_mat_[i][j])
                text.append(rtn)
            info_dict['each_graph_text_list'] = text
            info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
            if eps_dir:
                info_dict['plot_save_path_eps'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'eps')
                info_dict['plot_save_path_png'] = '{}/{}_{}_{}_{}{}.{}'.format(
                    eps_dir, plot_what, dataset, model, i, get_norm_str(norm), 'png')
            print(info_dict['plot_save_path_png'])
            vis_small(q, gs, info_dict)


if __name__ == '__main__':
    exp9()
