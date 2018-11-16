from config import FLAGS
from utils_siamese import convert_msec_to_sec_str, get_model_info_as_str, need_val
from time import time
import numpy as np
from collections import OrderedDict


def train_val_loop(data, eval, model, saver, sess):
    train_costs, train_times, val_results_dict = [], [], OrderedDict()
    print('Optimization Started!')
    for iter in range(FLAGS.iters):
        iter += 1
        # Train.
        feed_dict = model.get_feed_dict_for_train(data)
        train_cost, train_time = run_tf(
            feed_dict, model, saver, sess, 'train', iter=iter)
        train_costs.append(train_cost)
        train_times.append(train_time)

        # Validate.
        val_s = ''
        if need_val(iter):
            t = time()
            val_results, val_s = val(data, eval, model, saver, sess, iter)
            val_time = time() - t
            val_s += ' time={}'.format(convert_msec_to_sec_str(val_time))
            val_results_dict[iter] = val_results
            model.save(sess, saver, iter)

        print('Iter:{:04n} train_loss={:.5f} time={} {}'.format(
            iter, train_cost, convert_msec_to_sec_str(train_time), val_s))

        # if iter > FLAGS.early_stopping:
        #     most_recent = val_result_list[-1]
        #     moving_avg = np.mean(
        #         val_result_list[
        #         -(FLAGS.early_stopping // FLAGS.iters_val + 1):-1])
        #     delta = abs(most_recent - moving_avg)
        #     # if delta < 0.001:
        #     print("Early stopping: |{:.2f}-{:.2f}|={:.2f} < 0.001 ...".format(
        #         most_recent, moving_avg, delta))
        #     #     break

    print('Optimization Finished!')
    saver.save_train_val_info(train_costs, train_times, val_results_dict)
    return train_costs, train_times, val_results_dict


def val(data, eval, model, saver, sess, iter):
    gs1, gs2 = eval.get_val_gs_as_tuple(data)
    sim_mat, loss_list, time_list = run_pairs_for_val_test(
        gs1, gs2, eval, model, saver, sess, 'val')
    val_results, val_s = eval.eval_for_val(sim_mat, loss_list, time_list,
                                           model.get_eval_metrics_for_val())
    saver.log_val_results(val_results, iter)
    return val_results, val_s  # TODO: tensorboard


def test(data, eval, model, saver, sess, val_results_dict):
    best_iter = model.find_load_best_model(sess, saver, val_results_dict)
    saver.clean_up_saved_models(best_iter)
    gs1, gs2 = eval.get_test_gs_as_tuple(data)
    sim_mat, loss_list, time_list = run_pairs_for_val_test(
        gs1, gs2, eval, model, saver, sess, 'test')
    node_embs_list, graph_embs_mat, emb_time = collect_embeddings(
        gs1, gs2, model, saver, sess)
    attentions = collect_attentions(gs1, gs2, model, saver, sess)
    print('Evaluating...')
    results = eval.eval_for_test(
        sim_mat, loss_list, time_list, node_embs_list, graph_embs_mat, attentions,
        model.get_eval_metrics_for_test(), saver)
    if not FLAGS.plot_results:
        pretty_print_dict(results)
    print('Results generated with {} metrics; collecting embeddings'.format(
        len(results)))
    print(get_model_info_as_str())
    saver.save_test_info(
        sim_mat, time_list, best_iter, results, node_embs_list, graph_embs_mat,
        emb_time, attentions)
    return best_iter, results


def run_pairs_for_val_test(row_graphs, col_graphs, eval, model, saver,
                           sess, val_or_test):
    m = len(row_graphs)
    n = len(col_graphs)
    sim_mat = np.zeros((m, n))
    time_list = []
    loss_list = []
    print_count = 0
    flush = True
    for i in range(m):
        for j in range(n):
            g1 = row_graphs[i]
            g2 = col_graphs[j]
            true_sim = eval.get_true_sim(i, j, val_or_test, model)
            if true_sim is None:
                continue
            feed_dict = model.get_feed_dict_for_val_test(g1, g2, true_sim)
            (loss_i_j, sim_i_j), test_time = run_tf(
                feed_dict, model, saver, sess, val_or_test)
            if flush:
                (loss_i_j, sim_i_j), test_time = run_tf(
                    feed_dict, model, saver, sess, val_or_test)
                flush = False
            test_time *= 1000
            if val_or_test == 'test' and print_count < 100:
                print('{},{},{:.2f}mec,{:.4f},{:.4f}'.format(
                    i, j, test_time, sim_i_j, true_sim))
                print_count += 1
            sim_mat[i][j] = sim_i_j
            loss_list.append(loss_i_j)
            time_list.append(test_time)
    return sim_mat, loss_list, time_list


def run_tf(feed_dict, model, saver, sess, tvt, iter=None):
    if tvt == 'train':
        objs = [model.opt_op, model.train_loss]
    elif tvt == 'val':
        objs = [model.val_test_loss, model.pred_sim_without_act()]
    elif tvt == 'test':
        objs = [model.pred_sim_without_act()]
    elif tvt == 'test_emb':
        objs = [model.node_embeddings, model.graph_embeddings]
    elif tvt == 'test_att':
        objs = [model.attentions]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt, iter)
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'train':
        rtn = outs[-1]
    elif tvt == 'val' or tvt == 'test':
        np_result = model.apply_final_act_np(outs[-1])
        if tvt == 'val':
            rtn = (outs[-2], np_result)
        else:
            rtn = (0, np_result)
    elif tvt == 'test_emb':
        rtn = (outs[-2], outs[-1])
    else:
        rtn = outs[-1]
    return rtn, time_rtn


def collect_embeddings(test_gs, train_gs, model, saver, sess):
    assert (hasattr(model, 'node_embeddings'))
    if not hasattr(model, 'graph_embeddings'):
        return None, None
    all_gs = train_gs + test_gs
    node_embs_list = []
    graph_embs_mat = np.zeros((len(all_gs), model.embed_dim))
    emb_time_list = []
    for i, g in enumerate(all_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0)
        rtn, t = run_tf(
            feed_dict, model, saver, sess, 'test_emb')
        t *= 1000
        emb_time_list.append(t)
        node_embs, graph_embs = rtn
        assert (len(node_embs) == 2 and len(graph_embs) == 2)
        node_embs_list.append(node_embs[0])
        graph_embs_mat[i] = graph_embs[0]
    emb_time = np.mean(emb_time_list)
    print('graph embedding {:.5f}'.format(emb_time))
    print(graph_embs_mat[0])
    return node_embs_list, graph_embs_mat, emb_time


def collect_attentions(test_gs, train_gs, model, saver, sess):
    if not hasattr(model, 'attentions'):
        return None
    all_gs = train_gs + test_gs
    rtn = []
    for i, g in enumerate(all_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0)
        atts, _ = run_tf(
            feed_dict, model, saver, sess, 'test_att')
        assert (atts.shape[1] == 1)
        rtn.append(atts)
    print('attention')
    print(rtn[0])
    return rtn


def pretty_print_dict(d, indent=0):
    for key, value in sorted(d.items()):
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
