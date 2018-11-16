from __future__ import division
from __future__ import print_function

from config import FLAGS
from train import train_val_loop, test
from utils_siamese import get_model_info_as_str, \
    check_flags, convert_long_time_to_str
from data_siamese import SiameseModelData
from dist_calculator import DistCalculator
from models_factory import create_model
from saver import Saver
from eval import Eval
import tensorflow as tf
from time import time
import os


def main():
    t = time()
    check_flags()
    print(get_model_info_as_str())
    data = SiameseModelData()
    dist_calculator = DistCalculator(
        FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo)
    model = create_model(FLAGS.model, data.input_dim(), data, dist_calculator)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    saver = Saver(sess)
    sess.run(tf.global_variables_initializer())
    eval = Eval(data, dist_calculator)
    train_costs, train_times, val_results_dict = \
        train_val_loop(data, eval, model, saver, sess)
    best_iter, test_results = \
        test(data, eval, model, saver, sess, val_results_dict)
    overall_time = convert_long_time_to_str(time() - t)
    print(overall_time)
    saver.save_overall_time(overall_time)
    return train_costs, train_times, val_results_dict, best_iter, test_results


if __name__ == '__main__':
    main()
