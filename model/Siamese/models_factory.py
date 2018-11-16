from model_regression import SiameseRegressionModel
from model_ranking import SiameseRankingModel
from model_classification import SiameseClassificationModel


# from utils import get_save_path, save, load, create_dir_if_not_exists
# from utils_siamese import get_model_info_as_str
# from config import FLAGS
# from collections import OrderedDict


def create_model(model, input_dim, data, dist_calculator):
    if model == 'siamese_regression':
        return SiameseRegressionModel(input_dim, data, dist_calculator)
    elif model == 'siamese_ranking':
        return SiameseRankingModel(input_dim, data, dist_calculator)
    elif model == 'siamese_classification':
        return SiameseClassificationModel(input_dim, data, dist_calculator)
    else:
        raise RuntimeError('Unknown model {}'.format(model))

# def try_load_model():
#     d = get_model_lookup_dict()
#     ms = get_model_info_as_str()
#     sfn = d.get(ms)
#     if sfn:
#         return load(sfn)
#     else:
#         sfn = '{}/{}'.format(get_model_save_dir(), len(d))
#         d[ms] = sfn
#         return sfn
#
#
# def get_model_lookup_dict():
#     sfn = '{}/model_lookup_dict'.format(get_model_save_dir())
#     l = load(sfn)
#     if l:
#         return l
#     d = OrderedDict()
#     save(sfn, d)
#     return d
#
#
# def get_model_save_dir():
#     dir = '{}/siamese_models'.format(get_save_path())
#     create_dir_if_not_exists(dir)
#     return dir
