import math
import pickle
import textwrap
from common.bcolors import bcolors
from common.Struct import Struct
from common.utils import MyDict, Flags_Base, mean_std_str
import os
import numpy as np


def _compare(s1, s2):
    if s1 < s2:
        return -1
    elif s1 > s2:
        return 1
    else:
        return 0

class Result_Base(MyDict):
    def __init__(self, *args, **kwargs):
        super(Result_Base, self).__init__(*args, **kwargs)
        if 'score_fn' not in self.keys():
            self.score_fn = get_score

    @staticmethod
    def from_struct(s):
        r = Result(s.__dict__)
        return r

    def score(self):
        return self.score_fn(self)

    def __getattr__(self, name):
        if name == 'score':
            return self.score()
        return super().__getattr__(name)






    def compare_to(self, other):
        if other is None:
            return 1
        return _compare(self.score(), other.score())

    def __lt__(self, other):
        r = self.compare_to(other)
        return True if r < 0 else False

    def __gt__(self, other):
        c = self.compare_to(other)
        return True if c > 0 else False

    def __le__(self, other):
        c = self.compare_to(other)
        return True if c <= 0 else False

    def __eq__(self, other):
        c = self.compare_to(other)
        return True if c == 0 else False

    def _metric_str(self, m):
        m_val = self[m]
        std_val = self[f'{m}_std'] if hasattr(self, f'{m}_std') else 0.

        m_str = f'{mean_std_str(m_val, std_val)}' \
            if hasattr(self, f'{m}_std') else f'{m_val:2f}'
        return m_str


class Reg_Result(Result_Base):
    def __init__(self, *args, **kwargs):
        super(Reg_Result, self).__init__(*args, **kwargs)
        if 'mae' not in self.keys():
            self.mae = math.inf

        if 'rmse' not in self.keys():
            self.rmse = math.inf

        self.score_fn = get_score_reg

    def compare_to(self, other):
        if other is None:
            return 1
        c1 = _compare(self.score(), other.score())
        if c1 == 0:
            return _compare(-self.rmse, -other.rmse)
        else:
            return c1

    def mae_str(self, frmt='.2f'):
        return self._metric_str('mae')

    def rmse_str(self, frmt='.2f'):
        return self._metric_str('rmse')


    def print(self, pre_text='', color=bcolors.FAIL):
        if len(pre_text) > 0:
            print(color + pre_text + bcolors.ENDC, end=' ')

        s = ''
        if hasattr(self, 'epoch'):
            s += f'epoch:{self.epoch}, '

        if hasattr(self, 'loss'):
            s += f'loss= {self.loss:.4f}, '

        s += f'MAE={self.mae_str()},  RMSE={self.rmse_str()}'

        print(color + s + bcolors.ENDC)


def gen_reg_res(mae_arr, rmse_arr, loss_arr):
    res = Reg_Result()
    res.mae_arr, res.rmse_arr = mae_arr, rmse_arr
    res.mae, res.rmse = res.mae_arr.mean(), res.rmse_arr.mean()
    res.mae_std, res.rmse_std = res.mae_arr.std(), res.rmse_arr.std()
    res.loss = loss_arr.mean()
    return res

def gen_clf_res(y_pred, y_true, loss_arr, probs):
    from common.classification_res import report_multi_classification, report_bin_classification
    if len(np.unique(y_true)) == 2:
        res = report_bin_classification(y_true, y_pred, probs, percent=True)
    else:
        res = report_multi_classification(y_true, y_pred, probs, percent=True)
    if loss_arr is not None:
        res.loss = loss_arr.mean()
    return res


def get_score(res):
    return (res.acc + res.f_score)/2

class Result(MyDict):
    def __init__(self, *args, **kwargs):
        super(Result, self).__init__(*args, **kwargs)
        if 'acc' not in self.keys():
            self.acc = 0.

        if 'f_score' not in self.keys():
            self.f_score = 0.

        if 'score_fn' not in self.keys():
            self.score_fn = get_score

    @staticmethod
    def from_struct(s):
        r = Result(s.__dict__)
        return r

    def score(self):
        return self.score_fn(self)

    def compare_to(self, other):
        if other is None:
            return 1
        c1 = _compare(self.score(), self.score_fn(other))
        if c1 == 0:
            return _compare(self.acc, other.acc)
        else:
            return c1

    def __lt__(self, other):
        r = self.compare_to(other)
        return True if r < 0 else False

    def __gt__(self, other):
        c = self.compare_to(other)
        return True if c > 0 else False

    def __le__(self, other):
        c = self.compare_to(other)
        return True if c <= 0 else False

    def __eq__(self, other):
        c = self.compare_to(other)
        return True if c == 0 else False

    def print(self, pre_text='', color=bcolors.FAIL, conf_matrix=False):
        print_res(self, color, pre_text, conf_matrix)


def get_score_reg(res):
    return -res.mae


class Result2(Struct):
    def __init__(self, score_fn=None):
        super(Result2, self).__init__()
        self.acc, self.f_score = 0., 0.
        self.score_fn = score_fn if score_fn is not None else get_score

    @staticmethod
    def from_struct(s):
        r = Result2()
        r.__dict__ = s.__dict__
        return r

    def score(self):
        return self.score_fn(self)

    def compare_to(self, other):
        if other is None:
            return 1
        c1 = _compare(self.score(), get_score(other))
        if c1 == 0:
            return _compare(self.acc, other.acc)
        else:
            return c1

    def __lt__(self, other):
        r = self.compare_to(other)
        return True if r < 0 else False

    def __gt__(self, other):
        c = self.compare_to(other)
        return True if c > 0 else False

    def __le__(self, other):
        c = self.compare_to(other)
        return True if c <= 0 else False

    def __eq__(self, other):
        c = self.compare_to(other)
        return True if c == 0 else False

    def print(self, color=None, conf_matrix=False):
        if color is None:
            color = bcolors.FAIL

        if hasattr(self, 'report'):  # multiclass
            print(color + self.report + bcolors.ENDC)
            s = ''
            if hasattr(self, 'epoch'):
                s = f'epoch:{self.epoch}, ' + s

            if hasattr(self, 'loss'):
                s += f' loss= {self.loss:.4f}, '

            if hasattr(self, 'g_mean'):
                s += f' g_mean= {self.g_mean:.2f}, '

            print(color + s + bcolors.ENDC)
            return

        s = """accuracy: {:.2f}, f_score: {:.2f}, sensitivity:{:.2f}, precision:{:.2f},
        specificity:{:.2f}, g_mean:{:.2f}"""

        if hasattr(self, 'epoch'):
            s = f'epoch:{self.epoch}, ' + s

        if hasattr(self, 'loss'):
            s = f'loss= {self.loss:.4f}, ' + s

        if hasattr(self, 'roc_auc'):
            s += f', roc_auc= {self.roc_auc:.2f}, '

        print(color + s.format(self.acc, self.f_score, self.sensitivity,
                               self.precision, self.specificity, self.g_mean) + bcolors.ENDC)

        if conf_matrix:
            print('confusion matrix =')
            print(self.conf_matrix)


# def get_bestRes(ds,alg,path_res=None):
#     if(path_res is None):
#         path_res = '.\\Res_'
#
#     dir_name = path_res + ds
#     file_names = listdir(dir_name)
#     best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
#     for f in file_names:
#         if(f.lower().startswith(alg.lower()) and f.endswith('.self')):
#             with open(path.join(dir_name, f), 'rb') as file:
#                 score, params, res_info, hist = pickle.load(file)
#             if (score > best_score):
#                 best_score = score
#                 bestParams = params
#                 best_resInfo = res_info
#                 hist_best = hist
#
#     return best_score, bestParams, best_resInfo, hist_best

def get_bestRes(ds, alg, flags: Flags_Base = None, path_res=None, file_name=''):
    import torch
    if(path_res is None):
        path_res = '.\\Res_'

    dir_name = path_res + ds
    if len(file_name) > 0:
        file_path = os.path.join(dir_name, file_name)
        return load_res(file_path)

    best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        return best_score, bestParams, best_resInfo, hist_best
    file_names = os.listdir(dir_name)
    for f in file_names:
        if not (f.endswith('res') or f.endswith('mdl')):
            continue
        m = f.lower().split('__') # m[0] shows model name
        if len(m) > 0 and m[0] == alg.lower():
            if f.endswith('mdl'):
                try:
                    ch = torch.load(os.path.join(dir_name, f))
                    score, params = ch['best_score'], ch['args']
                    res_info, hist = ch['res'], ch['history']
                    if flags is None or flags.match(params):
                        best_score = score
                        bestParams = params
                        best_resInfo = res_info
                        hist_best = hist
                except BaseException as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    pass
    return best_score, bestParams, best_resInfo, hist_best


def load_res(file_path):
    import torch
    ch = torch.load(file_path)
    score, params = ch['best_score'], ch['args']
    res_info, hist = ch['res'], ch['hist']
    return score, params, res_info, hist


def print_params(params, name='params', width=150):
    print(name)
    print(textwrap.fill(str(params.__dict__), width=width))
    if hasattr(params, 'load_options'):
        print('load data options:')
        print(textwrap.fill(str(params.load_options.__dict__), width=width))


def save_res(ds, alg, best_score, params, res, hist):
    file_name = '.\\Res_%s\\%s__%s_%0.2f.self' % (ds, alg, ds, best_score)
    if hasattr(params,'a'):
        del params.a
    if hasattr(params, 'preds'):
        del params.preds

    with open(file_name, 'wb') as f:  # wb: open binary file for writing
        pickle.dump([best_score, params, res, hist], f)

    return file_name

def print_res(res, color=bcolors.FAIL, pre_text='', conf_matrix=False, summary_sw=False):
    if type(res) == Reg_Result:
        res.print(pre_text, color)
        return

    multi_class = res.get("multi_class", False) # hasattr(res, 'report'):  #multiclass
    if multi_class:
        if summary_sw:
            s = pre_text + """ score: {:.2f}, accuracy: {:.2f}, recall:{:.2f}, precision:{:.2f},
            f_score: {:.2f}""".format(res.score(), res.acc, res.sensitivity,
                  res.precision, res.f_score)
        else:
            print(color + res.report + bcolors.ENDC)
            s = pre_text

        if hasattr(res, 'epoch'):
            s = f'epoch:{res.epoch}, ' + s

        if hasattr(res, 'loss'):
            s += f' loss= {res.loss:.4f}, '

        if hasattr(res, 'g_mean'):
            s += f' g_mean= {res.g_mean:.2f}, '

        print(color + s + bcolors.ENDC)
        if conf_matrix:
            print('confusion matrix =')
            print(res.conf_matrix)

        return

    s = """score: {:.2f}, accuracy: {:.2f}, recall:{:.2f}, precision:{:.2f},
    specificity:{:.2f}, f_score: {:.2f}, g_mean:{:.2f}"""

    if hasattr(res, 'epoch'):
        s = f'epoch:{res.epoch}, ' + s

    if hasattr(res, 'loss'):
        s = f'loss= {res.loss:.4f}, ' + s

    if hasattr(res, 'roc_auc'):
        s += f', roc_auc= {res.roc_auc:.2f}, '

    print(color + s.format(res.score(), res.acc, res.sensitivity,
                  res.precision, res.specificity, res.f_score, res.g_mean) + bcolors.ENDC)

    print(res.score(), res.acc, res.sensitivity,
          res.precision, res.specificity, res.f_score, res.g_mean)

    if conf_matrix:
        print('confusion matrix =')
        print(res.conf_matrix)


def print_bestRes(ds,alg, path_res=None, show_params=True):
    best_score, params, best_resInfo = get_bestRes(ds, alg, path_res)
    print(bcolors.HEADER + 'Best Score = %0.2f' % (best_score) + bcolors.ENDC)
    if (best_score > 0 and show_params):
        print_params(params)
        print('confusion matrix =')
        print(best_resInfo.conf_matrix)
        print('Mean specificity = %0.2f, sensitivity = %0.2f' % ((best_resInfo.specificity * 100), (best_resInfo.sensitivity * 100)))
        print('Mean f_measure = %0.2f' % (best_resInfo.f_score * 100))
        print('Mean g_mean = %0.2f' % (best_resInfo.g_mean * 100))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n > 0:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def main():
    r = Reg_Result()
    r.mae = 1.5
    r2 = Reg_Result()
    r2.mae = 1
    if r > r2:
        print('r is better')
    r.print('r =')


    from classification_res import report_multi_classification
    # res =
    #
    # res.acc = 10
    # print(res.acc)
    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

