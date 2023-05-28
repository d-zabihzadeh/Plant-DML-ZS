from __future__ import print_function
from __future__ import division

import os

import torch
import json
import random, numpy as np
from common.Struct import Struct
from common.Result import print_params
from Evaluation import print_res
# from Visualization import plot_NMI_Recall


# __repr__ may contain `\n`, json replaces it by `\\n` + indent
json_dumps = lambda **kwargs: json.dumps(
    **kwargs
).replace('\\n', '\n    ')


class JSONEncoder(json.JSONEncoder):
    def default(self, x):
        # add encoding for other types if necessary
        if isinstance(x, range):
            return 'range({}, {})'.format(x.start, x.stop)
        if not isinstance(x, (int, str, list, float, bool)):
            return repr(x)
        return json.JSONEncoder.default(self, x)


def load_config(config_name = 'config.json'):
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config

def predict_batchwise(model, dataloader):
    # list with N lists, where N = |{image, label, index}|
    model_is_training = model.training
    model.eval()
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():

        # extract batches (A becomes list of samples)
        for batch in dataloader:
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(list(model.parameters())[0].device)
                    # predict model output for image
                    J = model(J).cpu()
                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def reset_rng(seed):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)  # set random seed for all gpus


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



def get_bestRes(ds,alg, g_split=.5, embed_dim=128, path_res=None):
    if(path_res is None):
        path_res = '.\\Res_'

    dir_name = path_res + ds
    file_names = listdir(dir_name)
    best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
    for f in file_names:
        if not f.endswith('res'):
            continue
        m = f.lower().split('__') # m[0] shows model name
        if len(m) > 0 and m[0] == alg.lower():
            with open(path.join(dir_name, f), 'rb') as file:
                score, params, res_info, hist = pickle.load(file)
            if hasattr(params, 'g_ipc'):
                continue
            if not hasattr(params, 'gallery_split'):
                params.gallery_split = 1 - params.query_split
            if params.embed_dim == embed_dim and round(params.gallery_split,2) == g_split and score > best_score:
                best_score = score
                bestParams = params
                best_resInfo = res_info
                hist_best = hist

    return best_score, bestParams, best_resInfo, hist_best


def get_bestRes_ipc(ds,alg, embed_dim=128, g_ipc=10, path_res=None):
    if (path_res is None):
        path_res = '.\\Res_'

    dir_name = path_res + ds
    file_names = listdir(dir_name)
    best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
    for f in file_names:
        if not f.endswith('res'):
            continue
        m = f.lower().split('__')  # m[0] shows model name
        if len(m) > 0 and m[0] == alg.lower():
            with open(path.join(dir_name, f), 'rb') as file:
                score, params, res_info, hist = pickle.load(file)
            if not hasattr(params, 'g_ipc'):
                continue
            if params.embed_dim == embed_dim and params.g_ipc == g_ipc and score > best_score:
                best_score = score
                bestParams = params
                best_resInfo = res_info
                hist_best = hist

    return best_score, bestParams, best_resInfo, hist_best


def save_res(ds, alg, best_score, params, res, hist):
    if hasattr(params, 'g_ipc'):
        file_name = '.\\Res_%s\\%s__%s__%0.2f__%d_ipc.res' \
                    % (ds, alg, ds, best_score, params.g_ipc)
    else:
        file_name = '.\\Res_%s\\%s__%s_%0.2f__%0.2f_gs.res' \
                    % (ds, alg, ds, best_score, params.gallery_split)
    with open(file_name, 'wb') as f:  # wb: open binary file for writing
        pickle.dump([best_score, params, res, hist], f)

    return file_name

def save_result(model, opt, params, res, hist, epoch):
    best_score = res.score
    res.best_epoch = epoch
    file_res = save_res(params.ds, params.model_name, best_score, params, res, hist)
    file_name = '.\\Res_%s\\%s__%0.2f__%s__%d.mdl' % (params.ds, params.model_name, best_score, params.ds, params.embed_dim)
    torch.save({
        'state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'best_score': best_score,
        'params': params
    }, file_name)

    return file_res, file_name


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_save(best_score, h, params, alg=None, model=None, opt=None, plot_train=True):
    recall1_list = np.array(h.recallK_list)[:, 0]
    best_epoch = np.argmax(recall1_list)
    best_recall1 = recall1_list[best_epoch]
    if alg is None:
        alg = params.alg

    if (best_recall1 > best_score):
        best_Res = Struct()
        best_Res.epoch, best_Res.nmi = best_epoch, h.nmi_list[best_epoch]
        best_Res.recallK, best_Res.acc = h.recallK_list[best_epoch], h.acc_list[best_epoch]
        save_res(params.ds, alg, best_recall1, params, best_Res, h)
        print_params(params)
        print_res(best_recall1, best_Res)
        # plot_NMI_Recall(params, h, save_flag=True, plot_train=plot_train)
        if model is not None:
            file_name = '.\\Res_%s\\%s__%0.2f__%s__%0.2f__%d.mdl' % (
            params.ds, alg, best_score, params.ds, params.query_split, params.embed_dim)
            torch.save({
                'state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'best_score': best_score,
                'params': params
            }, file_name)

            torch.save(model.state_dict(), './saved_models/%s__%s_%0.2f.mdl' % (alg, params.ds, best_recall1))

        return True, best_epoch
    else:
        return False, best_epoch


def check_save2(best_score, res, h, params, alg=None, model=None, opt=None, plot_train=True):
    if alg is None:
        alg = params.alg

    best_recall1 = res.recallK[0]
    if (best_recall1 > best_score):
        res.epoch = len(h.acc_list)-1
        best_score = best_recall1
        file_res = save_res(params.ds, alg, best_recall1, params, res, h)
        print_params(params)
        print_res(best_recall1, res)
        # plot_NMI_Recall(params, h, save_flag=True, plot_train=plot_train)
        if model is not None:
            file_mdl = '.\\Res_%s\\%s__%0.2f__%s__%0.2f_gs__%d.mdl' % (
            params.ds, alg, best_recall1, params.ds, params.gallery_split, params.embed_dim)
            torch.save({
                'state_dict': model.state_dict(),
                'opt_state_dict': opt.state_dict(),
                'best_score': best_score,
                'params': params
            }, file_mdl)
            return True, file_res, file_mdl

        return True, file_res, ''
    else:
        return False, '', ''

def update_hist_result(h, file_res):
    if file_res == '':
        return False

    with open(file_res, 'rb') as file:
        score, params, res, _ = pickle.load(file)

    with open(file_res, 'wb') as file:
        pickle.dump([score, params, res, h], file)

    return True


def conv2mat(ds, alg, g_split, path_res=None):
    from scipy.io import savemat
    best_score, params, res, hist = get_bestRes(ds, alg, g_split=g_split, path_res=path_res)
    if best_score > 0:
        mdic = {'best_score': best_score, 'params': params, 'res':res, 'hist':hist }
        file_name = '.\\Res_%s\\%s__%s_%0.2f.mat' % (ds, alg, ds, best_score)
        savemat(file_name, mdic)




from os import path, listdir
import pickle


def correct_metric(ds, path_res=None, del_flag = False):
    if(path_res is None):
        path_res = '.\\Res_' + ds

    file_names = listdir(path_res)
    files_to_del = []
    for f in file_names:
        try:
            if not f.endswith('.mat'):
                continue
            with open(path.join(path_res, f), 'rb') as file:
                score, params, res_info, hist = pickle.load(file)

                if not hasattr(params, 'alg'):
                    loc_ds = f.find(ds)
                    params.alg = f[:loc_ds-1] # -1 is for underscore char _

                if(not hasattr(hist, 'recallK_list')):
                    hist.recallK_list = hist.recallK_ist
                    hist.__dict__.pop('recallK_ist')

                if isinstance(hist.recallK_list[0],list):
                    hist.recallK_list[0] = np.array(hist.recallK_list[0])

                if isinstance(hist.recallK_train_list[0],list):
                    hist.recallK_train_list[0] = np.array(hist.recallK_train_list[0])

                if (res_info.recallK[0] <= 1):
                    res_info.recallK = res_info.recallK * 100
                    n, m = len(hist.recallK_list), len(hist.recallK_train_list)
                    for i in range(n):
                        hist.recallK_list[i] *= 100
                        if(i<m):
                            hist.recallK_train_list[i] *= 100
                    if hist.nmi_list[i] <= 1:
                        hist.nmi_list[i] = (hist.nmi_list[i]*100)

                n, m = len(hist.nmi_list), len(hist.nmi_train_list)
                for i in range(n):
                    if hist.nmi_list[i] <= 1:
                        hist.nmi_list[i] = (hist.nmi_list[i] * 100)
                    if (i < m and hist.nmi_train_list[i] <= 1):
                        hist.nmi_train_list[i] = (hist.nmi_train_list[i] * 100)

                score = res_info.recallK[0]
                alg = params.alg
                file_name = '.\\Res_%s\\%s__%s_%0.2f.res' % (ds, alg, ds, score)
                with open(file_name, 'wb') as f_new:  # wb: open binary file for writing
                    pickle.dump([score, params, res_info, hist], f_new)

                files_to_del.append(path.join(path_res, f))

        except Exception as ex:
            print('an error occured during processing file:', f)

    if del_flag:
        for f in files_to_del:
            os.remove(f)


def rescale_results(ds, path_res=None):
    if(path_res is None):
        path_res = '.\\Res_' + ds

    file_names = listdir(path_res)
    for f in file_names:
        try:
            if not f.endswith('.res'):
                continue
            with open(path.join(path_res, f), 'rb') as file:
                score, params, res_info, hist = pickle.load(file)

                if not hasattr(params, 'alg'):
                    loc_ds = f.find(ds)
                    params.alg = f[:loc_ds-1] # -1 is for underscore char _

                if(not hasattr(hist, 'recallK_list')):
                    hist.recallK_list = hist.recallK_ist
                    hist.__dict__.pop('recallK_ist')

                if isinstance(hist.recallK_list[0],list):
                    hist.recallK_list[0] = np.array(hist.recallK_list[0])

                if isinstance(hist.recallK_train_list[0],list):
                    hist.recallK_train_list[0] = np.array(hist.recallK_train_list[0])

                if (res_info.recallK[0] <= 1):
                    res_info.recallK = res_info.recallK * 100
                    n, m = len(hist.recallK_list), len(hist.recallK_train_list)
                    for i in range(n):
                        hist.recallK_list[i] *= 100
                        if(i<m):
                            hist.recallK_train_list[i] *= 100
                    if hist.nmi_list[i] <= 1:
                        hist.nmi_list[i] = (hist.nmi_list[i]*100)

                n, m = len(hist.nmi_list), len(hist.nmi_train_list)
                for i in range(n):
                    if hist.nmi_list[i] <= 1:
                        hist.nmi_list[i] = (hist.nmi_list[i] * 100)
                    if (i < m and hist.nmi_train_list[i] <= 1):
                        hist.nmi_train_list[i] = (hist.nmi_train_list[i] * 100)

                score = res_info.recallK[0]
                alg = params.alg
                file_name = '.\\Res_%s\\%s__%s_%0.2f.res' % (ds, alg, ds, score)
                with open(file_name, 'wb') as f_new:  # wb: open binary file for writing
                    pickle.dump([score, params, res_info, hist], f_new)

        except Exception as ex:
            print('an error occured during processing file:', f)


def main():
    ds_name, g_split = 'plant_village', .01
    alg = 'soft_triple'

    conv2mat(ds_name, alg, g_split)

    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')




