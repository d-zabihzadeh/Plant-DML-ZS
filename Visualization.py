import math
from scipy.interpolate import make_interp_spline, BSpline
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.utils
import cv2
from utils import get_bestRes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
import seaborn as sns


ds_names = {'plant_village': 'Plant Village'}
alg_disp_names = {'att_soft_triple': 'ZS-DML', 'soft_triple': 'Soft Triple',
                  'att_binomial': 'Binomial + GDFL', 'binomial': 'Binomial Loss',
                  'att_proxy_nca': 'Proxy-NCA + GDFL', 'proxy_nca': 'Proxy-NCA',
                  'att_triplet_hinge': 'Triplet + GDFL', 'triplet_hinge': 'Triplet Loss'
                  }
metric_disp_names = {'3NN':'3NN Accuracy', 'NMI':'NMI', 'Recall@1':'Recall@1'}
font_size, title_size = 14, 16
colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']


def show_annotate(text, loc='upper left'):
    from matplotlib.offsetbox import AnchoredText
    ax = plt.gca()
    at = AnchoredText(text,
                      prop=dict(size=15), frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

def t_sne_vis(X_embed, y_pred):
    plt.figure()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_embed = X_embed.to("cpu")
    X_embed = X_embed.numpy()
    X_embed_reduced = TSNE(n_components=2, random_state=0).fit_transform(X_embed)
    # plt.scatter(X_embed_reduced[:, 0], X_embed_reduced[:, 1],
    #             c=y_pred, cmap='jet')
    # plt.colorbar()
    # plt.show()

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_embed_reduced[:, 0], X_embed_reduced[:, 1], c=y_pred, cmap='jet')
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    class_names = {32:'Apple Scab', 33:'Apple Black Rot', 34:'Apple Cedar Rust',
                   35:'Apple Healthy', 36:'Blueberry Healthy', 37:'Cherry Healthy'}
    for i, key in enumerate(class_names):
        legend1.get_texts()[i].set_text(class_names[key])

    ax.add_artist(legend1)


def t_sne_3D(X_embed, y_pred):
    plt.figure()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_embed = X_embed.to("cpu")
    X_embed = X_embed.numpy()
    X_3D = TSNE(n_components=3, random_state=0).fit_transform(X_embed)
    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    class_names = {32:'Apple Scab', 33:'Apple Black Rot', 34:'Apple Cedar Rust',
                   35:'Apple Healthy', 36:'Blueberry Healthy', 37:'Cherry Healthy'}
    for c in np.unique(y_pred):
        ax1.plot(X_3D[y_pred == c, 0], X_3D[y_pred == c, 1], X_3D[y_pred == c, 2], '.', alpha=0.3,
                 label=class_names[c])
    plt.title('T-SNE 3D Visualization')
    # leg = plt.legend(prop={'size': font_size - 6} , loc='upper left')

    def update(handle, orig):
        handle.update_from(orig)
        handle.set_alpha(1)

    from matplotlib.collections import PathCollection
    from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
    plt.legend(prop={'size': font_size - 6} , loc='upper left',
               handler_map={PathCollection: HandlerPathCollection(update_func=update),
                plt.Line2D: HandlerLine2D(update_func=update)})



    # legend1 = ax1.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    # class_names = {32:'Apple Scab', 33:'Apple Black Rot', 34:'Apple Cedar Rust',
    #                35:'Apple Healthy', 36:'Blueberry Healthy', 37:'Cherry Healthy'}
    # for i, key in enumerate(class_names):
    #     legend1.get_texts()[i].set_text(class_names[key])
    # ax1.add_artist(legend1)
    return


def plot_confusion_mat(y_true, y_pred):
    cf_matrix = confusion_matrix(y_true, y_pred)
    font_size = 14
    plt.rcParams['text.usetex'] = False
    categories = list(range(32, 38))
    my_dpi = 100
    fig, ax = plt.subplots(constrained_layout=True, figsize=(800 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    sns.heatmap(cf_matrix / np.sum(cf_matrix, axis=-1), fmt='.1%', annot=True, xticklabels=categories,
                yticklabels=categories)
    # sns.heatmap(cf_matrix/np.sum(cf_matrix, axis=-1), fmt='.1%', annot=True, cmap='jet',
    #             xticklabels=categories,yticklabels=categories)
    plt.xlabel('Predicted label', fontsize=font_size)
    plt.ylabel('Actual label', fontsize=font_size)


def plot_RecallK(recallK_list, title,  kSet, file_name=''):
    plt.figure()
    recallK_list = np.array(recallK_list)
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y']
    epochs = range(1, len(recallK_list) + 1)

    for i in range(len(kSet)):
        plt.plot(epochs, recallK_list[:, i], color=colors[i], label='Test Recall@%d' % (kSet[i]))
    plt.title(title)
    plt.legend()
    if len(file_name) > 0:
        max_recall1 = np.max(recallK_list[:, 0])
        file_name = file_name +'%0.2f' % (max_recall1)
        plt.savefig(file_name + '.eps')
        plt.savefig(file_name + '.png')
    plt.show()


def plot_NMI_Recall(params, hist, save_flag = False, plot_train = True):
    ds, alg, g_split = params.ds, params.alg, params.g_split
    epochs = range(1, len(hist.nmi_list) + 1)
    plt.plot(epochs, hist.nmi_list, color='blue', marker='*', label='Test NMI')
    plt.xlabel('Epochs')
    plt.ylabel('NMI')
    if plot_train:
        plt.plot(epochs, hist.nmi_train_list, color='red', marker='^', label='Train NMI')
        plt.title('Test and Train NMI on %s' % (ds))
    else:
        plt.title('Test NMI on %s' % (ds))

    plt.legend()
    if save_flag:
        max_nmi = np.max(hist.nmi_list)
        fname = './figs/%s_%s_%0.2f_nmi_%0.2f'%(alg,ds, g_split, max_nmi)
        plt.savefig(fname +'.eps')
        plt.savefig(fname +'.png')
        fname_recall_test = './figs/%s_%s_%0.2f_test_recall_' % (alg, ds, g_split)
        fname_recall_train = './figs/%s_%s_train_recall_' % (alg, ds)
    else:
        fname_recall_test, fname_recall_train = '', ''

    plt.show()
    plot_RecallK(hist.recallK_list, 'Test RecallK on %s' % ds, hist.kSet, file_name=fname_recall_test)

    if plot_train:
        plot_RecallK(hist.recallK_train_list, 'Train RecallK on %s' % ds, hist.kSet, file_name=fname_recall_train)


def plot_NMI_recall_best(ds_name, alg):
    best_score, params, best_resInfo, hist = get_bestRes(ds_name, alg)
    plot_NMI_Recall(ds_name, hist)


import pickle
def plot_results_vs(file_name, alg_name, ds, vs_param):
    plt.figure()
    with open(file_name, 'rb') as f:
        x_vals, y_vals = pickle.load(f)

    x_range = range(0, len(x_vals))
    plt.plot(x_range, y_vals, color='b', label='%s' % (alg_name))
    plt.xticks(x_range, x_vals)
    plt.title('Test NMI vs %s on %s ' % (vs_param, ds_names[ds]), fontsize=title_size, fontweight='bold')
    plt.ylabel('NMI', fontsize=font_size)
    plt.xlabel(vs_param, fontsize=font_size)
    plt.show()

def plot_NMI_recall_dmls(ds, alg_list, gs, num_epoch = 20, kSet=None, save_flag = False,
                         annot='', annotate_loc = 'center', experiment_name=''):
    font_size, title_size = 14,16
    if kSet is None:
        kSet = [1, 2, 4, 8]

    ds_name = ds_names[ds]
    hist_list = []
    nmi_bottom, recall1_bottom = 0, 0
    for alg in alg_list:
        best_score, params, best_resInfo, hist = get_bestRes(ds, alg, g_split=gs)
        hist_list.append(hist)
        num_epoch = np.minimum(num_epoch, len(hist.nmi_list))
        nmi_bottom += hist.nmi_list[0]
        if not hasattr(hist, 'recallK_list'):
            hist.recallK_list = hist.recallK_ist

        hist.recallK_list = np.array(hist.recallK_list)
        recall1_bottom += hist.recallK_list[0, 0]


    nmi_bottom, recall1_bottom = nmi_bottom/ len(alg_list), recall1_bottom/len(alg_list)
    plt.figure()
    epochs = range(0, num_epoch)
    for i, alg in enumerate(alg_list):
        plt.plot(epochs, hist_list[i].nmi_list[:num_epoch], color=colors[i], label='%s' % (alg_disp_names[alg]))

    plt.title('NMI of Query Set on %s'% (ds_name), fontsize=title_size, fontweight='bold')
    plt.xticks(list(range(0,num_epoch,2)))
    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('NMI', fontsize=font_size)
    plt.ylim(top=100)
    if len(annot) > 0:
        show_annotate(annot, loc=annotate_loc)

    plt.legend(prop=dict(weight='bold'))
    if save_flag:
        fname = './figs/test_nmi_%s_%s' % (ds, experiment_name)
        plt.savefig(fname + '.eps')
        plt.savefig(fname + '.png')

    plt.show()

    for j in range(len(kSet)):
        plt.figure()
        for i, alg in enumerate(alg_list):
            plt.plot(epochs, hist_list[i].recallK_list[:num_epoch, j],
                     color=colors[i], label='%s'%(alg_disp_names[alg]))

        plt.title('Recall@%d of Query Set on %s' % (kSet[j],ds_name), fontsize=14, fontweight='bold')
        plt.xticks(list(range(0, num_epoch, 2)))
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Recall@%d'%(kSet[j]), fontsize=12)
        # if(j==0):
        #     plt.ylim(bottom=recall1_bottom)
        plt.ylim(top=100)
        if len(annot) > 0:
            show_annotate(annot, loc=annotate_loc)

        plt.legend(prop=dict(weight='bold'))

        if save_flag:
            fname = './figs/recall@%d_%s_%s' % (kSet[j], ds, experiment_name)
            plt.savefig(fname + '.eps')
            plt.savefig(fname + '.png')

        plt.show()


def plot_metric(metric, alg_list, vals, num_epoch, ds_name='Plant Village', annot='', annotate_loc='', file_name=''):
    font_size, title_size = 14,16
    plt.rcParams['text.usetex'] = True
    my_dpi = 100
    fig, ax = plt.subplots(constrained_layout=True, figsize=(800/my_dpi, 500/my_dpi), dpi=my_dpi)
    epochs = range(0, num_epoch)
    # ax.hlines(y=50, xmin=0, xmax=npoints, label="Softimpute", ls='--', color='c')

    for i, alg in enumerate(alg_list):
        # spl = make_interp_spline(epochs, vals[i][:num_epoch], k=3)  # type: BSpline
        # vals_smooth = spl(xnew)
        # plt.plot(xnew, vals_smooth, color=colors[i], label='%s' % (alg_disp_names[alg]))
        # plt.plot(epochs, vals[i][:num_epoch], color=colors[i], label='%s' % (alg_disp_names[alg]))
        alg_disp = alg_disp_names[alg]
        ax.plot(epochs, vals[i][:num_epoch], label=alg_disp)

    metric_name = metric_disp_names[metric]
    plt.title('%s of Query Set on %s' % (metric_name, ds_name), fontsize=title_size, fontweight='bold')
    plt.xticks(list(range(0, num_epoch, 2)), size=font_size)
    plt.xlabel('Epochs', fontsize=font_size)
    ax.set_ylabel(metric_name, fontsize=font_size+2)
    ax.tick_params(axis='y', labelsize=font_size + 1)
    # plt.ylim(top=95, bottom=55)
    if len(annot) > 0:
        show_annotate(annot, loc=annotate_loc)

    # plt.legend(prop=dict(weight='bold'), loc='upper left')
    plt.legend(prop={'size': font_size - 2, 'weight':'bold'}, loc='lower right')

    if len(file_name) > 0:
        fname = './figs/test_%s_%s' % (metric, file_name)
        plt.savefig(fname + '.eps')
        plt.savefig(fname + '.png')

    plt.show()


def plot_NMI_recall_dmls2(ds, alg_list, gs, num_epoch = 20, kSet=None, save_flag = False,
                         annot='', annotate_loc = 'center', file_name=''):
    if kSet is None:
        kSet = [1, 2, 4, 8]
    ds_name = ds_names[ds]
    hist_list = []
    nmi_bottom, recall1_bottom = 0, 0
    for alg in alg_list:
        best_score, params, best_resInfo, hist = get_bestRes(ds, alg, g_split=gs)
        hist_list.append(hist)
        num_epoch = np.minimum(num_epoch, len(hist.nmi_list))
        nmi_bottom += hist.nmi_list[0]
        if not hasattr(hist, 'recallK_list'):
            hist.recallK_list = hist.recallK_ist

        hist.recallK_list = np.array(hist.recallK_list)
        recall1_bottom += hist.recallK_list[0, 0]

    nmi_bottom, recall1_bottom = nmi_bottom/ len(alg_list), recall1_bottom/len(alg_list)
    vals = [h.nmi_list for h in hist_list]
    # plot_metric('NMI', alg_list, vals, num_epoch, ds_name, annot, annotate_loc, file_name=file_name)
    for j in range(len(kSet)):
        metric = 'Recall@%d'%kSet[j]
        vals = [h.recallK_list[:, j] for h in hist_list]
        plot_metric(metric, alg_list, vals, num_epoch, ds_name, annot, annotate_loc, file_name=file_name)

    vals = [h.acc_list for h in hist_list]
    metric = '3NN'
    plot_metric(metric, alg_list, vals, num_epoch, ds_name, annot, annotate_loc, file_name=file_name)



def main():
    ds = 'plant_village'
    gs = .10

    # alg_list = ['soft_triple', 'binomial', 'proxy_nca', 'triplet_hinge']
    alg_list = ['soft_triple', 'att_soft_triple', 'triplet_hinge', 'att_triplet_hinge']
    annot = 'Gallery Size = %d\%%' % (gs*100)
    experiment = '%s_%d_gs'%(ds, gs*100)
    plot_NMI_recall_dmls2(ds, alg_list, gs=gs, num_epoch=20, kSet=[1], save_flag=False,
                         annot=annot, annotate_loc='lower center', file_name=experiment)
    print('finished!!!')
    # with open('att_wEns_cat_cub_vs_lam', 'rb') as f:
    #     contents = pickle.load(f)
    # plot_results_vs('att_wEns_cat_flowers102_vs_lam.dat', 'WEDL-DML', 'flowers102', '$\lambda$')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
