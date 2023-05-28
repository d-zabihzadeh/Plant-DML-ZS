from Evaluation import eval_print_res, print_init_res, update_history, print_res
from common.Struct import Struct
from common.Result import print_params
from common.bcolors import bcolors
import torch
import torch.backends.cudnn as cudnn
from dataset import load_ds
import utils
import net
from models.bn_inception import BNInception_Attention
from Losses import SoftTriple
import time
from utils import check_save, get_bestRes, check_save2, update_hist_result
from tqdm import tqdm


def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch+1)%20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate


def override_best_params(params):
    params.num_epochs = 20
    # params.batch_size = 64
    # params.embedding_lr = .001
    params.model_lr = 1e-4
    params.embed_dim = 128
    return params


utils.reset_rng(seed=0)
alg = 'att_soft_triple'
ds_name, g_split = 'plant_village', .01

best_score, params, best_resInfo, hist_best = get_bestRes(ds_name, alg, g_split=g_split)
if(best_score == 0):
    params = Struct()
    params.ds = ds_name
    params.num_epochs = 20
    params.batch_size = 64
    params.weight_decay = 1e-4
    params.embed_dim = 128
    params.model_lr = .0001
    params.dml_layer_lr = .01
    params.embedding_lr = .0001
    params.la, params.gamma, params.tau, params.margin = 20, .1, .2, .01
    params.rate = .1  # decay rate
    params.K = 10
    params.eps = .01
    params.gallery_split, params.query_split = g_split, 1-g_split
    params.kNN_K = 3

print(bcolors.OKBLUE + 'best Score=%.2f'% best_score + bcolors.ENDC)
print_res(best_score, best_resInfo)
params = override_best_params(params)
print_params(params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device = torch.device('cpu')
config = utils.load_config()

# load data
# params.source_classes = range(0,60), # params.test_classes = range(120,140)
train_loader, test_loader = load_ds(params, config)
c = len(params.source_classes)

# create model
if alg.lower().startswith('att'):
    model = BNInception_Attention(params.embed_dim, is_pretrained=True).to(device)
else:
    model = net.bninception(params.embed_dim).to(device)

dml_layer = SoftTriple(c, params.embed_dim, params.la, params.gamma, params.tau, params.margin, params.K).cuda()
opt = torch.optim.Adam([{"params": model.parameters(), "lr": params.model_lr},
                              {"params": dml_layer.parameters(), "lr": params.dml_layer_lr}],
                             eps=params.eps, weight_decay=params.weight_decay)
scheduler = config['lr_scheduler']['type'](
    opt, **config['lr_scheduler']['args']
)
cudnn.benchmark = True
print_params(params)
h = print_init_res(best_score, best_resInfo, hist_best, model, test_loader, train_loader)
h.kSet = [1, 2, 4, 8]

epoch, file_res = 1, ''
while (epoch <= params.num_epochs):
    time_per_epoch = time.time()
    adjust_learning_rate(opt, epoch, params)
    model.train()
    dml_layer.train()
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (data, labels, _) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            opt.zero_grad()
            X_embed = model(data)
            loss = dml_layer(X_embed, labels)
            loss.backward()
            opt.step()
            t.set_postfix(Epoch='%d' % epoch, Iteration='%d' % batch_idx, Loss='{:.3f}'.format(loss))
            t.update()

    time_per_epoch = time.time() - time_per_epoch
    res, res_train = eval_print_res(model, test_loader, train_loader, epoch, time_per_epoch)[:2]
    update_history(h, res, res_train)
    flag, f_res, _ = check_save2(best_score, res, h, params, alg, model=model, opt=opt)
    if flag:
        best_score, file_res = res.recallK[0], f_res

    scheduler.step()
    epoch += 1

update_hist_result(h, file_res)
print('finished!!!')





