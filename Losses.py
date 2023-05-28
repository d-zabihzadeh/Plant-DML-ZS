from __future__ import absolute_import
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn import init
from Miner import TripletMiner
import math
from torch import nn
from torch.autograd import Variable
import sklearn.preprocessing


class Angular_mc_loss(nn.Module):
    def __init__(self, alpha=45, in_degree=True):
        super(Angular_mc_loss, self).__init__()
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.sq_tan_alpha = np.tan(alpha) ** 2

    def forward(self, embeddings, target, with_npair=False, lamb=2):
        n_pairs = self.get_n_pairs(target)
        if (len(n_pairs) == 0):
            return torch.tensor(0.0, requires_grad=True)
        n_pairs = n_pairs.cuda()
        f = embeddings[n_pairs[:, 0]]
        f_p = embeddings[n_pairs[:, 1]]
        # print(f, f_p)
        term1 = 4 * self.sq_tan_alpha * torch.matmul(f + f_p, torch.transpose(f_p, 0, 1))
        term2 = 2 * (1 + self.sq_tan_alpha) * torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(len(f)).cuda()
        f_apn = f_apn * mask
        loss = torch.mean(torch.logsumexp(f_apn, dim=1))
        if with_npair:
            loss_npair = self.n_pair_mc_loss(f, f_p)
            # print(loss, loss_npair)
            loss = loss_npair + lamb*loss

        return loss

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])
        n_pairs = np.array(n_pairs)
        return torch.LongTensor(n_pairs)

    @staticmethod
    def n_pair_mc_loss(f, f_p):
        n_pairs = len(f)
        term1 = torch.matmul(f, torch.transpose(f_p, 0, 1))
        term2 = torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(n_pairs).cuda()
        f_apn = f_apn * mask
        return torch.mean(torch.logsumexp(f_apn, dim=1))


class Trip_hinge_loss(nn.Module):
    def __init__(self, margin = 0.05, sim_flag = False, smooth_loss = False, type_of_triplets = 'all'):
        super(Trip_hinge_loss, self).__init__()
        self.margin = margin
        self.sim_flag = sim_flag
        self.smooth_loss = smooth_loss
        self.trip_miner = TripletMiner(margin=.1, type_of_triplets=type_of_triplets, sim_flag=sim_flag)

    def forward(self, X_embed, y):
        trip_indices = self.trip_miner.mine(X_embed, y)
        anchor_idx, positive_idx, negative_idx = trip_indices[0], trip_indices[1], trip_indices[2]
        if self.sim_flag:
            mat = torch.matmul(X_embed, X_embed.t())
        else:
            mat = torch.cdist(X_embed, X_embed)

        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = (
            ap_dist - an_dist if self.sim_flag else an_dist - ap_dist
        )
        loss = F.relu(self.margin - triplet_margin)

        return loss.mean() if len(loss) > 0 else torch.tensor(0.0, requires_grad=True)

# N-Pair loss
#     Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
#     Processing Systems. 2016.
#     http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
#     """
class N_pair_loss(nn.Module):
    def __init__(self, l2_reg=0.02):
        super(N_pair_loss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        n_pairs, n_negatives = self.get_n_pairs(target)

        if(len(n_pairs) == 0):
            return torch.tensor(0.0, requires_grad=True)

        if embeddings.is_cuda:
            n_pairs = n_pairs.cuda()
            n_negatives = n_negatives.cuda()

        anchors = embeddings[n_pairs[:, 0]]  # (n, embedding_size)
        positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        negatives = embeddings[n_negatives]  # (n, n-1, embedding_size)

        losses = self.n_pair_loss(anchors, positives, negatives) \
                 + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i + 1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1 + x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]

class Lifted_struct_loss(nn.Module):
    def __init__(self, margin =0.05, l2_reg=0.02, n_pos_pair=3):
        # n_pos_pair: number of pos pair per class that should be considered
        super(Lifted_struct_loss, self).__init__()
        self.margin, self.l2_reg, self.n_pos_pair = margin, l2_reg, n_pos_pair

    def forward(self, X_embed, target):
        mat = torch.cdist(X_embed, X_embed)
        loss, count = 0, 0
        n = len(X_embed)
        l_n = torch.zeros(n)
        for i in range(n):
            l_n[i] = (self.margin - mat[i][target != target[i].item()]).exp().sum()

        for i in range(n):
            m = 0
            for j in range(i + 1, n):
                if target[i].item() == target[j].item():
                    loss_neg = (l_n[i] + l_n[j]).log()
                    # Positive component
                    loss_pos = mat[i, j]
                    loss += F.relu(loss_neg + loss_pos) ** 2
                    count += 1
                    m += 1
                    if m >= self.n_pos_pair:
                        break

        # for i in range(n):
        #     t_i = target[i].item()
        #     for j in range(i + 1, n):
        #         t_j = target[j].item()
        #         if t_i == t_j:
        #             l_ni = (self.margin - mat[i][target != t_i]).exp().sum()
        #             l_nj = (self.margin - mat[j][target != t_j]).exp().sum()
        #             l_n = (l_ni + l_nj).log()
        #             # Positive component
        #             l_p = mat[i, j]
        #             loss += F.relu(l_n + l_p) ** 2
        #             count += 1

        l2_loss = self.l2_reg*torch.mean(torch.norm(X_embed, p=2, dim=1))
        return loss/count + l2_loss

class SoftTriple(nn.Module):
    def __init__(self, cN, dim, la=20, gamma=0.1, tau=0.2, margin=0.01, K=3, device=torch.device('cuda')):
        super(SoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        self.device = device
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(self.device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T

class ProxyNCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embedding, smoothing_const = 0.1, scaling_x = 1, scaling_p = 3, device='cuda'):
        torch.nn.Module.__init__(self)
        # initialize proxies s.t. norm of each proxy ~1 through div by 8
        # i.e. proxies.norm(2, dim=1)) should be close to [1,1,...,1]
        # TODO: use norm instead of div 8, because of embedding size
        self.proxies = Parameter(torch.randn(nb_classes, sz_embedding) / 8)
        # self.prev_proxies = self.proxies.cpu().data.numpy()
        self.smoothing_const = smoothing_const
        self.scaling_x = scaling_x
        self.scaling_p = scaling_p

    def forward(self, X, T):
        P = F.normalize(self.proxies, p = 2, dim = -1) * self.scaling_p
        X = F.normalize(X, p = 2, dim = -1) * self.scaling_x

        D = torch.cdist(X, P) ** 2
        T = binarize_and_smooth_labels(T, len(P), self.smoothing_const)
        # note that compared to proxy nca, positive included in denominator
        loss = torch.sum(-T * F.log_softmax(-D, -1), -1)
        return loss.mean()


class Smooth_Classifier(nn.Module):
    def __init__(self, params , device='cuda'):
        torch.nn.Module.__init__(self)
        self.train_classes = params.source_classes
        self.c = len(self.train_classes)
        self.smooth_factor = params.smooth_factor
        d = params.embed_dim
        if (params.h_dim > 0):
            self.classifier = nn.Sequential(
                nn.Linear(d, params.h_dim), nn.ReLU(),
                nn.Linear(params.h_dim, len(self.train_classes))).to(device)
        else:
            self.classifier = nn.Sequential(nn.Linear(d, len(self.train_classes))).to(device)


    def forward(self, X, y):
        y_smooth =  binarize_and_smooth_labels(y, self.c, smoothing_const=self.smooth_factor)
        class_out = F.softmax(self.classifier(X), dim=1)
        loss = torch.sum(-y_smooth * torch.log(class_out), -1).mean()
        return loss

class BinomialLoss(nn.Module):
    def __init__(self, alpha=25, beta=0, margin=0.5, hard_mining=None, **kwargs):
        super(BinomialLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets, return_extra_info = False):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets

        base = 0.5
        loss = list()
        c = 0
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])
            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]
            # if self.hard_mining is not None:
            #     neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
            #     pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 < neg_pair_[-1])
            #
            #     if len(neg_pair) < 1 or len(pos_pair) < 1:
            #         c += 1
            #         continue
            #
            #     pos_loss = 2.0 / self.beta * torch.mean(torch.log(1 + torch.exp(-self.beta * (pos_pair - 0.5))))
            #     neg_loss = 2.0 / self.alpha * torch.mean(torch.log(1 + torch.exp(self.alpha * (neg_pair - 0.5))))
            #
            # else:
            pos_pair = pos_pair_
            neg_pair = neg_pair_
            l = 0
            if len(pos_pair) > 0:
                l += torch.mean(torch.log(1 + torch.exp(-2 * (pos_pair - self.margin))))

            if len(neg_pair) > 0:
                l += torch.mean(torch.log(1 + torch.exp(self.alpha * (neg_pair - self.margin))))

            if(torch.is_tensor(l)):
                loss.append(l)
                c += 1

        loss = sum(loss) / n
        prec = float(c) / n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        if(return_extra_info):
            return loss, prec, mean_pos_sim, mean_neg_sim
        else:
            return loss


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes = range(0, nb_classes))
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


