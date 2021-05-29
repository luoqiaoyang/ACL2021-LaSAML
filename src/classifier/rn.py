import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class RN(BASE):
    '''
        "Relation Networks for Few-Shot Text Classification"
    '''
    def __init__(self, ebd_dim, args):
        super(RN, self).__init__(args)
        self.args = args

        self.ebd_dim = ebd_dim

        h = 50
        # h = self.args.induct_hidden_dim

        if self.args.embedding == 'meta':
            print('No relation module. Use Prototypical network style prediction')
        else:  # follow the original paper
            # self.Ws = nn.Linear(self.ebd_dim, self.ebd_dim)
            self.M = nn.Parameter(torch.Tensor(h, 1, 1, self.ebd_dim, self.ebd_dim).uniform_(-0.1,0.1))
            self.rel = nn.Linear(h, 1)
        # self.batchnorm = nn.BatchNorm1d(self.args.way)
        self.layernorm = nn.LayerNorm(h)

    def _compute_prototype(self, XS, YS):
        '''
            Compute the prototype for each class by averaging over the ebd.
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together
        sorted_YS, indices = torch.sort(YS)
        sorted_XS = XS[indices]

        prototype = []
        for i in range(self.args.way):
            prototype.append(torch.mean(
                sorted_XS[i*self.args.shot:(i+1)*self.args.shot], dim=0,
                keepdim=True))

        prototype = torch.cat(prototype, dim=0)

        return prototype

    def _compute_relation_score(self, prototype, XQ):
        '''
            Compute the relation score between each prototype and each query
            example

            @param prototype: way x ebd_dim
            @param XQ: query_size x ebd_dim

            @return score: query_size x way
        '''
        # import pdb; pdb.set_trace()
        prototype = prototype.unsqueeze(0).unsqueeze(0).unsqueeze(-2)
        # 1, 1, way, 1, ebd_dim
        XQ = XQ.unsqueeze(1).unsqueeze(-1).unsqueeze(0)
        # 1, query_size, 1, ebd_dim, 1

        score = torch.matmul(torch.matmul(prototype, self.M),
                             XQ)
        # h, query_size, way, 1, 1

        score = score.squeeze(-1).squeeze(-1).permute(1, 2, 0)
        # query_size, way, h

        
        score = F.relu(score)
        score = self.layernorm(score)
        # score = score.permute(0,2,1)
        # score = self.batchnorm(score)
        
        # score = score.permute(0,2,1)
        score = torch.sigmoid(self.rel(score)).squeeze(-1)

        return score

    def forward(self, XS, YS, XQ, YQ):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''

        YS, YQ = self.reidx_y(YS, YQ)

        prototype = self._compute_prototype(XS, YS)

        # # use parameter free comparison when distributional signatures are
        # # used
        # score = -self._compute_l2(prototype, XQ)
        # # score = -self._compute_cos(prototype, XQ)
        # # l2 and cos deosn't have much diff empirically across the 6
        # # datasets

        # loss = F.cross_entropy(score, YQ)


        # implementation based on the original paper
        score = self._compute_relation_score(prototype, XQ)

        # use regression as training objective
        YQ_onehot = self._label2onehot(YQ)

        loss = torch.sum((YQ_onehot.float() - score) ** 2)

        acc = BASE.compute_acc(score, YQ)

        return acc, loss
