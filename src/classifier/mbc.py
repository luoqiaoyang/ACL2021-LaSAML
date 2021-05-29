import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class MBC(BASE):
    '''
        Metric-based Classifier FOR FEW SHOT LEARNING
    '''
    def __init__(self, ebd_dim, args):
        super(MBC, self).__init__(args)
        self.args = args
        self.ebd_dim = ebd_dim

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
        
        if self.args.sim == "l2":
            
            pred = -self._compute_l2(prototype, XQ)
            # pred = pred / torch.mean(pred, dim=1).unsqueeze(1)
        elif self.args.sim == "cos":
            pred = -self._compute_cos(prototype, XQ)

        
        loss = F.cross_entropy(pred, YQ)

        acc = BASE.compute_acc(pred, YQ)            

        return acc, loss

    def _compute_l2(self, XS, XQ):
        '''
            Compute the pairwise l2 distance
            @param XS (support x): support_size x ebd_dim
            @param XQ (support x): query_size x ebd_dim

            @return dist: query_size x support_size

        '''
        if self.args.que_feature == "tag":
            diff = XS.unsqueeze(0).unsqueeze(2) - XQ.unsqueeze(1)
            dist = torch.norm(diff, dim=3)
            tmp_dist = [torch.diag(dist[i], 0) for i in range(dist.shape[0])]
            dist = torch.stack(tmp_dist)
        else:
            diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
            dist = torch.norm(diff, dim=2)

        return dist