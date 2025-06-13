# coding: utf-8
# @Time   : 2021/04/01
# @Author : Xin Zhou
# @Email  : enoche.chow@gmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):

    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = - torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).sum()
        return loss

class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss

class SSLLoss(nn.Module):
    def __init__(self, ssl_temp):
        super(SSLLoss, self).__init__()
        self.ssl_temp = ssl_temp

    def forward(self, uemb1, uemb2, iemb1, iemb2):
        u_norm1, u_norm2 = F.normalize(uemb1), F.normalize(uemb2)
        i_norm1, i_norm2 = F.normalize(iemb1), F.normalize(iemb2)

        u0 = torch.sum(torch.exp(torch.mul(u_norm1, u_norm2) / self.ssl_temp), dim=1)
        # inter views
        u1 = torch.sum(torch.exp(torch.matmul(u_norm1, u_norm2.t()) / self.ssl_temp), dim=1)
        # intra view
        u2 = torch.sum(torch.exp(torch.matmul(u_norm1, u_norm1.t()) / self.ssl_temp), dim=1) - \
             torch.exp(torch.sum(torch.mul(u_norm1, u_norm1), dim=1) / self.ssl_temp)
        u3 = torch.sum(torch.exp(torch.matmul(u_norm2, u_norm2.t()) / self.ssl_temp), dim=1) - \
             torch.exp(torch.sum(torch.mul(u_norm2, u_norm2), dim=1) / self.ssl_temp)
        ssl_u1 = -torch.sum(torch.log(u0 / (u1 + u2 +u0)))
        ssl_u2 = -torch.sum(torch.log(u0 / (u1 + u3 +u0)))

        i0 = torch.sum(torch.exp(torch.mul(i_norm1, i_norm2) / self.ssl_temp), dim=1)
        i1 = torch.sum(torch.exp(torch.matmul(i_norm1, i_norm2.t()) / self.ssl_temp), dim=1)
        i2 = torch.sum(torch.exp(torch.matmul(i_norm1, i_norm1.t()) / self.ssl_temp), dim=1) - \
             torch.exp(torch.sum(torch.mul(i_norm1, i_norm1), dim=1) / self.ssl_temp)
        i3 = torch.sum(torch.exp(torch.matmul(i_norm2, i_norm2.t()) / self.ssl_temp), dim=1) - \
             torch.exp(torch.sum(torch.mul(i_norm2, i_norm2), dim=1) / self.ssl_temp)
        ssl_i1 = -torch.sum(torch.log(i0 / (i1 + i2 +i0)))
        ssl_i2 = -torch.sum(torch.log(i0 / (i1 + i3 +i0)))

        ssl = (ssl_u1 + ssl_u2 + ssl_i1 + ssl_i2)/4
        return ssl


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_u = nn.Bilinear(n_h, n_h, 1)
        self.f_i = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, u_pos1, u_pos2, u_neg1, u_neg2, i_pos1, i_pos2, i_neg1, i_neg2):
        # positive
        u0 = self.f_u(u_pos1, u_pos2).T
        i0 = self.f_i(i_pos1, i_pos2).T

        # negative
        u1 = self.f_u(u_pos1, u_neg1).T
        u2 = self.f_u(u_pos1, u_neg2).T
        u3 = self.f_u(u_pos2, u_neg1).T
        u4 = self.f_u(u_pos2, u_neg2).T

        i1 = self.f_i(i_pos1, i_neg1).T
        i2 = self.f_i(i_pos1, i_neg2).T
        i3 = self.f_i(i_pos2, i_neg1).T
        i4 = self.f_i(i_pos2, i_neg2).T

        logits = torch.cat((u0, i0, u1, u2, u3, u4, i1, i2, i3, i4), 1)
        return logits
    

class SSLLoss1(nn.Module):
    def __init__(self, ssl_temp):
        super(SSLLoss1, self).__init__()
        self.ssl_temp = ssl_temp
        self.k = 30
        self.temp =50

    def forward(self, uemb1, uemb2, iemb1, iemb2):
        u_norm1, u_norm2 = F.normalize(uemb1), F.normalize(uemb2)
        i_norm1, i_norm2 = F.normalize(iemb1), F.normalize(iemb2)
        
        top_u1_mat = self.cos_matrix(u_norm1)
        top_u2_mat = self.cos_matrix(u_norm2)
        top_i1_mat = self.cos_matrix(i_norm1)
        top_i2_mat = self.cos_matrix(i_norm2)
        
        u_0 = torch.sum(torch.exp(torch.mul(torch.matmul(u_norm1, u_norm1.T), top_u1_mat) / self.temp), dim=1) +\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm1, u_norm2.T), top_u2_mat) / self.temp), dim=1) -\
                torch.sum(torch.exp(torch.mul(u_norm1, u_norm1) / self.ssl_temp))   #pos
        u_1 = torch.sum(torch.exp(torch.matmul(u_norm1, u_norm1.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm1, u_norm1.T) ,top_u1_mat))) +\
                torch.sum(torch.exp(torch.mul(u_norm1, u_norm1) / self.ssl_temp))   #neg -1
        u_2 =torch.sum(torch.exp(torch.matmul(u_norm1, u_norm2.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm1, u_norm2.T) ,top_u2_mat))) #neg-2
        
        u1_0 = torch.sum(torch.exp(torch.mul(torch.matmul(u_norm2, u_norm2.T), top_u2_mat) / self.temp), dim=1) +\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm2, u_norm1.T), top_u1_mat) / self.temp), dim=1) -\
                torch.sum(torch.exp(torch.mul(u_norm2, u_norm2) / self.ssl_temp))   #pos
        u1_1 = torch.sum(torch.exp(torch.matmul(u_norm2, u_norm2.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm2, u_norm2.T) ,top_u2_mat))) +\
                torch.sum(torch.exp(torch.mul(u_norm2, u_norm2) / self.ssl_temp))   #neg -1
        u1_2 =torch.sum(torch.exp(torch.matmul(u_norm2, u_norm1.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(u_norm2, u_norm1.T) ,top_u1_mat))) #neg-2
        
        ssl_u1 = -torch.sum(torch.log(u_0 / u_0 + u_1 + u_2))
        ssl_u2 = -torch.sum(torch.log(u1_0 / u1_0 + u1_1 + u1_2))
        
        i_0 = torch.sum(torch.exp(torch.mul(torch.matmul(i_norm1, i_norm1.T), top_i1_mat) / self.temp), dim=1) +\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm1, i_norm2.T), top_i2_mat) / self.temp), dim=1) -\
                torch.sum(torch.exp(torch.mul(i_norm1, i_norm1) / self.ssl_temp))   #pos
        i_1 =torch.sum(torch.exp(torch.matmul(i_norm1, i_norm1.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm1, i_norm1.T) ,top_i1_mat))) +\
                torch.sum(torch.exp(torch.mul(i_norm1, i_norm1) / self.ssl_temp))   #pos
        i_2 =torch.sum(torch.exp(torch.matmul(i_norm1, i_norm2.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm1, i_norm2.T) ,top_i2_mat)))
        
        i1_0 = torch.sum(torch.exp(torch.mul(torch.matmul(i_norm2, i_norm2.T), top_i2_mat) / self.temp), dim=1) +\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm2, i_norm1.T), top_i1_mat) / self.temp), dim=1) -\
                torch.sum(torch.exp(torch.mul(i_norm2, i_norm2) / self.ssl_temp))   #pos
        i1_1 =torch.sum(torch.exp(torch.matmul(i_norm2, i_norm2.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm2, i_norm2.T) ,top_i2_mat))) +\
                torch.sum(torch.exp(torch.mul(i_norm2, i_norm2) / self.ssl_temp))   #pos
        i1_2 =torch.sum(torch.exp(torch.matmul(i_norm2, i_norm1.T))) -\
            torch.sum(torch.exp(torch.mul(torch.matmul(i_norm2, i_norm1.T) ,top_i1_mat)))
        
        
        ssl_i1 = -torch.sum(torch.log(i_0 / i_0 + i_1 + i_2))
        ssl_i2 = -torch.sum(torch.log(i1_0 / i1_0 + i1_1 + i1_2))
        
        ssl = (ssl_i1 + ssl_u1 + ssl_i2 + ssl_u2) / 4
        return ssl
    
    def cos_matrix(self ,matrix):
        cos_matrix = torch.matmul(matrix, matrix.t())  ##u1  u1
        top_matrix = torch.topk(cos_matrix, self.k, dim=1, largest=True)
        topk_u_mat = torch.zeros_like(cos_matrix)
        for i in range(matrix.shape[0]):
            topk_u_mat[i, top_matrix.indices[i]] = 1
        return topk_u_mat

    
    
    