import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
2017 TPAMI Focal Loss for Dense Object Detection
'''
class Focal_Loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        super(Focal_Loss,self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma
        
        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))
        
    def forward(self, logits, labels):
        # assert preds.dim()==2 and labels.dim()==1
        logits = logits.view(-1,logits.size(-1))
        self.alpha = self.alpha.to(logits.device)
        
        logits_logsoft = F.log_softmax(logits, dim=1) # log_softmax
        logits_softmax = torch.exp(logits_logsoft)    # softmax

        logits_softmax = logits_softmax.gather(1,labels.view(-1,1))   
        logits_logsoft = logits_logsoft.gather(1,labels.view(-1,1))
        alpha = self.alpha.gather(0,labels.view(-1))

        loss = -torch.mul(torch.pow((1-logits_softmax), self.gamma), logits_logsoft)  
        loss = torch.mul(alpha, loss.t())

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, beta=0.9999):
        super(CB_loss,self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta       

    def forward(self, logits, labels):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float().to(logits.device) # 将 weights 转换为 PyTorch 的浮点数张量类型
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot # repeat 是 PyTorch 中用于复制张量的函数。它允许在指定的维度上重复张量的元素
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes) # 将 weights 张量在第 1 维度（列维度）上重复 self.no_of_classes 次。这样做的结果是，原始的一维张量在该维度上被扩展，使其大小变为原来的 self.no_of_classes 倍。

        pred = torch.softmax(logits, dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
        return cb_loss
    # if loss_type == "focal":
    #     cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    # elif loss_type == "sigmoid":
    #     cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    # elif loss_type == "softmax":
    #     pred = logits.softmax(dim = 1)
    #     cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    # return cb_loss


class Equal_Loss(nn.Module):
    def __init__(self,gamma=0.9,lambda_=0.00177,image_count_frequency=[None], size_average=True):
        super(Equal_Loss, self).__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.freq_info = torch.FloatTensor(image_count_frequency)
        self.size_average = size_average
        num_class_included = torch.sum(self.freq_info < self.lambda_)

        print('set up Equal_Loss:')
        print('    gamma = {}'.format(self.gamma))
        print('    lambda_ = {}'.format(self.lambda_))
        print('    {} classes included.'.format(num_class_included))

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight

    def forward(self,logits,labels):
        self.n_i, self.n_c = logits.size()
        self.pred_class_logits = logits

        target = F.one_hot(labels, self.n_c).float()
        beta = torch.rand(logits.size(0))
        beta = (beta < self.gamma).float()
        beta = beta.unsqueeze(1).expand(-1,logits.size(1)).to(logits.device)

        eql_w = 1 - beta * self.threshold_func() * (1 - target)
        logits_exp = torch.exp(logits)
        logits_exp_sum = torch.sum(eql_w*logits_exp, dim=1)

        logits_exp = logits_exp.gather(1,labels.view(-1,1))
        loss = -torch.log(torch.div(logits_exp.t(), logits_exp_sum))
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Grad_Libra_Loss(nn.Module):
    def __init__(self, alpha_pos=0, alpha_neg=0, size_average=True):
        super(Grad_Libra_Loss, self).__init__()

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.size_average = size_average

        print('Grad_Libra_Loss:')
        print('    alpha_pos = {}'.format(self.alpha_pos))
        print('    alpha_neg = {}'.format(self.alpha_neg))
    
    def Func(self, grads, alphas):
        return grads - alphas * torch.sin(grads)

    def forward(self,logits,labels):

        preds = torch.softmax(logits, dim=1)
        preds = preds.gather(1,labels.view(-1,1))
        grads = 1- preds
        weights = self.Func(grads, self.alpha_pos)

        loss = -weights*torch.log(preds)

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

if __name__ == '__main__':
    no_of_classes = 3
    logits = torch.rand(2,no_of_classes).float()
    print(logits)
    labels = torch.randint(0,no_of_classes, size = (2,))
    print(labels)

    # loss_fn = Focal_Loss(alpha=[1,2,3], gamma=2, num_classes = no_of_classes, size_average=True)

    loss_fn = CB_loss([10,50,100], no_of_classes, beta=0.9999)

    # loss_fn = Equal_Loss(gamma=0.9,lambda_=0.00177,image_count_frequency=[10,20,30], size_average=True)

    # loss_fn = Grad_Libra_Loss(alpha_pos=0.5, alpha_neg=0, size_average=True)

    loss = loss_fn(logits,labels)
    print(loss)
