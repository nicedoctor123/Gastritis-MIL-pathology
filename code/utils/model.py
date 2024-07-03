import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch.nn import ReLU, LayerNorm

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):

        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()] 
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()] 
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b) 
        A = self.attention_c(A)  
        return A, x


#----------------------------------------------------------------
class ABMIL(nn.Module):
    def __init__(self, size_arg = "small", dropout=0.25, n_classes=1, feat_size=1024):

        super(ABMIL, self).__init__()
        self.size_dict_path = {"small": [feat_size, feat_size//2, feat_size//4], "big": [feat_size, feat_size//2, feat_size*3//8]}
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc) 
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)]) 

        self.classifier = nn.Linear(size[2], n_classes)
        self.simpleRes = nn.Linear(size[0],size[0]) 

    def relocate(self): 
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:7')

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)
    
    def get_attention_scores(self,x):
        A, h_path = self.attention_net(x)  
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        return A

    def forward(self,x):
        # x = x + self.simpleRes(x)
        A, h_path = self.attention_net(x)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path) 
        h_path = self.rho(h_path).squeeze()
        h = h_path 

        logits  = self.classifier(h).unsqueeze(0) 

        return logits
#----------------------------------------------------------------
#----------------------------------------------------------------
# TransMIL 
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):        
            nn.init.xavier_normal_(m.weight)
            try:
                m.bias.data.zero_() 
            except:
                print(m)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,   
            pinv_iterations = 6,    
            residual = True,         
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, feat_size= 1024):
        super(TransMIL, self).__init__()
        # self.norm_init=(1024)
        self.pos_layer = PPEG(dim=feat_size//2)
        self._fc1 = nn.Sequential(nn.Linear(feat_size, feat_size//2), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, feat_size//2))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=feat_size//2)
        self.layer2 = TransLayer(dim=feat_size//2)
        self.norm = nn.LayerNorm(feat_size//2)
        self._fc2 = nn.Linear(feat_size//2,self.n_classes)
        initialize_weights(self)
        self.simpleRes = nn.Linear(feat_size,feat_size)

    def forward(self, data_x):

        h = data_x.unsqueeze(0) #[B, n, 1024]
        # h = self.norm_init(h)
        # h = h + self.simpleRes(h)
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]
        
        #---->predict
        logits = self._fc2(h)  #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
#         results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        # return torch.sigmoid(logits), Y_prob, Y_hat
        return logits
#----------------------------------------------------------------
#----------------------------------------------------------------
# PatchGCN
class PatchGCN(nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial', multires=False, resample=0,
        fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=4,dataname='ucec'):
        super(PatchGCN, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg             # edge_agg: 边的聚合方式。
        self.multires = multires             # multires: 是否使用多分辨率特征。
        self.num_layers = num_layers - 1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(0.25)]) 

        self.layers = torch.nn.ModuleList() 
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer') 
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(0.25)]) 

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*4, D=hidden_dim*4, dropout=dropout, n_classes=1) 
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*4, hidden_dim*4), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*4, n_classes)    

    def forward(self, x, edge_index):
        
        if self.edge_agg == 'spatial':
            edge_index = edge_index

        edge_attr = None
        
        # print(x.size())
        # print(x)
        x = self.fc(x)
        x_ = x 
        
        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        
        return logits
#----------------------------------------------------------------
#----------------------------------------------------------------
# DSMIL
class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x
    
# BClassifier (Bayes Classifier)
class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.simpleRes = nn.Linear(1024,1024)
    def forward(self, x):
        # x = x + self.simpleRes(x)
        feats, classes = self.i_classifier(x) 
        prediction_bag, A, B = self.b_classifier(feats, classes)
        prediction_max, _ = torch.max(classes, 0)
        # print(prediction_max.shape, '---------------',prediction_bag.shape)
        # input()
        # return classes, prediction_bag, A, B
        return 0.5*(prediction_max.view(1,-1)+prediction_bag.view(1,-1))
#----------------------------------------------------------------

#----------------------------------------------------------------
# CLAM
class Clam_Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Clam_Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Clam_Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Clam_Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""
class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, feat_size=1024):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [feat_size, feat_size//2, feat_size//4], "big": [feat_size, feat_size//2, feat_size*3//8]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Clam_Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Clam_Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        self.simpleRes = nn.Linear(size[0],size[0])

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        # h =  h + self.simpleRes(h)
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        # return logits, Y_prob, Y_hat, A_raw, results_dict
        return logits, total_inst_loss
#----------------------------------------------------------------