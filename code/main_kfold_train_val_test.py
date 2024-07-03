import os
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import torch
import torch.nn as nn
softmax = nn.Softmax(dim=1)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import sys
from utils.dataset import WSIDataset, PatchGCN_Dataset
from utils.loss import Focal_Loss, Equal_Loss, Grad_Libra_Loss, CB_loss
from utils.model import ABMIL, TransMIL, FCLayer, BClassifier, MILNet, PatchGCN, CLAM_SB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix,precision_score,recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

def set_random_seed(seed):
    random.seed(seed)                 
    np.random.seed(seed)              
    torch.manual_seed(seed)           
    torch.cuda.manual_seed(seed)      
    torch.cuda.manual_seed_all(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.enabled = True 

def creat_dirs_for_result(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%H%M%S") 
    train_name = str(timestamp)+'_'+str(args.model_name)+'_'+str(args.label_name)
    log_dir = os.path.join(args.output_dir, train_name, 'logs')
    model_dir = os.path.join(args.output_dir, train_name, 'models')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir

def save_args_json(data, log_dir):

    file = os.path.join(log_dir, 'args.json')
    if not os.path.exists(file):
        with open(file, 'w') as f:  
            json.dump(data, f, indent=4) 
def get_logger(log_dir, name=None): 
    logger = logging.getLogger(name) 
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(message)s')

    file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'train.log'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handle=logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)

    logger.addHandler(file_handler) 
    logger.addHandler(console_handle)

    return logger

def train_one_epoch(model, train_loader, loss_fn, device, optimizer, batch_size):
    running_loss = 0.0 
    model.train() 
    slide_ids, preds, labels, probs= [],[],[],[]

    batch_out, batch_pred, batch_label = [],[],[]
    

    for i,(slide_id, bag, label) in enumerate(train_loader): 
        slide_id = slide_id[0]

        bag = bag.squeeze(0).to(device) 
        label = label.squeeze(0).to(device) 
   

        output = model(bag) 
        _, pred = torch.max(output, dim=1)  
        
        slide_ids.append(slide_id) 
        preds.append(pred.detach().item()) 
        labels.append(label.detach().item()) 
        probs.append(softmax(output).detach().cpu().numpy().squeeze(0)) 

        batch_out.append(output)
        batch_pred.append(pred.detach().item())
        batch_label.append(label)

        if (i+1) % batch_size == 0 or i == len(train_loader) - 1: 
            batch_out = torch.cat(batch_out) 
            batch_label = torch.tensor(batch_label, device=device)

            loss = loss_fn(batch_out, batch_label) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            batch_label= [lab.detach().item() for lab in batch_label]
            print(f'loss: {loss.item()}, label: {batch_label}, pred: {batch_pred} ') 
    
       
            running_loss += loss.item()*len(batch_label) 
            batch_out, batch_pred, batch_label = [],[],[] 

    epoch_loss = running_loss / len(train_loader.dataset) 

    return epoch_loss, slide_ids, labels, preds, probs 

def evaluate(model, dataloader, loss_fn, device):
    model.eval() 
    with torch.no_grad(): 
        running_loss = 0.0
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0).to(device)

            label = label.to(device)
            output = model(bag)
            _, pred = torch.max(output, dim=1) 
            loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += loss.item()
        loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(labels, preds)   
    return loss, slide_ids, labels, preds, probs

def PatchGCN_train_one_epoch(model, train_loader, loss_fn, device, optimizer, batch_size):
    running_loss = 0.0
    model.train()

    slide_ids, preds, labels, probs= [],[],[],[]

    batch_out, batch_pred, batch_label = [],[],[]

    for i,(slide_id, x, adj, label) in enumerate(train_loader):
        slide_id = slide_id[0]
        x = x.squeeze(0).to(device)
        adj = adj.squeeze(0).to(device)
        label = label.squeeze(0).to(device)

        output = model(x,adj)

        _, pred = torch.max(output, dim=1) 
        slide_ids.append(slide_id)
        preds.append(pred.detach().item())
        labels.append(label.detach().item())
        probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

        batch_out.append(output)
        batch_pred.append(pred.detach().item())
        batch_label.append(label)

        if (i+1) % batch_size == 0 or i == len(train_loader) - 1:
            batch_out = torch.cat(batch_out)
            batch_label = torch.tensor(batch_label, device=device)

            loss = loss_fn(batch_out, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_label= [lab.detach().item() for lab in batch_label]
            print(f'loss : {loss.item()}, label : {batch_label}, pred: {batch_pred}')

            running_loss += loss.item()*len(batch_label)
            batch_out, batch_pred, batch_label = [],[],[]
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, slide_ids, labels, preds, probs

def PatchGCN_evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, x, adj, label in dataloader:
            slide_id = slide_id[0]
            x = x.squeeze(0).to(device)
            adj = adj.squeeze(0).to(device)
            
            label = label.to(device)
            output = model(x,adj)
            _, pred = torch.max(output, dim=1)  
            loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += loss.item()
        
        loss = running_loss / len(dataloader.dataset)
    return loss, slide_ids, labels, preds, probs

def CLAM_train_one_epoch(model, train_loader, loss_fn, device, optimizer, batch_size):
    running_loss = 0.0
    model.train()

    slide_ids, preds, labels, probs= [],[],[],[]

    batch_out, batch_pred, batch_label, batch_inst_loss = [],[],[],[]

    for i,(slide_id, bag, label) in enumerate(train_loader):
        slide_id = slide_id[0]
        bag = bag.squeeze(0).to(device)
        label = label.squeeze(0).to(device)

        output, inst_loss = model(bag, label=label, instance_eval=True) 
        _, pred = torch.max(output, dim=1)  

        slide_ids.append(slide_id)
        preds.append(pred.detach().item())
        labels.append(label.detach().item())
        probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

        batch_out.append(output)
        batch_pred.append(pred.detach().item())
        batch_label.append(label)
        batch_inst_loss.append(inst_loss)

        if (i+1) % batch_size == 0 or i == len(train_loader) - 1:
            batch_out = torch.cat(batch_out)
            batch_label = torch.tensor(batch_label, device=device)

            bag_loss = loss_fn(batch_out, batch_label)
            inst_loss = sum(batch_inst_loss)/len(batch_inst_loss)
            loss = 0.7*bag_loss + 0.3*inst_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_label= [lab.detach().item() for lab in batch_label]
            print(f'bag_loss: {bag_loss.item()}, inst_loss: {inst_loss.item()}, label: {batch_label}, pred: {batch_pred} ')

            running_loss += loss.item()*len(batch_label)
            batch_out, batch_pred, batch_label, batch_inst_loss = [],[],[],[]
    
    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss, slide_ids, labels, preds, probs

def CLAM_evaluate(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        slide_ids, preds, labels, probs= [],[],[],[]

        for slide_id, bag, label in dataloader:
            slide_id = slide_id[0]
            bag = bag.squeeze(0).to(device)

            label = label.to(device)
            output, inst_loss = model(bag, label=label, instance_eval=True)
            _, pred = torch.max(output, dim=1) 
            bag_loss = loss_fn(output, label)

            slide_ids.append(slide_id)
            preds.append(pred.detach().item())
            labels.append(label.detach().item())
            probs.append(softmax(output).detach().cpu().numpy().squeeze(0))

            running_loss += (0.7*bag_loss.item() + 0.3*inst_loss.item())
        
        loss = running_loss / len(dataloader.dataset)

    return loss, slide_ids, labels, preds, probs

def train_and_eval(args, train_set, valid_set, test_set, class_feat):

    train_loader = DataLoader(train_set, batch_size=1, drop_last=False, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, drop_last=False, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
    

    device = torch.device(args.gpus) 
    if args.model_name == 'ABMIL':
        model = ABMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'TransMIL':
        model = TransMIL(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'DSMIL':
        i_classifier = FCLayer(in_size=args.feat_size, out_size=args.num_class)
        b_classifier = BClassifier(input_size=args.feat_size, output_class=args.num_class)
        model = MILNet(i_classifier, b_classifier).to(device)
    elif args.model_name == 'CLAM':
        model = CLAM_SB(n_classes=args.num_class, feat_size= args.feat_size).to(device)
    elif args.model_name == 'PatchGCN':
        model = PatchGCN(n_classes=args.num_class, num_features= args.feat_size).to(device)
    else:
        raise NotImplementedError(f'no model:{args.model_name}')

    if args.weighted_loss:
        loss_fn = torch.nn.CrossEntropyLoss(weight= torch.FloatTensor(class_feat[args.label_name][3]).to(device))
    else:
        # loss_fn = nn.CrossEntropyLoss()
        # Focal_Loss
        # loss_fn = Focal_Loss(alpha=None, gamma=2, num_classes = args.num_class, size_average=True)
        # CB_loss 
        # loss_fn = CB_loss(samples_per_cls=[1161,219,213,113], no_of_classes=args.num_class)
        # Equalization loss 
        # loss_fn = Equal_Loss(gamma=0.9,lambda_=0.00177,image_count_frequency=[675,1079,172], size_average=True)
        # Grad_Libra loss 
        loss_fn = Grad_Libra_Loss(alpha_pos = args.alpha_pos, alpha_neg = args.alpha_neg, size_average=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= args.epochs, eta_min=args.lr_min) 


    log_dir, model_dir = creat_dirs_for_result(args) 
    save_args_json(args.__dict__, log_dir) 
    writer = SummaryWriter(log_dir)
    logger = get_logger(log_dir, name=args.label_name+str(datetime.now().strftime("_%Y%m%d%H%M%S"))) 

    best_acc = 0.0
    kfold_test_acc = 0.0
    kfold_test_probs = []
    kfold_test_preds = []
    model_save_list = []
    for epoch in tqdm(range(args.epochs)):

        if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
            train_loss, train_ids, train_labels, train_preds, train_probs = train_one_epoch(model, train_loader, loss_fn, device, optimizer, args.batch_size)
        elif args.model_name == 'CLAM':
            train_loss, train_ids, train_labels, train_preds, train_probs = CLAM_train_one_epoch(model, train_loader, loss_fn, device, optimizer, args.batch_size)
        elif args.model_name == 'PatchGCN':
            train_loss, train_ids, train_labels, train_preds, train_probs = PatchGCN_train_one_epoch(model, train_loader, loss_fn, device, optimizer, args.batch_size)
        train_acc = accuracy_score(train_labels, train_preds)
        print('[epoch %d] train_loss: %.3f train_acc: %.3f' % (epoch + 1, train_loss, train_acc)) # %d取整
        scheduler.step()

        if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
            val_loss, val_ids, val_labels, val_preds, val_probs = evaluate(model, valid_loader, loss_fn, device)
        elif args.model_name == 'CLAM':
            val_loss, val_ids, val_labels, val_preds, val_probs = CLAM_evaluate(model, valid_loader, loss_fn, device)
        elif args.model_name == 'PatchGCN':
            val_loss, val_ids, val_labels, val_preds, val_probs = PatchGCN_evaluate(model, valid_loader, loss_fn, device)
        val_acc = accuracy_score(val_labels, val_preds)
        print('[epoch %d] val_loss: %.3f val_acc: %.3f' % (epoch + 1, val_loss, val_acc))

        if args.model_name in ['ABMIL', 'TransMIL', 'DSMIL']:
            test_loss, test_ids, test_labels, test_preds, test_probs = evaluate(model, test_loader, loss_fn, device)
        elif args.model_name == 'CLAM':
            test_loss, test_ids, test_labels, test_preds, test_probs = CLAM_evaluate(model, test_loader, loss_fn, device)
        elif args.model_name == 'PatchGCN':
            test_loss, test_ids, test_labels, test_preds, test_probs = PatchGCN_evaluate(model, test_loader, loss_fn, device)
        test_acc = accuracy_score(test_labels, test_preds)
        print('[epoch %d] test_loss: %.3f test_acc: %.3f' % (epoch + 1, test_loss, test_acc))

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        writer.add_scalar('train_loss', train_loss, epoch + 1)
        writer.add_scalar('val_loss', val_loss, epoch + 1)
        writer.add_scalar('test_loss', val_loss, epoch + 1)
        writer.add_scalar('train_acc', train_acc, epoch + 1)
        writer.add_scalar('val_acc', val_acc, epoch + 1)
        writer.add_scalar('test_acc', test_acc, epoch + 1)
        logger.info('[epoch %d] train_loss: %.4f train_acc: %.4f val_loss: %.4f val_acc: %.4f test_loss: %.4f  test_acc: %.4f' % 
                     (epoch + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
        

        if val_acc > best_acc:
            best_acc = val_acc
            kfold_test_acc = test_acc
            kfold_test_preds = test_preds
            kfold_test_probs = test_probs

            model_save_name = f'model_e{epoch+1:d}_{best_acc:.4f}.pth'
            model_save_list.append(model_save_name)
            torch.save(model.state_dict(), os.path.join(model_dir, model_save_name))
            if len(model_save_list)> args.model_save_num and os.path.isfile(os.path.join(model_dir, model_save_list[-(args.model_save_num+1)])):
                os.remove(os.path.join(model_dir, model_save_list[-(args.model_save_num+1)]))

            csv = {'slide_id':train_ids+val_ids+test_ids, args.label_name:train_labels+val_labels+test_labels,
                   'pred':train_preds+val_preds+test_preds, 'prob':train_probs+val_probs+test_probs}
            csv = pd.DataFrame(csv)
            csv.to_csv(os.path.join(log_dir, 'best_epoch_output.csv'),index=False)

            NPV_values, Recall_values, Precision_values, CUI_neg_values, CUI_pos_values = [], [], [], [], []
            conf_matrix = confusion_matrix(test_labels, test_preds)
            for s in range(args.num_class): 
                 TN = np.sum(np.delete(np.delete(conf_matrix, s, axis=0), s, axis=1))  # True Negatives
                 FN = np.sum(conf_matrix[s]) - conf_matrix[s][s] # False Negatives
                 NPV = TN / (TN + FN)
                 Recall = conf_matrix[s][s] / (sum(conf_matrix[s][j] for j in range(args.num_class)))
                 Precision = conf_matrix[s][s] / (sum(conf_matrix[j][s] for j in range(args.num_class)))

                 NPV_values.append(NPV * Precision)
                 Recall_values.append(Recall)
                 Precision_values.append(Precision)
                 CUI_pos_values.append(Precision * Recall)
                 CUI_neg_values.append(NPV * Recall)

                 NPV_avg = round(np.mean(NPV_values), 4)
                 Recall_avg = round(np.mean(Recall_values), 4)
                 Precision_avg = round(np.mean(Precision_values), 4)
                 CUI_pos_avg = round(np.mean(CUI_pos_values), 4)
                 CUI_neg_avg = round(np.mean(CUI_neg_values), 4)


            with open(os.path.join(log_dir, 'best_epoch.txt'), mode='w') as file:
                file.write('{:d}\n'.format(epoch+1))
                file.write('{:.4f}\n'.format(train_acc))
                file.write('{:.4f}\n'.format(val_acc))
                file.write('{:.4f}\n'.format(test_acc))
                file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovr")))
                file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="macro", multi_class="ovo")))
                file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovr")))
                # file.write('micro_auc_ovo {:.4f}\n'.format(roc_auc_score(test_labels, test_probs, average="micro", multi_class="ovo")))#这种算不了
                file.write(str(classification_report(test_labels, test_preds)))
                file.write(str(confusion_matrix(train_labels, train_preds))+'\n')
                file.write(str(confusion_matrix(val_labels, val_preds))+'\n')
                file.write(str(confusion_matrix(test_labels, test_preds))+'\n')
                file.write(f'avg: {CUI_pos_avg},{CUI_neg_avg}')

    # save the latest_model
    model_save_name = 'model_latest.pth'
    model_save_list.append(model_save_name)
    torch.save(model.state_dict(), os.path.join(model_dir, model_save_name))
    
    csv = pd.DataFrame({'model_save_list':model_save_list})
    csv.to_csv(os.path.join(model_dir, 'model_save_list.txt'),index=False)

    return kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=..., help='Random seed to use.')
    parser.add_argument('--gpus', type=int, default=0, help="GPU indices ""comma separated, e.g. '0,1' ")
    parser.add_argument('--fea_dir',type=str, default='...',help='.h5 featrure files extracted by CLAM')
    parser.add_argument('--label_dir',type=str, default='...', help='splitted label file')
    parser.add_argument('--fold_num',type=int, default=5)
    parser.add_argument('--output_dir', default='...', help='Path to experiment output, config, checkpoints, etc.')

    parser.add_argument('--model_name',type=str, default='...', help='What model architecture to use.')
    parser.add_argument('--alpha_pos',type = float, default=0.5, help='the positive value of Grad_Libra_Loss alpha.')
    parser.add_argument('--alpha_neg',type = float, default=0.5, help='the negtive value of Grad_Libra_Loss alpha.')
    parser.add_argument('--label_name',type=str, default='...', help='What label to use.')
    parser.add_argument('--lr', type=float, default=..., help='max Learning rate of CosineAnnealingLR.')
    parser.add_argument('--lr_const', type=bool, default=..., help='max Learning rate of CosineAnnealingLR.')
    parser.add_argument('--lr_min', type=float, default=..., help='Dmin Learning rate of CosineAnnealingLR.')
    parser.add_argument('--wd', type=float, default=..., help=' weight decay of optimizer')
    parser.add_argument('--weighted_loss', type=bool, default=False, help=' weight of loss')
    parser.add_argument('--batch_size', type=int, default=..., help='Dataloaders batch size.')
    parser.add_argument('--feat_size', default=..., type=int, help='Dimension of the feature size [1024]')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--model_save_num', type=int, default=2, help='Number of models saved during train.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = args_parser()
    set_random_seed(args.seed)
    class_feat ={
                  'label1': [3,'256_50_1024',[0.32, 0.22, 0.46],[2.85, 1.78, 11.20]],
                  'label2': [3,'256_50_1024',[0.32, 0.15, 0.25, 0.28],[19.26, 1.86, 3.85, 6.69]],
                  'label3': [4,'256_50_1024',[0.22, 0.20, 0.27, 0.32],[2.84, 2.50, 4.98, 21.64]],
                  'label4': [4,'256_50_1024',[0.15, 0.26, 0.28, 0.31],[1.77, 4.86, 5.85, 16.75]],
                  }

    args.num_class = class_feat[args.label_name][0]

    if args.model_name != 'PatchGCN':
        args.fea_dir = os.path.join(args.fea_dir, 'h5_files')
    else:
        args.fea_dir = os.path.join(args.fea_dir, 'graph_files')
    
    feat_name = args.fea_dir.split('/')[-2]
    args.feat_size = int(feat_name.split('_')[-1])
    print(args.label_name, args.model_name, 'feat_name:', feat_name, '\n')

    args.output_dir = os.path.join(args.output_dir, args.model_name, args.label_name,feat_name) 

    print(args.output_dir, '\n')
    print(os.path.join(args.label_dir, args.label_name+'.txt'))
    df = pd.read_csv(os.path.join(args.label_dir, args.label_name+'.txt'))
    df = df[[df.keys()[0], args.label_name]]

    skf=StratifiedKFold(n_splits=args.fold_num, random_state=args.seed, shuffle=True)
    test_acc_list, slide_ids, labels, preds, probs = [],[],[],[],[]
    for i, (train_index,test_index) in enumerate(skf.split(df[df.keys()[0]], df[args.label_name])):

        timestamp = datetime.now().strftime("%m%d%H%M%S")
        temp_train_csv = os.path.join(args.label_dir, args.label_name+'.'+timestamp+'_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_train.csv')
        temp_valid_csv = os.path.join(args.label_dir, args.label_name+'.'+timestamp+'_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_valid.csv')
        temp_test_csv = os.path.join(args.label_dir, args.label_name+'.'+timestamp+'_'+args.model_name+'_'+feat_name+'_'+str(i+1)+'fold_test.csv')

        train_df = df.iloc[train_index]

        train_df, valid_df = train_test_split(train_df, test_size=1/(args.fold_num-1), stratify = train_df[args.label_name])

        train_df.to_csv(temp_train_csv,index=0)
        valid_df.to_csv(temp_valid_csv,index=0)
        df.iloc[test_index].to_csv(temp_test_csv,index=0)
        if args.model_name != 'PatchGCN':
            train_set = WSIDataset(args.fea_dir, temp_train_csv, preload = True)
            valid_set = WSIDataset(args.fea_dir, temp_valid_csv, preload = True)
            test_set =  WSIDataset(args.fea_dir, temp_test_csv, preload = True)
        else:
            train_set = PatchGCN_Dataset(args.fea_dir, temp_train_csv, preload = True)
            valid_set = PatchGCN_Dataset(args.fea_dir, temp_valid_csv, preload = True)
            test_set =  PatchGCN_Dataset(args.fea_dir, temp_test_csv, preload = True)

        if args.lr_const: 
            args.lr_min = args.lr
        else:
            args.lr_min = args.lr/10
        
        kfold_test_acc, test_ids, test_labels, kfold_test_preds, kfold_test_probs = train_and_eval(args, train_set, valid_set, test_set, class_feat)

        test_acc_list.append(kfold_test_acc)
        slide_ids += test_ids
        labels    += test_labels
        preds     += kfold_test_preds
        probs     += kfold_test_probs

        if os.path.isfile(temp_train_csv):
            os.remove(temp_train_csv)
        if os.path.isfile(temp_valid_csv):
            os.remove(temp_valid_csv)
        if os.path.isfile(temp_test_csv):
            os.remove(temp_test_csv)

    NPV_values_kfold, Recall_values_kfold, Precision_values_kfold, CUI_pos_values_kfold,CUI_neg_values_kfold = [], [], [], [], []
    conf_matrix_kfold = confusion_matrix(labels, preds)
    for s in range(args.num_class): 
        TN = np.sum(np.delete(np.delete(conf_matrix_kfold, s, axis=0), s, axis=1))  
        FN = np.sum(conf_matrix_kfold[s]) - conf_matrix_kfold[s][s] 
        NPV = TN / (TN + FN)
        Recall = conf_matrix_kfold[s][s] / (sum(conf_matrix_kfold[s][j] for j in range(args.num_class)))
        Precision = conf_matrix_kfold[s][s] / (sum(conf_matrix_kfold[j][s] for j in range(args.num_class)))

        NPV_values_kfold.append(NPV)
        Recall_values_kfold.append(Recall)
        Precision_values_kfold.append(Precision)
        CUI_pos_values_kfold.append(Precision * Recall)
        CUI_neg_values_kfold.append(NPV * Recall)

        NPV_avg = round(np.mean(NPV_values_kfold), 4)
        Recall_avg = round(np.mean(Recall_values_kfold), 4)
        Precision_avg = round(np.mean(Precision_values_kfold), 4)
        CUI_pos_avg = round(np.mean(CUI_pos_values_kfold), 4)
        CUI_neg_avg = round(np.mean(CUI_neg_values_kfold), 4)

    with open(os.path.join(args.output_dir, 'kfold_result.txt'), mode='w') as file:
        file.write(str(test_acc_list)+'\n')
        file.write('average {:.4f}\n'.format(sum(test_acc_list)/len(test_acc_list)))
        file.write('macro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovr")))
        file.write('macro_auc_ovo {:.4f}\n'.format(roc_auc_score(labels, probs, average="macro", multi_class="ovo")))
        file.write('micro_auc_ovr {:.4f}\n'.format(roc_auc_score(labels, probs, average="micro", multi_class="ovr")))

        file.write(str(classification_report(labels, preds, digits=4)))
        file.write(str(confusion_matrix(labels, preds))+'\n')
        file.write(f'avg: {CUI_pos_avg},{CUI_neg_avg}\n')

    csv = {'slide_id':slide_ids, args.label_name:labels, 'preds':preds, 'probs':probs}
    csv = pd.DataFrame(csv)
    csv.to_csv(os.path.join(args.output_dir, 'prob_output.csv'),index=False)
    
    np.save(os.path.join(args.output_dir, 'probs.npy'), np.array(probs))
