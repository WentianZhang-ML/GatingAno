import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import utils 
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import f1_score, balanced_accuracy_score



def image_wise_anomaly_detection(score_dict, image_label_dict):
    score = []
    label = []
    for key in score_dict.keys():
        if key in image_label_dict.keys():
            score.append(float(score_dict[key]))
            label.append(int(image_label_dict[key]))
    auc = roc_auc_score(label, score)
    fpr, tpr, thres = roc_curve(label, score)    
    sum_ = [tpr[i]+1-fpr[i] for i in range(len(thres))]
    idx = np.argmax(np.array(sum_))
    TT = thres[idx]
    prob_f1 = [1 if s >= TT else 0 for s in score]
    acc_f1 = accuracy_score(label, prob_f1)
    f1 = f1_score(label, prob_f1)
    return auc, acc_f1, f1, TT
def pixel_wise_anomaly_detection(score, label):
    auc = roc_auc_score(label, score, average='micro')
    fpr, tpr, thres = roc_curve(label, score)    
    sum_ = [tpr[i]+1-fpr[i] for i in range(len(thres))]
    idx = np.argmax(np.array(sum_))
    TT = thres[idx]
    prob_f1 = [1 if s >= TT else 0 for s in score]
    acc = balanced_accuracy_score(label, prob_f1)
    f1 = f1_score(label, prob_f1, average='micro')
    return auc, acc, f1, TT

def eval_image(dataset, net, alpha):
    score_dict = {}
    img_label_dict = {}
    net.eval()
    data_batch = tqdm(dataset)
    data_batch.set_description("Evaluate")
    with torch.no_grad():
        criterion = nn.MSELoss().cuda()
        pdist = nn.PairwiseDistance(p=2).cuda()
        for iter_, (input_, img_id, img_hog, img_label) in enumerate(data_batch):

            input_ = input_.cuda()
            feature, recon_image = net(input_, alpha)
            feature_recon, rere_image = net(recon_image[:,0,:,:].unsqueeze(1), alpha)
            f_1 = torch.flatten(feature, start_dim=1)
            f_2 = torch.flatten(feature_recon, start_dim=1)
            scores = pdist(f_1, f_2)

            for i in range(len(scores)):
                img_label_dict[img_id[i]] = int(img_label[i])
                if (img_id[i] in score_dict.keys()):
                    score_dict[img_id[i]].append(float(scores[i]))
                else:
                    score_dict[img_id[i]] = []
                    score_dict[img_id[i]].append(float(scores[i]))

        score_dict_ = {}
        for key in score_dict.keys():
            score_dict[key].sort(reverse=True)
            score_dict_[key] = np.std(score_dict[key], ddof=1)
        auc, acc, f1, thres = image_wise_anomaly_detection(score_dict_, img_label_dict)

    return auc, acc, f1, thres
def eval_pixel(dataset, net, alpha):
    score = []
    label = []
    loss_ = []
    test_loss = utils.AverageMeter()
    net.eval()
    data_batch = tqdm(dataset)
    data_batch.set_description("Evaluate")
    with torch.no_grad():
        criterion = nn.MSELoss().cuda()
        pdist_l1 = nn.PairwiseDistance(p=1).cuda()
        for iter_, (input_, img_id, img_hog, img_roi_label) in enumerate(data_batch):
            input_ = input_.cuda()
            feature, recon_image = net(input_, alpha)
            scores = pdist_l1(input_,recon_image[:,0,:,:].unsqueeze(1))
            loss = criterion(input_,recon_image[:,0,:,:].unsqueeze(1))
            test_loss.update(loss.item())

            for i in range(len(scores)):
                score_ = list(scores[i].view(-1).cpu().data.numpy())
                label_ = list(img_roi_label[i].view(-1).cpu().data.numpy())
                score.extend(score_)
                label.extend(label_)
        auc, acc, f1, thres = pixel_wise_anomaly_detection(score, label)
        loss_ave = test_loss.avg
    return auc, acc, f1, thres
