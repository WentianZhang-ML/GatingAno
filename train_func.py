import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from evaluate import eval_image, eval_pixel
import utils
from models import AdversarialLoss


def save_checkpoint(state, is_best, epoch, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(save_path, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'),
                        os.path.join(save_path, 'model_best_.pth.tar'))

def weight_decay(alpha, deta, len_batch):

    alpha1 = alpha[0] - 0.05 / float(len_batch)
    alpha2 = alpha[1] - 0.04 / float(len_batch)
    alpha3 = alpha[2] - 0.03 / float(len_batch)
    alpha4 = alpha[3] - deta / float(len_batch)
    if alpha1 < 0:
        alpha1 = 0
    if alpha2 < 0:
        alpha2 = 0
    if alpha3 < 0:
        alpha3 = 0
    if alpha4 < 0:
        alpha4 = 0
    return [alpha1,alpha2,alpha3,alpha4]

def train_step(net, net_d, dataset_name, args, config, logger, gating = False):

    start_epoch = -1
    criterion = nn.MSELoss().cuda()
    adversarial_loss= AdversarialLoss().cuda()

    net = torch.nn.DataParallel(net).cuda()
    net_d = torch.nn.DataParallel(net_d).cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=config.weight_decay_base,amsgrad=True)
    optimizer_D = torch.optim.Adam(net_d.parameters(), lr=args.lr, weight_decay=config.weight_decay_base)

    if 'pixel' in args.level:
        train_data, test_data = utils.get_dataset_pixel(config.dataset[dataset_name], config.batch_size_base)
    else:
        train_data, test_data = utils.get_dataset_image(config.dataset[dataset_name], config.batch_size_base)

    len_batch = train_data.__len__()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs_base, last_epoch=-1)

    [best_auc, best_precision, best_recall, best_acc, best_f1, best_thres] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    logger.info('**************************** start training target model! ******************************\n')
    logger.info(
        '---------|---------------------- VALID ---------------------|-- Training --|---------- Current Best ----------|\n')
    logger.info(
        '  epoch  |   AUC   PRECISION   RECALL   ACC   F-1   Thres   |     loss     |    AUC     ACC     F-1    Thres  |\n')
    logger.info(
        '--------------------------------------------------------------------------------------------------------------|\n')
    alpha_ = [1,1,1,1]
    for epoch in range(start_epoch+1, config.n_epochs_base):
        training_loss1 = utils.AverageMeter()
        training_loss2 = utils.AverageMeter()
        data_batch = tqdm(train_data)
        for iter_, (input_, img_id, img_hog, img_label) in enumerate(data_batch):
            if gating:
                alpha_cur = weight_decay(alpha_, args.deta, float(len_batch))
            else:
                alpha_cur = [1,1,1,1]

            input_ = input_.cuda()
            img_hog = img_hog.cuda()
            net.train()
            net_d.train()

            feature, recon_image = net(input_, alpha_cur)
            recon_img = recon_image[:, 0, :, :].unsqueeze(1)
            recon_hog = recon_image[:, 1, :, :].unsqueeze(1)
            loss1 = criterion(input_, recon_img)
            loss2 = criterion(img_hog, recon_hog)
            
            L_recon = loss1+loss2

            # adversarial loss
            #-------------------image----------------------
            # discriminator adversarial loss
            real_B = input_
            fake_B = recon_img
            dis_input_real = real_B
            dis_input_fake = fake_B.detach()
            dis_real, dis_real_feat = net_d(dis_input_real)
            dis_fake, dis_fake_feat = net_d(dis_input_fake)
            dis_real_loss = adversarial_loss(dis_real, True, True)
            dis_fake_loss = adversarial_loss(dis_fake, False, True)
            dis_loss_recon = (dis_real_loss + dis_fake_loss) / 2
            # # generator adversarial loss
            gen_input_fake = fake_B
            gen_fake, gen_fake_feat = net_d(gen_input_fake)
            gen_loss_recon = adversarial_loss(gen_fake, True, False)
            #-------------------hog----------------------
            # discriminator adversarial loss
            real_B_ = img_hog
            fake_B_ = recon_hog
            dis_input_real_ = real_B_
            dis_input_fake_ = fake_B_.detach()
            dis_real_, dis_real_feat_ = net_d(dis_input_real_)
            dis_fake_, dis_fake_feat_ = net_d(dis_input_fake_)
            dis_real_loss_ = adversarial_loss(dis_real_, True, True)
            dis_fake_loss_ = adversarial_loss(dis_fake_, False, True)
            dis_loss_recon_hog = (dis_real_loss_ + dis_fake_loss_) / 2
            # # generator adversarial loss
            gen_input_fake_ = fake_B_
            gen_fake_, gen_fake_feat_ = net_d(gen_input_fake_)
            gen_loss_recon_hog = adversarial_loss(gen_fake_, True, False)

            dis_loss = 0.05*dis_loss_recon+ 0.05*dis_loss_recon_hog
            gen_loss = L_recon + 0.05*gen_loss_recon + 0.05*gen_loss_recon_hog

            optimizer.zero_grad()
            gen_loss.backward()
            optimizer.step()
            optimizer_D.zero_grad()
            dis_loss.backward()
            optimizer_D.step()

            training_loss1.update(gen_loss.item())
            training_loss2.update(dis_loss.item())
            alpha_ = alpha_cur
            data_batch.set_postfix(gen_loss=training_loss1.avg, dis_loss=training_loss2.avg)
        scheduler.step()
        if (epoch + 1) % args.eval_epoch == 0:

            if 'pixel' in args.level:
                [auc, acc, f1, thre]=eval_pixel(test_data, net, alpha=alpha_cur)
            else:
                [auc, acc, f1, thre]=eval_image(test_data, net, alpha=alpha_cur)
            thres = thre
            precision = 1
            recall = 1

            is_best = auc >= best_auc
            if is_best:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_thres = thres

                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.model,
                    'state_dict': net.state_dict(),
                    'best_auc': best_auc,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, is_best, epoch, save_path = args.save_dir )
            logger.info(
                '  %3d  |  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  |  %5.6f  |  %5.6f  |  %5.3f  %5.3f  %5.3f  %5.3f  |'
                % (
                    epoch + 1,
                    auc * 100, precision * 100, recall * 100, acc * 100, f1 * 100, thres,
                    training_loss1.avg, training_loss2.avg,
                    float(best_auc * 100), float(best_acc * 100), float(best_f1 * 100), best_thres))