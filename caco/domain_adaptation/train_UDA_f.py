import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from caco.model.deeplabv2_f import get_deeplab_v2_f
from advent.model.discriminator import get_fc_discriminator
from advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from advent.utils.func import loss_calc, bce_loss
from advent.utils.loss import entropy_loss
from advent.utils.func import prob_2_entropy
from advent.utils.viz_segmask import colorize_mask

from caco.utils.loss_caco import CaContrast_loss
CaContrast = CaContrast_loss()
import matplotlib.pyplot as plt


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'caco':
        train_caco(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")


def train_caco(model, trainloader, targetloader, cfg):
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    model_ema = get_deeplab_v2_f(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
    model_ema.load_state_dict(model.state_dict().copy())
    model_ema.train()
    model_ema.to(device)

    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    # PL_queue: [1, 19, M=100], queue: [1, 2048, 19, M=100], queue_prob: [1, 19, 19, M=100]
    queue = torch.zeros(1, 2048, cfg.NUM_CLASSES, cfg.TRAIN.dict_len)
    PL_queue = torch.ones(1, cfg.NUM_CLASSES, cfg.TRAIN.dict_len, dtype=torch.int64) * (-1)
    queue_prob = torch.zeros(1, cfg.NUM_CLASSES, cfg.NUM_CLASSES, cfg.TRAIN.dict_len)

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            param_k.data = param_k.data.clone() * cfg.TRAIN.move_momentum + param_q.data.clone() * (1. - cfg.TRAIN.move_momentum)
        for buffer_q, buffer_k in zip(model.buffers(), model_ema.buffers()):
            buffer_k.data = buffer_q.data.clone()

        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()

        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)

        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux_pooled, pred_src_main_pooled, _, f_src_main = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux_pooled)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main_pooled)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch

        with torch.no_grad():
            pred_trg_aux_pooled, pred_trg_main_pooled, _, f_trg_main = model_ema(images.cuda(device))
            queue, PL_queue, queue_prob, queue_train, PL_queue_train,  queue_prob_train = queue_update(queue, PL_queue, queue_prob, f_src_main, labels, f_trg_main, F.softmax(pred_trg_main_pooled), cfg)

        # CaCo flow.
        with torch.no_grad():
            pred_trg_aux_pooled_ref, pred_trg_main_pooled_ref = pred_trg_aux_pooled.detach().clone(), pred_trg_main_pooled.detach().clone()
        images_target_aug = data_aug(images.cuda(device), cfg)
        pred_trg_aux_aug, pred_trg_main_aug, _, f_trg_main_aug = model(images_target_aug.cuda(device))
        interp_aug_target = nn.Upsample(size = (pred_trg_main_pooled_ref.shape[-2], pred_trg_main_pooled_ref.shape[-1]), mode='bilinear', align_corners=True)
        pred_trg_main_aug_pooled = interp_aug_target(pred_trg_main_aug)
        out_trg_main_aug_pooled = F.softmax(pred_trg_main_aug_pooled)
        out_trg_main_pooled_ref = F.softmax(pred_trg_main_pooled_ref)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_aug_pooled = interp_aug_target(pred_trg_aux_aug)
            out_trg_aux_aug_pooled = F.softmax(pred_trg_aux_aug_pooled)
            out_trg_aux_pooled_ref = F.softmax(pred_trg_aux_pooled_ref)

        out_trg_d = dict()
        out_trg_d['out_trg_main_aug_pooled'] = out_trg_main_aug_pooled
        out_trg_d['out_trg_main_pooled_ref'] = out_trg_main_pooled_ref
        out_trg_d['out_trg_aux_aug_pooled'] = out_trg_aux_aug_pooled
        out_trg_d['out_trg_aux_pooled_ref'] = out_trg_aux_pooled_ref

        with torch.no_grad():
            interp_aug_target_pl = nn.Upsample(size = (pred_trg_main_pooled.shape[-2], pred_trg_main_pooled.shape[-1]), mode='bilinear', align_corners=True)
            PL_trg_pred = (1.0 * F.softmax(pred_trg_main_pooled) + 0.0 * (interp_aug_target_pl(F.softmax(pred_trg_main_aug))))
            PL_trg = torch.argmax(PL_trg_pred, dim=1)
            interp_aug_target_f = nn.Upsample(size = (f_trg_main.shape[-2], f_trg_main.shape[-1]), mode='bilinear', align_corners=True)
        f_trg_main_aug = interp_aug_target_f(f_trg_main_aug)

        loss_caco = loss_caco_cal(out_trg_d, PL_trg, f_trg_main_aug, PL_queue_train, queue_train, queue_prob_train, cfg)

        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_aug_ori = interp_target(pred_trg_aux_aug_pooled)
            d_out_aux_aug = d_aux(prob_2_entropy(F.softmax(pred_trg_aux_aug_ori)))
            loss_adv_trg_aux_aug = bce_loss(d_out_aux_aug, source_label)
        else:
            loss_adv_trg_aux_aug = 0
        pred_trg_main_aug_ori = interp_target(pred_trg_main_aug_pooled)
        d_out_main_aug = d_main(prob_2_entropy(F.softmax(pred_trg_main_aug_ori)))
        loss_adv_trg_main_aug = bce_loss(d_out_main_aug, source_label)

        loss = 0 * (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main_aug
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux_aug) + loss_caco
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux_aug_ori.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main_aug_ori.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {
                          'loss_seg_src_main': loss_seg_src_main,
                          'loss_caco': loss_caco}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break

        sys.stdout.flush()


def l1_loss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss)
    return loss


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def loss_caco_cal(out_trg_d, PL_trg, f_trg_main_aug, PL_queue_train, queue_train, queue_prob_train, cfg):
    loss_caco_o_main = l1_loss(out_trg_d['out_trg_main_aug_pooled'], out_trg_d['out_trg_main_pooled_ref'])
    if cfg.TRAIN.MULTI_LEVEL:
        loss_caco_o_aux = l1_loss(out_trg_d['out_trg_aux_aug_pooled'], out_trg_d['out_trg_aux_pooled_ref'])
    else:
        loss_caco_o_aux = 0
    loss_caco_o = loss_caco_o_main + loss_caco_o_aux
    loss_caco_f = CaContrast_cal(PL_trg.detach().clone(), f_trg_main_aug,
                                 PL_queue_train.cuda(cfg.GPU_ID).detach().clone(), queue_train.cuda(cfg.GPU_ID).detach().clone(),
                                 queue_prob_train.cuda(cfg.GPU_ID).detach().clone(), cfg)

    if [PL_queue_train != -1][0].sum() != cfg.NUM_CLASSES*cfg.TRAIN.dict_len:
        print('dict is loading.....  ', int([PL_queue_train != -1][0].sum()), '/1900')
        loss_caco_f = 0
    loss_caco = loss_caco_o + cfg.TRAIN.featurec * loss_caco_f
    return loss_caco


def CaContrast_cal(label_label_aug1, feature, labels, feature_ma, pred1_ma, cfg):
    with torch.no_grad():
        # [bsz, n_samples]
        label_label_aug1 = (F.interpolate(label_label_aug1.clone().type(torch.FloatTensor).unsqueeze(1), size=feature.shape[2:4], mode='nearest')).view(label_label_aug1.shape[0], -1)
        labels = (labels.clone().type(torch.FloatTensor)).view(labels.shape[0], -1)

        ent_ma = entropy_cal(pred1_ma)
        ent_ma = torch.sum((ent_ma.clone()).view(ent_ma.shape[0],ent_ma.shape[1], -1), dim=1) / np.log2(cfg.NUM_CLASSES)
        reli = torch.clamp((1 - ent_ma + cfg.TRAIN.ent_clamp), min=1.0-cfg.TRAIN.ent_clamp, max=1.0+cfg.TRAIN.ent_clamp)
        reli = reli.view(label_label_aug1.shape[0], -1)

    # [bsz, n_samples, n_views, ...]
    feature = F.normalize(feature.view(feature.shape[0], feature.shape[1], -1), dim=1)
    feature = feature.transpose(1,2).unsqueeze(2)
    with torch.no_grad():
        feature_ma = F.normalize(feature_ma.view(feature_ma.shape[0], feature_ma.shape[1], -1), dim=1)
        feature_ma = feature_ma.transpose(1,2).unsqueeze(2)

    loss = CaContrast(features=feature, labels=label_label_aug1, features_2=feature_ma, labels_2=labels, reliability=reli, cfg=cfg)
    return loss


def entropy_cal(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    return - torch.mul(v, torch.log2(v + 1e-30))


def queue_update(queue, PL_queue, queue_prob, f_src_main, labels, f_trg_main, pred_trg_main_pooled, cfg):
    labels = (F.interpolate(labels.clone().type(torch.FloatTensor).unsqueeze(1), size=f_src_main.shape[2:4], mode='nearest'))
    labels_trg = torch.argmax(pred_trg_main_pooled, dim=1).unsqueeze(1)
    labels_trg[torch.max(pred_trg_main_pooled, dim=1)[0].unsqueeze(1) < 0.9] = 255

    for i in range(cfg.NUM_CLASSES):
        if [labels == i][0].sum() > 0:
            queue[0, :, i, 0] = torch.mean(f_src_main.view(f_src_main.shape[0], f_src_main.shape[1], -1)[:, :, labels.squeeze(0).squeeze(0).view(-1) == i], dim=2)
            queue_prob[0, :, i, 0] = 0.0
            queue_prob[0, i, i, 0] = 1.0
            PL_queue[0, i, 0] = i

            queue[0, :, i, :] = torch.roll(queue[0, :, i, :], -1, 1)
            queue_prob[0, :, i, :] = torch.roll(queue_prob[0, :, i, :], -1, 1)
            PL_queue[0, i, :] = torch.roll(PL_queue[0, i, :], -1, 0)

        if [labels_trg == i][0].sum() > 0:
            queue[0, :, i, 0] = torch.mean(f_trg_main.view(f_trg_main.shape[0], f_trg_main.shape[1], -1)[:, :, labels_trg.squeeze(0).squeeze(0).view(-1) == i], dim=2)
            queue_prob[0, :, i, 0] = torch.mean(pred_trg_main_pooled.view(pred_trg_main_pooled.shape[0], pred_trg_main_pooled.shape[1], -1)[:, :, labels_trg.squeeze(0).squeeze(0).view(-1) == i], dim=2)
            PL_queue[0, i, 0] = i

            queue[0, :, i, :] = torch.roll(queue[0, :, i, :], -1, 1)
            queue_prob[0, :, i, :] = torch.roll(queue_prob[0, :, i, :], -1, 1)
            PL_queue[0, i, :] = torch.roll(PL_queue[0, i, :], -1, 0)

    queue_train = queue.view(queue.shape[0], queue.shape[1], -1)[:, :, PL_queue.squeeze(0).view(-1) != -1].unsqueeze(3)
    queue_prob_train = queue_prob.view(queue_prob.shape[0], queue_prob.shape[1], -1)[:, :, PL_queue.squeeze(0).view(-1) != -1].unsqueeze(3)
    PL_queue_train = PL_queue[PL_queue != -1].unsqueeze(0).unsqueeze(2)
    return queue, PL_queue, queue_prob, queue_train, PL_queue_train,  queue_prob_train


def data_aug(input_images, cfg):
    scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
    scaled_size_target = (round(cfg.TRAIN.INPUT_SIZE_TARGET[1] * scale_ratio / 8) * 8, round(cfg.TRAIN.INPUT_SIZE_TARGET[0] * scale_ratio / 8) * 8)
    interp_target_sc = nn.Upsample(size=scaled_size_target, mode='bilinear', align_corners=True)
    output_images = interp_target_sc(input_images)
    return output_images
