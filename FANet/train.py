#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
用于建筑物识别，使用f1_score来挑选最好的模型，修改于2018.6.24
'''
import datetime
import os
import random
import numpy as np
import torchvision.transforms as standard_transforms
#from scipy.misc import imread, imsave
from imageio import imread, imsave
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
#from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils.simul_transforms as simul_transforms
import utils.transforms as extended_transforms
from datasets import potsdam
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
import torch
from torch import nn
'''
check_mkdir：检测是否有指定的文件夹，如果没有则创建
evaluate：用于计算模型的精度等
AverageMeter：用于计算一次epoch全部迭代的平均loss
CrossEntropyLoss2d：用于计算全卷积的loss
'''
#torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True#cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
#                       -- If this is set to false, uses some in-built heuristics that might not always be fastest.

outputs_path = '../results'
exp_name = 'FANet'

args = {
    'train_batch_size': 5,
    'epoch_num': 100,
    'lr': 1e-2,
    'lr_decay': 0.9,
    'max_iter': 100e4,
    'weight_decay': 5e-4,
    'input_size': (320, 320),
    'momentum': 0.9,
    'lr_patience': 10,  # large patience denotes fixed lr
    'snapshot': '',  # empty string denotes no snapshot
    'print_freq': 20, #how long print once
    'val_batch_size': 5,
    'val_save_to_img_file': True, #whether save image
    'val_img_sample_rate': 0.01  # randomly sample some validation results to display
}


def main(train_args):
    net = FANet(num_classes=potsdam.num_classes).cuda()
    
    # 是否有已经训练的结果，如果有可以继续训练，只需要把保存的pth文件名输入train_args['snapshot']
    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0, 'f1_score': 0}
    else:
        print('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(outputs_path, exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11]), 'f1_score': float(split_snapshot[14])}
    #net.final = nn.Conv2d(64, potsdam.num_classes, kernel_size=1, bias=False).cuda()
    net.train()

    mean_std = ([0.419, 0.435, 0.449], [0.179, 0.168, 0.166])
    #short_size = int(min(train_args['input_size']) / 0.875)
    train_simul_transform = simul_transforms.Compose([
        #simul_transforms.Scale(short_size),
        #simul_transforms.RandomCrop(train_args['input_size']),
        #simul_transforms.RandomHorizontallyFlip()
    ])
    val_simul_transform = simul_transforms.Compose([
        #simul_transforms.Scale(short_size),
        #simul_transforms.CenterCrop(train_args['input_size'])
    ])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    restore_transform = standard_transforms.Compose([
        extended_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    #加载测试数据集
    val_set = potsdam.Potsdam('val', simul_transform=None, transform=input_transform,
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=train_args['val_batch_size'], num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=potsdam.ignore_label).cuda()#用于计算fcn的loss

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], momentum=train_args['momentum'])

    #如果要加载之前训练的optimizer
    if len(train_args['snapshot']) < 0:
        optimizer.load_state_dict(torch.load(os.path.join(outputs_path, exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']

    #检测文件输出路径
    check_mkdir(outputs_path)
    check_mkdir(os.path.join(outputs_path, exp_name))

    with open(os.path.join(outputs_path, exp_name, 'record' + '.txt'), 'w') as f:
        f.write(str(train_args) + '\n\n')

    #scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=train_args['lr_patience'], min_lr=1e-10, verbose=True)

    #迭代进行训练和测试
    pre = ''
    for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
        #加载训练数据集
        train_set = potsdam.Potsdam('train', simul_transform=None,
                                          transform=input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], num_workers=8, shuffle=True)

        begin = datetime.datetime.now()
        train(train_loader, net, criterion, optimizer, epoch, train_args)
        end = datetime.datetime.now()
        with open(os.path.join(outputs_path, exp_name, 'record' + '.txt'), 'a') as f:
            f.write('--------------------------------------------------------------------' + '\n' +'this epoch take time:' + str(end-begin) + '\n')

        val_loss, pre = validate(val_loader, net, criterion, optimizer, epoch, train_args, restore_transform, pre)
        #scheduler.step(val_loss)


#训练函数
def train(train_loader, net, criterion, optimizer, epoch, train_args):
    train_loss = AverageMeter()#保存每次epoch的loss均值，进行新一次的epoch归零
    
    curr_iter = (epoch - 1) * len(train_loader) * int(train_args['train_batch_size'])
    print('[-----------------every train_loader include %d images----------------]' % (len(train_loader)*int(train_args['train_batch_size'])))
    for i, data in enumerate(train_loader):
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
        optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']
        test = (1 - float(curr_iter) / train_args['max_iter']) ** train_args['lr_decay']
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]#检验input_image与label尺寸是否相等
        #N = inputs.size(0)#得到inputs的batchsize
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(inputs)
        assert outputs.size()[2:] == labels.size()[1:]#检验output_image与label尺寸是否相等
        assert outputs.size()[1] == potsdam.num_classes

        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), N)
        
        if (i + 1) % train_args['print_freq'] == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [lr %.10f], [curr_iter %d]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, optimizer.param_groups[1]['lr'], curr_iter
            ))


#检验函数
def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, pre):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, labels_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, labels = data
        #N = inputs.size(0)
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda() #使用volatile=True参数为了不进行梯度计算

        outputs = net(inputs)
        val_loss.update(criterion(outputs, labels).item(), N)

        predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()#预测测试数据

        for i in inputs:
            if random.random() > train_args['val_img_sample_rate']:
                inputs_all.append(None)
            else:
                inputs_all.append(i.data.cpu())
        labels_all.append(labels.data.cpu().numpy())
        predictions_all.append(predictions)

    labels_all = np.concatenate(labels_all)
    predictions_all = np.concatenate(predictions_all)

    acc, acc_cls_all, acc_cls, mean_iu, fwavacc, f1_score = evaluate(predictions_all, labels_all, potsdam.num_classes)

    torch.save(net.state_dict(), os.path.join(outputs_path, exp_name, str(epoch) + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(outputs_path, exp_name, 'opt_' + str(epoch) + '.pth'))
    if epoch > 5:
        os.remove(os.path.join(outputs_path, exp_name, str(epoch-5) + '.pth'))
        os.remove(os.path.join(outputs_path, exp_name, 'opt_' + str(epoch-5) + '.pth'))

    if f1_score > train_args['best_record']['f1_score']:
        #删除之前最好的model
        if epoch > 1 and pre != '':
            os.remove(os.path.join(outputs_path, exp_name, pre + '.pth'))
            os.remove(os.path.join(outputs_path, exp_name, 'opt_' + pre + '.pth'))

        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        train_args['best_record']['f1_score'] = f1_score
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_f1_score_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1_score, optimizer.param_groups[1]['lr']
        )
        pre = snapshot_name
        #open(os.path.join(outputs_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')
        torch.save(net.state_dict(), os.path.join(outputs_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(outputs_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(outputs_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        for idx, data in enumerate(zip(inputs_all, labels_all, predictions_all)):
            if data[0] is None:
                continue
            inputs_pil = restore(data[0])
            labels_pil = data[1]
            predictions_pil = data[2]
            if train_args['val_save_to_img_file']:
                inputs_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                imsave(os.path.join(to_save_dir, '%d_prediction.png' % idx), predictions_pil)
                imsave(os.path.join(to_save_dir, '%d_label.png' % idx), labels_pil)

    print('--------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1_score %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1_score))

    print('every class acc: [clutter %.5f], [building %.5f]' % (
        acc_cls_all[0], acc_cls_all[1]))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1_score %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['f1_score'], train_args['best_record']['epoch']))

    print('--------------------------------------------------------------------')
    with open(os.path.join(outputs_path, exp_name, 'record' + '.txt'), 'a') as f:
        f.write('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1_score %.5f]' % (epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1_score)  + '\n' + 'every class acc: [clutter %.5f], [building %.5f]' % (acc_cls_all[0], acc_cls_all[1]) + '\n' + 'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1_score %.5f], [epoch %d]' % (train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['f1_score'], train_args['best_record']['epoch'])+ '\n\n')
    net.train()
    return val_loss.avg, pre


if __name__ == '__main__':
    main(args)
