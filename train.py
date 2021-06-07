import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path, saving_path
from datasets import ImageFolder, VideoImageFolder, VideoSequenceFolder, VideoImage2Folder, ImageFlowFolder, ImageFlow2Folder, ImageFlow3Folder
from misc import AvgMeter, check_mkdir, CriterionKL3, CriterionKL, CriterionPairWise, CriterionStructure

from models.net import INet
from MGA.mga_model import MGA_Network
from torch.backends import cudnn
import time
from utils.utils_mine import load_part_of_model, load_part_of_model2, load_MGA
# from module.morphology import Erosion2d
from itertools import cycle
import random
import numpy as np

cudnn.benchmark = True

device_id = 0
device_id2 = 0

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

random.seed(2021)
np.random.seed(2021)
# torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = saving_path
exp_name = 'VideoSaliency' + '_' + time_str

args = {
    'distillation': True,
    'L2': False,
    'KL': True,
    'structure': True,
    'iter_num': 200000,
    'iter_save': 4000,
    'iter_start_seq': 0,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 10 * 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.925,
    'snapshot': '',
    # 'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2021-04-06 11:56:00', '92000.pth'),
    'pretrain': '',
    'mga_model_path': 'pre-trained/MGA_trained.pth',
    # 'imgs_file': 'Pre-train/pretrain_all_seq_DUT_DAFB2_DAVSOD.txt',
    'imgs_file': 'Pre-train/pretrain_all_seq_DAFB2_DAVSOD_flow.txt',
    'imgs_file2': 'Pre-train/pretrain_all_seq_DUT_TR_DAFB2.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_DAVSOD_5f.txt',
    # 'train_loader': 'video_image'
    'train_loader': 'flow_image3',
    # 'train_loader': 'video_sequence'
    'image_size': 256,
    'crop_size': 224,
    'self_distill': 0.1,
    'teacher_distill': 0.6
}

imgs_file = os.path.join(datasets_root, args['imgs_file'])
# imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

joint_transform = joint_transforms.Compose([
    joint_transforms.ImageResize(args['image_size']),
    joint_transforms.RandomCrop(args['crop_size']),
    # joint_transforms.ColorJitter(hue=[-0.1, 0.1], saturation=0.05),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])

# joint_transform = joint_transforms.Compose([
#     joint_transforms.ImageResize(290),
#     joint_transforms.RandomCrop(256),
#     joint_transforms.RandomHorizontallyFlip(),
#     joint_transforms.RandomRotate(10)
# ])

# joint_seq_transform = joint_transforms.Compose([
#     joint_transforms.ImageResize(520),
#     joint_transforms.RandomCrop(473)
# ])

input_size = (473, 473)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
if args['train_loader'] == 'video_sequence':
    train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'video_image':
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'flow_image':
    train_set = ImageFlowFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'flow_image2':
    train_set = ImageFlow2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2', video_seq_gt_path + '/DAFB2',
                                 joint_transform, (args['crop_size'], args['crop_size']), img_transform, target_transform)
elif args['train_loader'] == 'flow_image3':
    imgs_file2 = os.path.join(datasets_root, args['imgs_file2'])
    train_set = ImageFlow3Folder(video_train_path, imgs_file, imgs_file2, joint_transform, img_transform, target_transform)
    # train_set = ImageFlowFolder(video_train_path, imgs_file,
    #                             joint_transform, img_transform, target_transform)
    # train_set2 = VideoImageFolder(video_train_path, imgs_file2,
    #                               joint_transform, img_transform, target_transform)
else:
    train_set = VideoImage2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2', video_seq_gt_path + '/DAFB2',
                                  joint_transform, None, input_size, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
# if train_set2 is not None:
#     train_loader2 = DataLoader(train_set2, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
# erosion = Erosion2d(1, 1, 5, soft_max=False).cuda()
if args['L2']:
    criterion_l2 = nn.MSELoss().cuda()
    # criterion_pair = CriterionPairWise(scale=0.5).cuda()
if args['KL']:
    criterion_kl = CriterionKL3().cuda()

if args['structure']:
    criterion_str = CriterionStructure().cuda()

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def main():
    teacher = None
    if args['distillation']:
        teacher = MGA_Network(nInputChannels=3, n_classes=1, os=16,
                              img_backbone_type='resnet101', flow_backbone_type='resnet34')
        teacher = load_MGA(teacher, args['mga_model_path'], device_id=device_id2)
        teacher.eval()
        teacher.cuda(device_id2)

    net = INet(cfg=None).cuda(device_id).train()
    bkbone, remains = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            # param.requires_grad = False
            bkbone.append(param)
        # elif 'flow' in name or 'linearf' in name or 'decoder' in name:
        #     print('flow related:', name)
        #     flow_modules.append(param)
        # elif 'flow' in name or 'linearf' in name or 'decoder' in name:
        #     print('decoder related:', name)
        #     flow_modules.append(param)
        else:
            print('remains:', name)
            remains.append(param)
    # fix_parameters(net.named_parameters())
    # optimizer = optim.SGD([
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
    #      'lr': 2 * args['lr']},
    #     {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
    #      'lr': args['lr'], 'weight_decay': args['weight_decay']}
    # ], momentum=args['momentum'])

    optimizer = optim.SGD([{'params': bkbone}, {'params': remains}],
                          lr=args['lr'], momentum=args['momentum'],
                          weight_decay=args['weight_decay'], nesterov=True)

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 0.5 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
        # optimizer.param_groups[2]['lr'] = args['lr']

    net = load_part_of_model(net, 'pre-trained/SNet.pth', device_id=device_id)
    if len(args['pretrain']) > 0:
        print('pretrain model from ' + args['pretrain'])
        net = load_part_of_model(net, args['pretrain'], device_id=device_id)
        # fix_parameters(student.named_parameters())

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer, teacher)


def train(net, optimizer, teacher=None):
    curr_iter = args['last_iter']
    while True:

        # loss3_record = AvgMeter()
        # dataloader_iterator = iter(train_loader2)
        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = 0.5 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                  ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            # optimizer.param_groups[2]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                       ) ** args['lr_decay']
            #
            # optimizer.param_groups[3]['lr'] = 0.1 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
            #                                                 ) ** args['lr_decay']
            #
            # inputs, flows, labels, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab = data
            inputs, flows, labels, inputs2, labels2 = data
            # data2 = next(dataloader_iterator)
            # inputs2, labels2 = data2
            if curr_iter % 2 == 0:
                train_single(net, inputs, flows, labels, optimizer, curr_iter, teacher)
            else:
                train_single2(net, inputs2, labels2, optimizer, curr_iter)
            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return

def train_single(net, inputs, flows, labels, optimizer, curr_iter, teacher):
    inputs = Variable(inputs).cuda(device_id)
    flows = Variable(flows).cuda(device_id)
    labels = Variable(labels).cuda(device_id)

    # prediction = torch.nn.Sigmoid()(prediction)
    # prediction = prediction.data.cpu().numpy()
    # prediction = F.upsample(prediction, size=(), mode='bilinear', align_corners=True)
    # from matplotlib import pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(prediction[0, 0, :, :])
    # plt.show()
    optimizer.zero_grad()

    out1u, out2u = net(inputs, flows)

    loss0 = criterion_str(out1u, labels)
    loss1 = criterion_str(out2u, labels)
    # loss2 = criterion_str(out3u, labels)
    # loss3 = criterion_str(out3r, labels)
    # loss4 = criterion_str(out4r, labels)
    # loss5 = criterion_str(out5r, labels)

    # loss2_k = criterion_kl(F.adaptive_avg_pool2d(out2r_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))
    # loss3_k = criterion_kl(F.adaptive_avg_pool2d(out3r_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))
    # loss4_k = criterion_kl(F.adaptive_avg_pool2d(out4r_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))
    # loss5_k = criterion_kl(F.adaptive_avg_pool2d(out5r_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))

    # loss6 = criterion_str(out2f, labels)
    # loss7 = criterion_str(out3f, labels)
    # loss8 = criterion_str(out4f, labels)

    # loss6_k = criterion_kl(F.adaptive_avg_pool2d(out2f_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))
    # loss7_k = criterion_kl(F.adaptive_avg_pool2d(out3f_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))
    # loss8_k = criterion_kl(F.adaptive_avg_pool2d(out4f_k, (1, 1)), F.adaptive_avg_pool2d(pred3_k, (1, 1)))

    # loss6_k = criterion_kl(F.adaptive_avg_pool2d(out2f_k, (1, 1)), F.adaptive_avg_pool2d(out4f_k, (1, 1)))
    # loss7_k = criterion_kl(F.adaptive_avg_pool2d(out3f_k, (1, 1)), F.adaptive_avg_pool2d(out4f_k, (1, 1)))
    # print(loss6_k, '---', loss7_k)
    # loss9 = criterion_str(out3f_flow, labels)

    if args['distillation']:
        inputs = inputs.cuda(device_id2)
        flows = flows.cuda(device_id2)
        prediction, _, _, _, _ = teacher(inputs, flows)
        prediction = prediction.cuda(device_id)
        loss0_t = criterion_str(out1u, F.sigmoid(prediction))
        loss1_t = criterion_str(out2u, F.sigmoid(prediction))
        # loss2_t = criterion_str(out3u, F.sigmoid(prediction))
        # loss3_t = criterion_str(out3r, F.sigmoid(prediction))
        # loss4_t = criterion_str(out4r, F.sigmoid(prediction))
        # loss5_t = criterion_str(out5r, F.sigmoid(prediction))

        # loss6_t = criterion_str(out2f, F.sigmoid(prediction))
        # loss7_t = criterion_str(out3f, F.sigmoid(prediction))
        # loss8_t = criterion_str(out4f, F.sigmoid(prediction))
        # loss9_t = criterion_str(out3f_flow, F.sigmoid(prediction))

        distill_loss_t = (loss0_t + loss1_t) / 2

    # loss2_d = criterion_str(out2r, F.sigmoid(out2u))
    # loss3_d = criterion_str(out3r, F.sigmoid(out2u))
    # loss4_d = criterion_str(out4r, F.sigmoid(out2u))
    # loss5_d = criterion_str(out5r, F.sigmoid(out2u))
    #
    # loss7_d = criterion_str(out2f, F.sigmoid(out2u))
    # loss8_d = criterion_str(out3f, F.sigmoid(out2u))
    # loss9_d = criterion_str(out4f, F.sigmoid(out2u))

    # loss10 = criterion_str(out1a, labels)
    # loss11 = criterion_str(out2a, labels)

    total_loss = (loss0 + loss1) / 2
    # distill_loss = loss6_k + loss7_k
    if args['distillation']:
        total_loss = total_loss + args['teacher_distill'] * distill_loss_t
        # total_loss = total_loss + 0.1 * distill_loss + 0.5 * distill_loss_t
    else:
        # total_loss = total_loss + 0.1 * distill_loss
        total_loss = total_loss
    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss1, args['train_batch_size'], curr_iter, optimizer)

    return

def train_single2(net, inputs, labels, optimizer, curr_iter):
    inputs = Variable(inputs).cuda(device_id)
    labels = Variable(labels).cuda(device_id)

    optimizer.zero_grad()

    out1u, out2u, = net(inputs)

    loss0 = criterion_str(out1u, labels)
    loss1 = criterion_str(out2u, labels)
    # loss2 = criterion_str(out3u, labels)
    # loss3 = criterion_str(out3r, labels)
    # loss4 = criterion_str(out4r, labels)
    # loss5 = criterion_str(out5r, labels)

    # loss6 = criterion_str(out3f_flow, labels)

    total_loss = (loss0 + loss1) / 2
    # distill_loss = loss6_k + loss7_k + loss8_k

    # total_loss = total_loss + 0.1 * distill_loss
    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss1, args['train_batch_size'], curr_iter, optimizer)

    return

def print_log(total_loss, loss0, loss1, loss2, batch_size, curr_iter, optimizer, type='normal'):
    total_loss_record.update(total_loss.data, batch_size)
    loss0_record.update(loss0.data, batch_size)
    loss1_record.update(loss1.data, batch_size)
    loss2_record.update(loss2.data, batch_size)
    # loss3_record.update(loss3.data, batch_size)
    # loss4_record.update(loss4.data, batch_size)
    log = '[iter %d][%s], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f] ' \
          '[lr %.13f]' % \
          (curr_iter, type, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
           optimizer.param_groups[1]['lr'])
    print(log)
    open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()
