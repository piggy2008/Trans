#!/usr/bin/python3
#coding=utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from MGA.ResNet import ResNet34
from module.Transformer import MapFuse, Attention2

import time
# from utils.utils_mine import visualize

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('pre-trained/resnet50-19c8e357.pth'), strict=False)

class SFM(nn.Module):
    def __init__(self):
        super(SFM, self).__init__()
        self.conv1h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4h = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4l = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv1f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv2f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv3f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.conv4f = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        # self.se_triplet = SETriplet(64, 64, 64, 64)
    def forward(self, low, high, flow):
        if high.size()[2:] != low.size()[2:]:
            high = F.interpolate(high, size=low.size()[2:], mode='bilinear')
        if flow.size()[2:] != low.size()[2:]:
            flow = F.interpolate(flow, size=low.size()[2:], mode='bilinear')
        out1h = self.conv1h(high)
        out2h = self.conv2h(out1h)
        out1l = self.conv1l(low)
        out2l = self.conv2l(out1l)
        out1f = self.conv1f(flow)
        out2f = self.conv2f(out1f)
        fuse  = out2h * out2l * out2f
        # fuse = self.se_triplet(out2h, out2l, out2f)
        out3h = self.conv3h(fuse) + out1h
        out4h = self.conv4h(out3h)
        out3l = self.conv3l(fuse) + out1l
        out4l = self.conv4l(out3l)
        out3f = self.conv3f(fuse) + out1f
        out4f = self.conv4f(out3f)

        return out4l, out4h, out4f

    def initialize(self):
        weight_init(self)

class Decoder_flow(nn.Module):
    def __init__(self):
        super(Decoder_flow, self).__init__()
        self.cfm45  = SFM()
        self.cfm34  = SFM()
        self.cfm23  = SFM()

    def forward(self, out2h, out3h, out4h, out5v, out2f, out3f, out4f, fback=None):
        if fback is not None:
            refine5      = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4      = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3      = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2      = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v        = out5v+refine5
            # print('out4h:', out4h.shape)
            # print('refine4:', refine4.shape)
            # print('out4f:', out4f.shape)
            out4f = F.interpolate(out4f, size=refine4.size()[2:], mode='bilinear')
            out4h, out4v, out4b = self.cfm45(out4h + refine4, out5v, out4f + refine4)
            out4b = F.interpolate(out4b, size=refine3.size()[2:], mode='bilinear')
            out3f = F.interpolate(out3f, size=refine3.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h + refine3, out4v, out3f + out4b + refine3)
            out3b = F.interpolate(out3b, size=refine2.size()[2:], mode='bilinear')
            out2f = F.interpolate(out2f, size=refine2.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h+refine2, out3v, out2f + out3b + refine2)
        else:
            out4h, out4v, out4b = self.cfm45(out4h, out5v, out4f)
            out4b = F.interpolate(out4b, size=out3f.size()[2:], mode='bilinear')
            out3h, out3v, out3b = self.cfm34(out3h, out4v, out3f + out4b)
            out3b = F.interpolate(out3b, size=out2f.size()[2:], mode='bilinear')
            out2h, pred, out2b = self.cfm23(out2h, out3v, out2f + out3b)
        return out2h, out3h, out4h, out5v, out2b, out3b, out4b, pred

    def initialize(self):
        weight_init(self)


class INet(nn.Module):
    def __init__(self, cfg):
        super(INet, self).__init__()
        self.cfg      = cfg
        self.bkbone   = ResNet()
        self.flow_bkbone = ResNet34(nInputChannels=3, os=16, pretrained=False)
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.flow_align4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.flow_align1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.feedback1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.feedback2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.feedback3 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # self.mf1 = Attention2(64, 14, 4)
        self.mf2 = Attention2(64, 28, 4)
        self.mf3 = Attention2(64, 56, 4)
        self.decoder1 = Decoder_flow()
        # self.decoder2 = Decoder_flow()
        # self.decoder3 = Decoder_flow()
        # self.se_many2 = SEMany2Many4(6, 64)
        # self.gnn_embedding = GNN_Embedding()
        self.linearpa = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearpb = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearpc = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearp3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        #
        # self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        # self.linearf1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearf2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearf3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.linearf4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        # self.EP = EP()

        self.initialize()

    def forward(self, x, flow=None, shape=None):
        out2h, out3h, out4h, out5v = self.bkbone(x) # layer1, layer2, layer3, layer4
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(out5v)
        if flow is not None:
            flow_layer4, flow_layer1, _, flow_layer2, flow_layer3 = self.flow_bkbone(flow)
            out1f, out2f = self.flow_align1(flow_layer1), self.flow_align2(flow_layer2)
            out3f, out4f = self.flow_align3(flow_layer3), self.flow_align4(flow_layer4)
            out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1 = self.decoder1(out2h, out3h, out4h, out5v, out2f, out3f, out4f)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.mf1(out2h, out3h, out4h, out5v, out2f, out3f, out4f)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = torch.split(feedback1, 64, 1)
            # pred1 = self.feedback1(out2h + out3h + out4h + out5v + out2f + out3f + out4f)

            out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.mf2(out2h, out3h, out4h, out5v, out2f, out3f, out4f)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = torch.split(feedback2, 64, 1)
            pred2 = self.feedback2(out4h + out5v)

            out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.mf3(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = torch.split(feedback2, 64, 1)
            pred3 = self.feedback3(out4h + out5v)

            shape = x.size()[2:] if shape is None else shape

            pred1a = F.interpolate(self.linearpa(pred1), size=shape, mode='bilinear')
            pred2a = F.interpolate(self.linearpb(pred2), size=shape, mode='bilinear')
            pred3a = F.interpolate(self.linearpc(pred3), size=shape, mode='bilinear')


            return pred1a, pred2a, pred3a
        else:
            out5f = F.interpolate(out5v, size=out4h.shape[2:], mode='bilinear')
            out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1 = self.decoder1(out2h, out3h, out4h, out5v, out3h, out4h, out5f)

            out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.mf2(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred1)

            pred2 = self.feedback2(out4h + out5v)

            out2h, out3h, out4h, out5v, out2f, out3f, out4f = self.mf3(out2h, out3h, out4h, out5v, out2f, out3f, out4f, pred2)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = torch.split(feedback2, 64, 1)
            pred3 = self.feedback3(out4h + out5v)

            # feedback3 = self.mf3(out2h + pred2, out3h + pred2, out4h + pred2,
            #                      out5v + pred2, out2f + pred2, out3f + pred2, out4f + pred2)
            # out2h, out3h, out4h, out5v, out2f, out3f, out4f = torch.split(feedback2, 64, 1)
            # pred3 = self.feedback2(feedback3)

            shape = x.size()[2:] if shape is None else shape

            pred1a = F.interpolate(self.linearpa(pred1), size=shape, mode='bilinear')
            pred2a = F.interpolate(self.linearpb(pred2), size=shape, mode='bilinear')
            pred3a = F.interpolate(self.linearpc(pred3), size=shape, mode='bilinear')

            return pred1a, pred2a, pred3a

    def initialize(self):
        # if self.cfg.snapshot:
        #     self.load_state_dict(torch.load(self.cfg.snapshot))
        # else:
        weight_init(self)


if __name__ == '__main__':
        net = INet(cfg=None)
        input = torch.zeros([2, 3, 380, 380])
        # size: 380*380
        # out2h:95*95 out3h:48*48 out4h:24*24 out5v:12*12
        # out1f:95*95 out2f:48*48 out3f:24*24 out4v:24*24
        # size: 224*224
        # out2h:56*56 out3h:28*28 out4h:14*14 out5v:7*7
        # out1f:56*56 out2f:28*28 out3f:14*14 out4v:14*14
        output = net(input, input)
        output = net(input)
