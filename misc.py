import numpy as np
import os
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
from math import exp
# import pydensecrf.densecrf as dcrf
import torch.nn.functional as F
torch_ver = torch.__version__[:3]

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.dsn_weight = dsn_weight

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        # print(preds[0].size())
        # print(target.size())

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2

class CriterionKL(nn.Module):
    def __init__(self):
        super(CriterionKL, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, target):
        assert preds.size() == target.size()

        n, c, w, h = preds.size()
        softmax_preds = F.softmax(target.permute(0, 2, 3, 1).contiguous().view(-1, c), dim=1)
        loss = (torch.sum(-softmax_preds * self.log_softmax(preds.permute(0, 2, 3, 1).contiguous().view(-1, c)))) / w / h

        return loss

class CriterionKL2(nn.Module):
    def __init__(self):
        super(CriterionKL2, self).__init__()

    def forward(self, preds, target):
        assert preds.size() == target.size()

        b, c, w, h = preds.size()
        preds = F.softmax(preds.view(b, -1), dim=1)
        target = F.softmax(target.view(b, -1), dim=1)
        loss = (preds * (preds / target).log()).sum() / b

        return loss

class CriterionStructure(nn.Module):
    def __init__(self):
        super(CriterionStructure, self).__init__()
        self.gamma = 2
        self.ssim = SSIM(window_size=11,size_average=True)

    def forward(self, pred, target):
        assert pred.size() == target.size()

        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        ##### focal loss #####
        p_t = torch.exp(-wbce)
        f_loss = (1 - p_t) ** self.gamma * wbce

        ##### ssim loss #####
        # ssim = 1 - self.ssim(pred, target)

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (1 * wbce + 0.8 * wiou + 0.05 * f_loss).mean()

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class CriterionKL3(nn.Module):
    def __init__(self):
        super(CriterionKL3, self).__init__()

    def KLD(self, input, target):
        input = input / torch.sum(input)
        target = target / torch.sum(target)
        eps = sys.float_info.epsilon
        return torch.sum(target * torch.log(eps + torch.div(target, (input + eps))))

    def forward(self, input, target):
        assert input.size() == target.size()

        return _pointwise_loss(lambda a, b:self.KLD(a,b), input, target)

class CriterionPairWise(nn.Module):
    def __init__(self, scale):
        super(CriterionPairWise, self).__init__()
        self.scale = scale

    def L2(self, inputs):
        return (((inputs ** 2).sum(dim=1)) ** 0.5).reshape(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]) + 1e-8

    def similarity(self, inputs):
        inputs = inputs.float()
        tmp = self.L2(inputs).detach()
        inputs = inputs / tmp
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        return torch.einsum('icm, icn->imn', [inputs, inputs])

    def sim_dis_compute(self, preds, targets):
        sim_err = ((self.similarity(targets) - self.similarity(preds)) ** 2) / ((targets.size(-1) * targets.size(-2)) ** 2) / targets.size(0)
        sim_dis = sim_err.sum()
        return sim_dis

    def forward(self, preds, targets):
        total_w, total_h = preds.shape[2], preds.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        max_pooling = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        loss = self.sim_dis_compute(max_pooling(preds), max_pooling(targets))
        return loss

class CriterionDice(nn.Module):
    def __init__(self):
        super(CriterionDice, self).__init__()

    def forward(self, pred, target):
        n = target.size(0)
        smooth = 1
        pred = F.sigmoid(pred)
        pred_flat = pred.view(n, -1)
        target_flat = target.view(n, -1)

        intersection = pred_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / n

        return loss

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
# def crf_refine(img, annos):
#     def _sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     assert img.dtype == np.uint8
#     assert annos.dtype == np.uint8
#     assert img.shape[:2] == annos.shape
#
#     # img and annos should be np array with data type uint8
#
#     EPSILON = 1e-8
#
#     M = 2  # salient or not
#     tau = 1.05
#     # Setup the CRF model
#     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
#
#     anno_norm = annos / 255.
#
#     n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
#     p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))
#
#     U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
#     U[0, :] = n_energy.flatten()
#     U[1, :] = p_energy.flatten()
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
#
#     # Do the inference
#     infer = np.array(d.inference(1)).astype('float32')
#     res = infer[1, :]
#
#     res = res * 255
#     res = res.reshape(img.shape[:2])
#     return res.astype('uint8')

if __name__ == '__main__':
    pixel_wise_loss = CriterionKL3()
    pair_wise_loss = CriterionPairWise(scale=0.5)
    ppa_wise_loss = CriterionStructure()
    preds = torch.rand([2, 1, 380, 380])
    # print(torch.sum(F.softmax(preds, dim=1)))
    targets = torch.ones([2, 1, 380, 380])
    # loss = pixel_wise_loss(F.sigmoid(preds), F.sigmoid(preds))
    loss = F.kl_div(preds, preds)
    # loss2 = pair_wise_loss(preds, targets)
    print(ppa_wise_loss(preds, targets))
    # print(loss2)
