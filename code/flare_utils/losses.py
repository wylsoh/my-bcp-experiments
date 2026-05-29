import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from abc import ABC
import contextlib

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass

class mask_DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(mask_DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class DiceLossbcp(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossbcp, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def _dice_mask_loss(self, score, target, mask):
        target = target.float()
        mask = mask.float()
        smooth = 1e-10
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth ) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, mask=None, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        if mask is not None:
            # bug found by @CamillerFerros at github issue#25
            mask = mask.repeat(1, self.n_classes, 1, 1, 1).type(torch.float32)
            for i in range(0, self.n_classes): 
                dice = self._dice_mask_loss(inputs[:, i], target[:, i], mask[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        else:
            for i in range(0, self.n_classes):
                dice = self._dice_loss(inputs[:, i], target[:, i])
                class_wise_dice.append(1.0 - dice.item())
                loss += dice * weight[i]
        return loss / self.n_classes

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss







class DiceLoss3D(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss3D, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """
        input_tensor: [B, D, H, W] 整数标签
        返回 one-hot 编码: [B, C, D, H, W]
        """
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, margin=None):
        """
        score, target: [B, D, H, W] -> 单个类别的预测与标签
        """
        smooth = 1e-5
        target = target.float()

        if margin is not None:
            intersect = torch.sum(score * target * margin)
            y_sum = torch.sum(target * target * margin)
            z_sum = torch.sum(score * score * margin)
        else:
            intersect = torch.sum(score * target)
            y_sum = torch.sum(target * target)
            z_sum = torch.sum(score * score)

        loss = (2. * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, margin=None, weight=None, softmax=False):
        """
        inputs: [B, C, D, H, W] - 模型输出概率或 logits
        target: [B, D, H, W] - 标签
        margin: [B, 1, D, H, W] or None - 可选的权重 mask
        weight: list or tensor of shape [C] - 每类的权重
        """
        if softmax:
            inputs = F.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)  # [B, C, D, H, W]

        if weight is None:
            weight = [1.0] * self.n_classes

        assert inputs.size() == target.size(), "predict & target shape do not match"

        total_loss = 0.0
        for i in range(self.n_classes):
            input_i = inputs[:, i]
            target_i = target[:, i]
            if margin is not None:
                loss_i = self._dice_loss(input_i, target_i, margin)
            else:
                loss_i = self._dice_loss(input_i, target_i)
            total_loss += loss_i * weight[i]

        return total_loss / self.n_classes

dice_loss_3d = DiceLoss3D(14)
def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def Binary_dice_loss_weight(predictive, target, ep=1e-8):
    # import ipdb; ipdb.set_trace()
    # predictive = nn.Softmax(dim=1)(predictive)
    target = torch.argmax(target, dim=1)
    predictive = predictive[:, 1, ...]

    N = predictive.shape[0]
    inter_2 = predictive * target
    union_2 = predictive + target

    inter_2 = inter_2.view(N, 1, -1).sum(2)
    union_2 = union_2.view(N, 1, -1).sum(2)

    # smooth to prevent overfitting
    # [https://github.com/pytorch/pytorch/issues/1249]
    # NxC
    dice = (2 * inter_2 + ep) / (union_2 + ep)

    return 1 - dice.mean(1)


def kl_loss(inputs, targets, ep=1e-8):
    kl_loss = nn.KLDivLoss(reduction='mean')
    consist_loss = kl_loss(torch.log(inputs + ep), targets)
    return consist_loss


def soft_ce_loss(inputs, target, ep=1e-8):
    logprobs = torch.log(inputs + ep)
    return torch.mean(-(target[:, 0, ...] * logprobs[:, 0, ...] + target[:, 1, ...] * logprobs[:, 1, ...]))


def mse_loss(input1, input2):
    return torch.mean((input1 - input2)**2)


def mse_loss_mask(input1, input2, mask):
    loss = (input1 - input2)**2 * mask
    return loss.sum() / (mask.sum() + 1e-8)


def ce_loss_mask(input1, input2, mask):
    loss_f = nn.CrossEntropyLoss(reduction='none')
    loss = loss_f(input1, input2) * mask
    return loss.sum() / (mask.sum() + 1e-8)


def ce_loss_gap(input1, mask, map):
    loss_f = nn.CrossEntropyLoss(reduction='none')
    loss = loss_f(input1, mask)
    gap = 1 - torch.abs(input1 - map)
    return (loss * gap).mean()


def mse_loss_weight(input1, input2, weight):
    # weight = weight.expand_as(input1)
    loss = (input1 - input2)**2
    loss = torch.mean(loss, dim=(1, 2, 3)) * weight
    return torch.mean(loss)


def mse_loss_weight_3d(input1, input2, weight):
    loss = (input1 - input2)**2
    loss = torch.mean(loss, dim=(1, 2, 3, 4)) * weight
    return torch.sum(loss)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # ex: input_tensor(label) -> temp_prob : Nx96x96x96 -> Nx1x96x96x96
        #     tensor_list -> output_tensor: Cx[Nx1x96x96x96] -> NxCx96x96x96
        input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10
        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i], weighted_pixel_map)
            # class_wise_dice.append(dice_loss)
            loss += dice_loss * weight[i]

        return loss / self.n_classes



def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]):  # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / (
                np.max(posdis) - np.min(posdis))
            sdf[boundary == 1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf


class meanIOU:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(self.num_classes * label_true[mask].astype(int) + label_pred[mask],
                           minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        # for lp, lt in zip(predictions, gts):
        self.hist = self._fast_hist(predictions.flatten(), gts.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.base_dim() == 4
    assert target.base_dim() == 4
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(2), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(3), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.base_dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)

    loss = F.cross_entropy(predict, target, size_average=True)
    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.base_dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class DiceLoss_weight(nn.Module):

    def __init__(self, nclass=None, class_weights=None, smooth=1e-5, thres=0.5):
        # https://github.com/grant-jpg/FUSSNet/blob/0b7632154a69909f5c48fe7e2fde8809ec2914d1/utils1/loss.py#L80
        super(DiceLoss_weight, self).__init__()
        self.smooth = smooth
        self.thres = thres

    def get_mask(self, out):
        _, masks = torch.max(out, dim=1)
        # masks = out.argmax(dim=1).detach()
        return masks

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)

        target = self.get_mask(target)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        pred, _ = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC

        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean(1)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax)**2
    return mse_loss


#针对多分类问题，二分类问题更简单一点
class SoftIoULoss(nn.Module):

    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(SoftIoULoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot - inter

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (1 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class RegionLoss_2D(nn.Module):

    def __init__(self, spatial_size):
        super(RegionLoss_2D, self).__init__()

        self.average_pool = nn.AdaptiveAvgPool2d((spatial_size, spatial_size))

    def forward(self, p1, p2, dis_f='MSE'):

        p1_avg = self.average_pool(p1)
        p2_avg = self.average_pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        sim_a = torch.cosine_similarity(aa.unsqueeze(1), aa.unsqueeze(0), dim=-1)
        sim_b = torch.cosine_similarity(bb.unsqueeze(1), bb.unsqueeze(0), dim=-1)

        if dis_f == 'MSE':
            diff = nn.MSELoss()(sim_a, sim_b)
        else:
            diff = nn.L1Loss()(sim_a, sim_b)

        return diff


class RegionLoss_3D(nn.Module):
    # https://github.com/Youmin-Kim/GLD/blob/main/gld.py
    def __init__(self, spatial_size, pool='Avg'):
        super(RegionLoss_3D, self).__init__()

        if pool == 'Avg':
            self.pool = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))
        elif pool == 'Max':
            self.pool = nn.AdaptiveMaxPool3d((spatial_size, spatial_size, spatial_size))

    def forward(self, p1, p2, dis_f='MSE'):

        p1_avg = self.pool(p1)
        p2_avg = self.pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        sim_a = torch.cosine_similarity(aa.unsqueeze(1), aa.unsqueeze(0), dim=-1)
        sim_b = torch.cosine_similarity(bb.unsqueeze(1), bb.unsqueeze(0), dim=-1)

        if dis_f == 'MSE':
            diff = nn.MSELoss()(sim_a, sim_b)
        else:
            diff = nn.L1Loss()(sim_a, sim_b)

        return diff


class RegionLoss_3D_Mask(nn.Module):
    # https://github.com/Youmin-Kim/GLD/blob/main/gld.py
    def __init__(self, spatial_size, pool='Avg'):
        super(RegionLoss_3D_Mask, self).__init__()

        if pool == 'Avg':
            self.pool = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))
        elif pool == 'Max':
            self.pool = nn.AdaptiveMaxPool3d((spatial_size, spatial_size, spatial_size))

    def forward(self, p1, p2, mask, dis_f='MSE'):

        p1 = p1 * mask
        p2 = p2 * mask

        p1_avg = self.pool(p1)
        p2_avg = self.pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        sim_a = torch.cosine_similarity(aa.unsqueeze(1), aa.unsqueeze(0), dim=-1)
        sim_b = torch.cosine_similarity(bb.unsqueeze(1), bb.unsqueeze(0), dim=-1)

        if dis_f == 'MSE':
            diff = nn.MSELoss()(sim_a, sim_b)
        else:
            diff = nn.L1Loss()(sim_a, sim_b)

        return diff


class RegionLoss_3D_kl(nn.Module):
    # https://github.com/Youmin-Kim/GLD/blob/main/gld.py
    def __init__(self, spatial_size, pool='Avg'):
        super(RegionLoss_3D_kl, self).__init__()

        if pool == 'Avg':
            self.pool = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))
        elif pool == 'Max':
            self.pool = nn.AdaptiveMaxPool3d((spatial_size, spatial_size, spatial_size))

    def forward(self, p1, p2, dis_f='MSE'):

        p1_avg = self.pool(p1)
        p2_avg = self.pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1)
        # aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        # bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        # add for loop on single image
        # use kl_div batch or single
        N = p1.shape[0]
        diff = 0
        for i in range(N):
            sim_a = torch.cosine_similarity(aa[i].unsqueeze(1), aa[i].unsqueeze(0), dim=-1)
            sim_b = torch.cosine_similarity(bb[i].unsqueeze(1), bb[i].unsqueeze(0), dim=-1)
            diff += nn.MSELoss()(sim_a, sim_b)

        # sim_a = torch.cosine_similarity(aa.unsqueeze(1), aa.unsqueeze(0), dim=-1)
        # sim_b = torch.cosine_similarity(bb.unsqueeze(1), bb.unsqueeze(0), dim=-1)

        # if dis_f == 'MSE':
        #     diff = nn.MSELoss()(sim_a, sim_b)
        # else:
        #     diff = nn.L1Loss()(sim_a, sim_b)

        return diff / N


class RegionLoss_3D_multi(nn.Module):
    # https://github.com/Youmin-Kim/GLD/blob/main/gld.py
    def __init__(self, spatial_size, spatial_size2, pool='Avg'):
        super(RegionLoss_3D_multi, self).__init__()

        if pool == 'Avg':
            self.pool1 = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))
            self.pool2 = nn.AdaptiveAvgPool3d((spatial_size2, spatial_size2, spatial_size2))
        elif pool == 'Max':
            self.pool = nn.AdaptiveMaxPool3d((spatial_size, spatial_size, spatial_size))

    def forward(self, p1, p2, dis_f='MSE'):

        p1_avg = self.pool1(p1)
        p2_avg = self.pool1(p2)

        p3_avg = self.pool2(p1)
        p4_avg = self.pool2(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        cc = p3_avg.view(p3_avg.size(0), p3_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        dd = p4_avg.view(p4_avg.size(0), p4_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        # add for loop on single image
        # use kl_div batch or single
        p1c = torch.cat((aa, cc), dim=0)
        p2c = torch.cat((bb, dd), dim=0)

        sim_a = torch.cosine_similarity(p1c.unsqueeze(1), p1c.unsqueeze(0), dim=-1)
        sim_b = torch.cosine_similarity(p2c.unsqueeze(1), p2c.unsqueeze(0), dim=-1)

        # sim_c = torch.cosine_similarity(cc.unsqueeze(1), cc.unsqueeze(0), dim=-1)
        # sim_d = torch.cosine_similarity(dd.unsqueeze(1), dd.unsqueeze(0), dim=-1)

        if dis_f == 'MSE':
            diff = nn.MSELoss()(sim_a, sim_b)
        else:
            diff = nn.L1Loss()(sim_a, sim_b)

        return diff


class RegionLoss_3D_multi_three(nn.Module):
    # https://github.com/Youmin-Kim/GLD/blob/main/gld.py
    def __init__(self, spatial_size=[1, 3, 5], pool='Avg'):
        super(RegionLoss_3D_multi_three, self).__init__()

        if pool == 'Avg':
            self.pool1 = nn.AdaptiveAvgPool3d((spatial_size[0], spatial_size[0], spatial_size[0]))
            self.pool2 = nn.AdaptiveAvgPool3d((spatial_size[1], spatial_size[1], spatial_size[1]))
            self.pool3 = nn.AdaptiveAvgPool3d((spatial_size[2], spatial_size[2], spatial_size[2]))
        elif pool == 'Max':
            self.pool = nn.AdaptiveMaxPool3d((spatial_size, spatial_size, spatial_size))

    def forward(self, p1, p2, dis_f='MSE'):

        p1_avg = self.pool1(p1)
        p2_avg = self.pool1(p2)

        p3_avg = self.pool2(p1)
        p4_avg = self.pool2(p2)

        p5_avg = self.pool2(p1)
        p6_avg = self.pool2(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        cc = p3_avg.view(p3_avg.size(0), p3_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        dd = p4_avg.view(p4_avg.size(0), p4_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        ee = p5_avg.view(p5_avg.size(0), p5_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        ff = p6_avg.view(p6_avg.size(0), p6_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)

        # add for loop on single image
        # use kl_div batch or single

        sim_a = torch.cosine_similarity(aa.unsqueeze(1), aa.unsqueeze(0), dim=-1)
        sim_b = torch.cosine_similarity(bb.unsqueeze(1), bb.unsqueeze(0), dim=-1)

        sim_c = torch.cosine_similarity(cc.unsqueeze(1), cc.unsqueeze(0), dim=-1)
        sim_d = torch.cosine_similarity(dd.unsqueeze(1), dd.unsqueeze(0), dim=-1)

        sim_e = torch.cosine_similarity(ee.unsqueeze(1), ee.unsqueeze(0), dim=-1)
        sim_f = torch.cosine_similarity(ff.unsqueeze(1), ff.unsqueeze(0), dim=-1)

        if dis_f == 'MSE':
            diff1 = nn.MSELoss()(sim_a, sim_b)
            diff2 = nn.MSELoss()(sim_c, sim_d)
            diff3 = nn.MSELoss()(sim_e, sim_f)
        else:
            diff = nn.L1Loss()(sim_a, sim_b)

        return (diff1 + diff2 + diff3) / 3.


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        #print("features:", features.size())

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            #print("labels", labels)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            #print("mask", mask)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        import ipdb
        ipdb.set_trace()

        return loss


class RegionLoss_3D_info(nn.Module):

    def __init__(self, spatial_size):
        super(RegionLoss_3D_info, self).__init__()

        self.average_pool = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))

        # self.project_head = nn.Conv3d()
        size_c = 16
        self.project_head = nn.Sequential(nn.Linear(size_c, size_c), nn.ReLU(), nn.Linear(size_c, size_c))
        self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, features, size, temperature):

        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = torch.cat([torch.arange(size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / temperature
        return logits, labels

    def forward(self, p1, p2, temperature=0.1):

        p1_avg = self.average_pool(p1)
        p2_avg = self.average_pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        mm = torch.cat((aa, bb), dim=0)
        mm = self.project_head(mm)
        logits, labels = self.info_nce_loss(features=mm, size=aa.shape[0], temperature=temperature)

        loss = self.criterion(logits, labels)

        return loss


class RegionLoss_3D_cos(nn.Module):

    def __init__(self, spatial_size):
        super(RegionLoss_3D_cos, self).__init__()

        self.average_pool = nn.AdaptiveAvgPool3d((spatial_size, spatial_size, spatial_size))

        # self.project_head = nn.Conv3d()
        size_c = 16
        self.project_head = nn.Sequential(nn.Linear(size_c, size_c), nn.ReLU(), nn.Linear(size_c, size_c))
        self.criterion = torch.nn.CrossEntropyLoss()

    def info_nce_loss(self, features, size, temperature):

        # labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = torch.cat([torch.arange(size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / temperature
        return logits, labels

    def forward(self, p1, p2, temperature=0.1):

        p1_avg = self.average_pool(p1)
        p2_avg = self.average_pool(p2)

        aa = p1_avg.view(p1_avg.size(0), p1_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        bb = p2_avg.view(p2_avg.size(0), p2_avg.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        mm = torch.cat((aa, bb), dim=0)
        mm = self.project_head(mm)
        logits, labels = self.info_nce_loss(features=mm, size=aa.shape[0], temperature=temperature)

        loss = self.criterion(logits, labels)

        return loss

class MarginWeightedCrossEntropyLoss3D(nn.Module):
    def __init__(self):
        super(MarginWeightedCrossEntropyLoss3D, self).__init__()

    def forward(self, inputs, targets, margin=None, softmax=True):
        """
        inputs: (B, C, D, H, W) — logits
        targets: (B, D, H, W) — int64 ground truth labels
        margin: (B, D, H, W) or (B, 1, D, H, W) — weight map
        """
        if softmax:
            inputs = F.log_softmax(inputs, dim=1)  # softmax over channel
            loss_map = F.nll_loss(inputs, targets, reduction='none')  # shape: (B, D, H, W)
        else:
            loss_map = F.cross_entropy(inputs, targets, reduction='none')  # shape: (B, D, H, W)

        if margin is not None:
            margin = margin.float()
            # If margin has shape (B, 1, D, H, W), squeeze to (B, D, H, W)
            if margin.dim() == 5:
                margin = margin.squeeze(1)
            loss = (margin * loss_map).mean()
        else:
            loss = loss_map.mean()

        return loss


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)
def _l2_normalize(d):
    # pdb.set_trace()
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8  ###2-p length of vector
    return d
class VAT3d(nn.Module):

    def __init__(self, xi=10.0, epi=6.0, ip=1):
        super(VAT3d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = dice_loss_3d

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x)[0], dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)  ### initialize a random tensor between [-0.5, 0.5]
        d = _l2_normalize(d)  ### an unit vector
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)
                pred_hat = model(x + self.xi * d)[0]
                p_hat = F.softmax(pred_hat, dim=1)
                adv_distance = self.loss(p_hat, torch.argmax(pred,dim=1,keepdim=True))
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
            pred_hat = model(x + self.epi * d)[0]
            p_hat = F.softmax(pred_hat, dim=1)
            lds = self.loss(p_hat, torch.argmax(pred,dim=1,keepdim=True))
        return lds

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    # print(y_onehot.size())

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.weight = weight

    def forward(self, x, y, loss_mask=None, is_training=True):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        if self.weight is not None:  # <--
            if not self.do_bg and self.batch_dice:
                dc *= self.weight[1:]
            else:
                raise NotImplementedError

        if not is_training:
            return dc
        else:
            return -dc.mean()
