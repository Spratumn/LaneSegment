import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    class_num (Int): class number equal to `C`
    weight (Tensor, optional): a manual rescaling weight given to each class.
        If given, has to be a Tensor or list of size `C`
    size_average (bool, optional): By default,the losses are averaged over each loss element in the batch. Note that for
        some losses, there are multiple elements per sample. If the field :attr:`size_average`
        is set to ``False``, the losses are instead summed for each minibatch. Ignored
        when reduce is ``False``. Default: ``True``
    ignore_index (int, optional): Specifies a target value that is ignored
        and does not contribute to the input gradient. When :attr:`size_average` is
        ``True``, the loss is averaged over non-ignored targets.
    """

    def __init__(self, class_num, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        if weight is not None:
            # if isinstance(weight, list):
            #     weight = torch.Tensor(weight)
            assert isinstance(weight, torch.Tensor)
            assert class_num == weight.size(0)
        
        self.class_num = class_num
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        """
        inputs: shape of (N,C) or (N,C,H,W)
        target: shape of (N) or (N,H,W)
        """
        if inputs.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(inputs.size(0), target.size(0)))
        if inputs.size()[2:] != target.size()[1:]:
            raise ValueError('Expected input feature_size ({}) to match target feature_size ({}).'
                             .format(inputs.size()[2:], target.size()[1:]))
        dim_num = inputs.dim()
        if dim_num < 2:
            raise ValueError('Expected 2 or more dimensions (got {})'.format(dim_num))
        if dim_num > 2:
            # N,C,H,W => N,C,H*W
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  
            # N,C,H*W => N,H*W,C
            inputs = inputs.transpose(1, 2)  
            # N,H*W,C => N*H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2)) 
        # N,H,W => N*H*W,1 
        target = target.view(-1, 1)
        target = target.long()
        logpt = F.log_softmax(inputs, dim=1)
        loss = -1 * logpt.gather(1, target)  # N*H*W,C => N*H*W,1
        if self.weight is not None:
            self.weight = self.weight.view(-1, 1)
            target_weight = self.weight.gather(0, target)
            loss = loss * target_weight
        if self.ignore_index in range(self.class_num):
            ignore = torch.ones(self.class_num)
            ignore[self.ignore_index] = 0
            ignore = ignore.view(-1, 1)
            target_ignore = ignore.gather(0, target)
            loss = loss * target_ignore
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2, alpha=1,
                 weight=None, size_average=True, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert 0 < self.alpha <= 1
        if weight is not None:
            # if isinstance(weight, list):
            #     weight = torch.Tensor(weight)
            assert isinstance(weight, torch.Tensor)
            assert class_num == weight.size(0)
        self.class_num = class_num
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        """
        inputs: shape of (N,C) or (N,C,H,W)
        target: shape of (N) or (N,H,W)
        """
        if inputs.size(0) != target.size(0):
            raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                             .format(inputs.size(0), target.size(0)))
                            
        if inputs.size()[2:] != target.size()[1:]:
            raise ValueError('Expected input feature_size ({}) to match target feature_size ({}).'
                             .format(inputs.size()[2:], target.size()[1:]))
        dim_num = inputs.dim()
        if dim_num < 2:
            raise ValueError('Expected 2 or more dimensions (got {})'.format(dim_num))

        if dim_num > 2:
            # N,C,H,W => N,C,H*W
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  
            # N,C,H*W => N,H*W,C
            inputs = inputs.transpose(1, 2)  
            # N,H*W,C => N*H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2)) 
        # N,H,W => N*H*W,1 
        target = target.view(-1, 1)
        target = target.long()
        pt = F.softmax(inputs, dim=1)
        # N*H*W,C => N*H*W,1
        pt = pt.gather(1, target) 
        logpt = pt.log()
        if self.weight is not None:
            self.weight = self.weight.view(-1, 1)
            target_weight = self.weight.gather(0, target)
            pt = pt * target_weight
            logpt = logpt * target_weight
        if self.ignore_index in range(self.class_num):
            ignore = torch.ones(self.class_num)
            ignore[self.ignore_index] = 0
            ignore = ignore.view(-1, 1)
            target_ignore = ignore.gather(0, target)
            pt = pt * target_ignore
            logpt = logpt * target_ignore
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            loss = self.alpha * loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

