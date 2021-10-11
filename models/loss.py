import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from functools import partial
from sklearn.metrics import pairwise_distances
import numpy as np


def kd_loss(n_img, out, prev_out, T=2):
    """
    Compute the knowledge-distillation (KD) loss with soft targets.

    Parameters
        ----------
        target_outputs : ,required
            Outputs from the frozen Net A
        outputs : , required
            Outputs of the original classes from the adaptive Net B
        n_img: int, required
            Number of images from the new m classes in a mini-batch
        T : optional
            Temperature factor normally set to 2
    
    Returns
        -------
        KD_loss:
            Computed knowledge-distillation loss

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of target_outputs
    and outputs expects the input tensor to be log probabilities! 
    """
    log_p = torch.log_softmax(out / T, dim=1)
    q = torch.softmax(prev_out / T, dim=1)
    res = torch.nn.functional.kl_div(log_p, q, reduction='none')
    res = res.sum() / n_img

    #KD_loss = -1/n_img * nn.KLDivLoss((F.log_softmax(outputs/T, dim=1), F.softmax(target_outputs/T, dim=1)) , reduction=None)
    
    return res

def DistillationLoss(logits_a, logits_b, temperature=1.):
    return F.binary_cross_entropy_with_logits(
        torch.sigmoid(logits_a / temperature), 
        torch.sigmoid(logits_b / temperature)
    )


def pairwise_distance(x, y):
    """
    Compute the euclidean distances between each representation of the first batch with each representation of the second batch.

    Parameters
        ----------
        x : ,required
            Matrix (batch_dim, features) that represent a batch of representations
        y : , required
            Matrix (batch_dim, features) that represent a batch of representations
    
    Returns
        -------
        output:
            Matrix (batch_dim_y, batch_dim_x) that contains in the FIRST row the distances beetween each representation of x with the FIRST representation of y,
            in the SECOND row the distances beetween each representation of x with the SECOND representation of y....
    """

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):
    """
    Compute the gaussian kernel matrix K where each element k(R_i,R_j) = exp(-(||R_i - R_j||_2^2)/(2*sigma_m)). sigma_m mean m variances in the Gaussian kernel
    and returns the sum of each element of the matrix
    Parameters
        ----------
        x : , required
            Matrix (batch_dim, features) that represent a batch of representations
        y : , required
            Matrix (batch_dim, features) that represent a batch of representations
        sigmas: , required
            Vector of the m variances in the gaussian kernel
    Returns
        -------
        Return Values:
            The sum of the values in the Gaussian kernel matrix
    """

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    #print(dist_.shape)
    #print(beta.shape)
    exp_arg = torch.matmul(beta, dist_)
    #print(exp_arg.shape)
    #print(torch.sum(torch.exp(-exp_arg), 0).shape)
    return torch.sum(torch.exp(-exp_arg), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix): 
    """
    Compute the maximum mean discrepancy between two matrix given a specific kernel matrix.
    Parameters
        ----------
        x : , required
            Matrix (batch_dim, features) that represent a batch of representations
        y : , required
            Matrix (batch_dim, features) that represent a batch of representations
        kernel: , required
            Matrix that represent a specific kernel
    Returns
        -------
        Cost:

    """
    #cost1 = kernel(x,x) + kernel(y,y) - 2*kernel(x,y)
    #print((cost1)/(x.shape[0]**2))

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):
    """
    Set a vector of sigmas of the gaussian kernel, compute the gaussian kernel
    and compute the MMD loss between a batch of source features and a batch of target features
    Parameters
        ----------
        source_features : , required
            Matrix (batch_dim, features) that represent a batch of representations, in particular the batch of features extracted from the adaptive Net
        target_features : , required
            Matrix (batch_dim, features) that represent a batch of representations, in particular the batch of features extracted from the frozen Net
    Returns
        -------
        Loss value:
            MMD loss
    """

    if(source_features.is_cuda or target_features.is_cuda):
        use_gpu = 1
    else:
        use_gpu = 0
        
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
	
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        #print(bandwidth)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        #print(bandwidth_list)
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        #print(kernel_val)
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss


if __name__ == "__main__":
    a = torch.tensor([[2.1, 100.], [1.012, 3.11], [1., 3]])
    b = torch.tensor([[2., 5.003], [1., 3.], [1.08, 3]])


    #print(pairwise_distances(a.numpy(),b.numpy(),"euclidean")**2)
    #print(pairwise_distance(a,b))

    print(mmd_loss(a,b))

    loss = MMD_loss(kernel_num=5)
    print(loss(a,b))

    

    