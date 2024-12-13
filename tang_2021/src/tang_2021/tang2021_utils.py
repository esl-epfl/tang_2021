import torch
import numpy as np
from scipy.fftpack import fft
from scipy.sparse import linalg
import scipy.sparse as sp

"""
Some code are adapted from https://github.com/liyaguang/DCRNN,
https://github.com/xlwang233/pytorch-DCRNN and 
https://github.com/tsy935/eeg-gnn-ssl, which are
licensed under the MIT License.
"""

INCLUDED_CHANNELS = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

def last_relevant_pytorch(output, lengths, batch_first=True):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension).long()
    masks = masks.to(output.device)
    last_output = output.gather(2, masks).squeeze(time_dimension) # [Samanos] CHANGED: was last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)
    return last_output

def compute_sampling_threshold(cl_decay_steps, global_step):
    """
    Compute scheduled sampling threshold
    """
    return cl_decay_steps / \
        (cl_decay_steps + np.exp(global_step / cl_decay_steps))

class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean  # (1,num_nodes,1)
        self.std = std  # (1,num_nodes,1)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, is_tensor=False, device=None, mask=None):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
            device: device
            mask: shape (batch_size,) nodes where some signals are masked
        """
        mean = self.mean.copy()
        std = self.std.copy()
        if len(mean.shape) == 0:
            mean = [mean]
            std = [std]
        if is_tensor:
            mean = torch.FloatTensor(mean)
            std = torch.FloatTensor(std)
            if device is not None:
                mean = mean.to(device)
                std = std.to(device)
        return (data * std + mean)

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(
        adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P