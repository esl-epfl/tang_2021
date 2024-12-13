import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tang_2021.architecture import DCRNNModel_classification
import tang_2021.tang2021_utils as tang2021_utils

def load_model(args, device):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model.pth.tar')

    model = DCRNNModel_classification(args=args, num_classes=args.num_classes, device=device)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)

    return model

def load_thresh():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return float(np.load(os.path.join(dir_path, 'best_thresh.npy')))

class hyperparams:
    def __init__(self):
        self.num_nodes = 19
        self.num_rnn_layers = 2
        self.rnn_units = 64
        self.input_dim = 384
        self.output_dim = 384
        self.max_seq_len = 57
        self.num_classes = 1
        self.graph_type = 'combined'
        self.max_diffusion_step = 2
        self.cl_decay_steps = 3000
        self.top_k = 3
        self.dcgru_activation = 'tanh'
        self.filter_type = 'laplacian'
        self.dropout = 0.0

class SeizureDataset(nn.Module):
    def __init__(self, data, window_size_sec, fs, args):
        super(SeizureDataset, self).__init__()

        dir_path = os.path.dirname(os.path.realpath(__file__))
        means_dir = os.path.join(dir_path, 'standardize', 'means_seq2seq_fft_60s_szdetect_single.pkl')
        stds_dir = os.path.join(dir_path, 'standardize', 'stds_seq2seq_fft_60s_szdetect_single.pkl')
        with open(means_dir, 'rb') as f:
            means = pickle.load(f)
        with open(stds_dir, 'rb') as f:
            stds = pickle.load(f)
        self.scaler = tang2021_utils.StandardScaler(means, stds)

        self.window_size = int(window_size_sec*fs)
        self.fs = fs
        self.recording_duration = int(data.shape[1] / fs)
        self.max_seq_len = args.max_seq_len
        self.graph_type = args.graph_type
        self.top_k = args.top_k
        self.filter_type = args.filter_type

        data = self.remap_channels(data)
        self.data = data

        window_idx = np.arange(0, self.recording_duration*self.fs, self.fs).astype(int) # 1s between each window
        self.window_idx = window_idx[window_idx < self.recording_duration*self.fs - self.window_size]
        
        self.adj_mat_dir = os.path.join(dir_path, 'dist_graph_adj_mx.pkl')
        adj_mat = self._get_combined_graph(swap_nodes=None)
        self.supports = self._compute_supports(adj_mat)

    def __len__(self):
        return len(self.window_idx)
    
    def remap_channels(self, eeg_data):
        idx_remap = [0, 11, 1, 12, 2, 13, 3, 14, 4, 15,
                     5, 16, 6, 17, 7, 18, 8, 9, 10]
        idx_remap = np.array(idx_remap)
        eeg_remap = eeg_data[idx_remap, :]
        return eeg_remap
    
    def fourier_transform(self, eeg_data):
        physical_time_step_size = int(self.fs * self.max_seq_len)
        eeg_fft, _ = tang2021_utils.computeFFT(eeg_data, physical_time_step_size)
        return eeg_fft
    
    def _get_combined_graph(self, swap_nodes=None):
        """
        Get adjacency matrix for pre-computed distance graph
        Returns:
            adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
        """
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]
        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]
                            ] = adj_mat[node_pair[1], node_pair[0]]
                adj_mat_new[node_pair[1], node_pair[0]
                            ] = adj_mat[node_pair[0], node_pair[1]]

        return adj_mat_new
    
    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                tang2021_utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports
    
    def __getitem__(self, idx):
        eeg_clip = self.data[:, self.window_idx[idx]:self.window_idx[idx]+self.window_size]
        eeg_clip = self.fourier_transform(eeg_clip)
        eeg_clip = self.scaler.transform(eeg_clip)
        eeg_clip = torch.Tensor(eeg_clip)

        supports = self.supports
        seq_len = torch.LongTensor([self.max_seq_len])

        return (eeg_clip, seq_len, supports)

def get_dataloader(data, window_size_sec, fs, args):
    dataset = SeizureDataset(data, window_size_sec, fs, args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
    return dataloader

def predict(model, dataloader, device, recording_duration, window_size_sec=57, overlap_sec=56, fs=256, thresh=0.5):
    y_preds = []
    
    model.eval()
    with torch.no_grad():
        for i, (eeg_clip, seq_len, supports) in enumerate(dataloader):
            eeg_clip = eeg_clip.to(device)
            seq_len = seq_len.view(-1).to(device)
            for j in range(len(supports)):
                supports[j] = supports[j].to(device)
            logits = model(eeg_clip, seq_len, supports)
            logits = logits.view(-1)
            y_prob = torch.sigmoid(logits).cpu().detach().numpy()
            y_pred = (y_prob > thresh).astype(int)
            y_preds.extend(list(y_pred))

    y_preds = np.array(y_preds)

    if len(y_preds) == 0:
        return np.zeros(int(recording_duration*fs))

    y_predict = np.zeros(int(recording_duration*fs))
    window_size = int(window_size_sec*fs)
    overlap = int(overlap_sec*fs)
    for i in range(window_size_sec, len(y_preds)):
        # for each time point, assign majority of all overlaps
        overlap_left = max(0, i*(window_size-overlap))
        overlap_right = min(len(y_predict), (i+1)*(window_size-overlap))
        majority_vote = np.mean(y_preds[max(0, i-window_size_sec):min(i, recording_duration)]) > 0.5
        y_predict[int(overlap_left):int(overlap_right)] = majority_vote.astype(int)

    return y_predict
