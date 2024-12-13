import torch
import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg
from tang_2021.utils import load_model, load_thresh, hyperparams, get_dataloader, predict

def main(edf_file, outFile):
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)

    assert eeg.montage is Eeg.Montage.UNIPOLAR, "Error: Only unipolar montages are supported."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = hyperparams()
    model = load_model(args=args, device=device)
    thresh = load_thresh()

    dataloader = get_dataloader(data=eeg.data, window_size_sec=57, fs=eeg.fs, args=args)

    recording_duration = eeg.data.shape[1] / eeg.fs
    window_size_sec = 57
    overlap_sec = 56

    y_predict = predict(model=model, dataloader=dataloader, device=device, recording_duration=recording_duration, 
                        window_size_sec=window_size_sec, overlap_sec=overlap_sec, fs=eeg.fs, thresh=thresh)

    hyp = Annotations.loadMask(y_predict, eeg.fs)
    hyp.saveTsv(outFile)

