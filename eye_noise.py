import torch
import numpy as np
from scipy.fft import fft
from sklearn.decomposition import FastICA
from einops import rearrange
from scipy.signal import savgol_filter


def filter_eye_noise(words_tensor):
    ica = FastICA(n_components=2, algorithm='deflation', random_state=21)
    words_tensor = words_tensor.numpy()
    t_left = [(0, 3, 4), (0, 4, 5), (1, 4, 5)]
    t_right = [(2, 6, 7), (1, 5, 6), (2, 5, 6)]
    comps_left = np.zeros(shape=(3, 1024))
    comps_right = np.zeros(shape=(3, 1024))
    for i in range(3):
        comps = ica.fit_transform(words_tensor[:, t_left[i]])
        a = comps[:, 0]
        b = comps[:, 1]

        diff1 = abs(a.mean() - a.max())
        diff2 = abs(b.mean() - b.max())
        if diff1 > diff2:
            first_component = comps[:, 0]
        else:
            first_component = comps[:, 1]

        first_fft = fft(first_component, 1024)
        first_fft = abs(first_fft)

        first_fft[0] = first_fft[-1] = 0
        comps_left[i] = first_fft
    new_comp_left = rearrange(np.mean(comps_left, axis=0), 'n -> 1 n')

    for i in range(3):
        comps = ica.fit_transform(words_tensor[:, t_right[i]])
        a = comps[:, 0]
        b = comps[:, 1]

        diff1 = abs(a.mean() - a.max())
        diff2 = abs(b.mean() - b.max())
        if diff1 > diff2:
            first_component = comps[:, 0]
        else:
            first_component = comps[:, 1]
        first_fft = fft(first_component, 1024)
        first_fft = abs(first_fft)

        first_fft[0] = first_fft[-1] = 0
        comps_right[i] = first_fft
    new_comp_right = rearrange(np.mean(comps_right, axis=0), 'n -> 1 n')

    words_tensor = abs(fft(rearrange(words_tensor, 't c -> c t'), axis=1))
    words_tensor = savgol_filter(words_tensor, 51, 3)
    words_tensor = np.concatenate((words_tensor, new_comp_left, new_comp_right), axis=0)
    return torch.from_numpy(words_tensor)
