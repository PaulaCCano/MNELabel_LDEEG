import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel.gui import label_ica_components

# === Parámetros de simulación del EEG ===
sfreq = 250.0
n_channels = 8
n_samples = 1000 # Número de muestras
ch_names = [f'EEG{i+1}' for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
data = np.random.randn(n_channels, n_samples) * 1e-6
raw = mne.io.RawArray(data, info)

# === Cargar matrices mixing y unmixing ===
mixing_csv_path = r"C:\Users\paula\OneDrive\Escritorio\Portables\eeg_wavelet\wICASSO\sub-CTR001_ses-V0_task-CE_desc-wavelet_eeg_wICAsso_mixing.csv"
unmixing_csv_path = r"C:\Users\paula\OneDrive\Escritorio\Portables\eeg_wavelet\wICASSO\sub-CTR001_ses-V0_task-CE_desc-wavelet_eeg_wICAsso_unmixing.csv"

mixing_matrix = pd.read_csv(mixing_csv_path, header=None).dropna(axis=1).values
unmixing_matrix = pd.read_csv(unmixing_csv_path, header=None).dropna(axis=1).values
n_components = mixing_matrix.shape[1]

# === Crear objeto ICA manualmente ===
ica = ICA(n_components=n_components, random_state=97)
ica.current_fit = 'unwhitened'
ica.n_components_ = n_components
ica.mixing_matrix_ = mixing_matrix
ica.unmixing_matrix_ = unmixing_matrix
ica.pca_components_ = np.eye(n_channels)[:n_components]
ica.pca_mean_ = np.zeros(n_channels)
ica.pca_explained_variance_ = np.ones(n_components)

# ✅ Agrega atributos que MNE espera
ica.reject_ = None
ica.ch_names = raw.info['ch_names']
ica.pre_whitener_ = np.ones((n_components,))


