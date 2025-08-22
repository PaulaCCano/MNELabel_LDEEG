import numpy as np

# Cargar el archivo .npz
datos = np.load("C:/Users/paula/OneDrive/Escritorio/Portables/eeg_wavelet/wICASSO/sub-SAN053_ses-V0_task-CE_desc-wavelet_eeg_wICAsso_raw_ICAdata_ICAlabel.npz.npy", allow_pickle=True)

# Mostrar los datos
print(datos)
