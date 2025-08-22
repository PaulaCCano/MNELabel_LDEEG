# Se importa MNE y herramientas de MNE para realizar el ICA
# y una función que permite etiquetar los componentes ICA mediante una GUI
# Se importa la librería os para manejar rutas de archivos
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel.gui import label_ica_components, _label_components
from mne_icalabel import label_components
import numpy as np

# Carga un archivo de ejemplo de datos EEG de la base de datos local
# path_data = 'C:/Users/paula/OneDrive/Escritorio/Portables/eeg_wavelet/wICASSO'
# output_path = 'C:\Users\paula\OneDrive\Escritorio\Portables\ica_components'
sample_data_raw_file = r"C:\Users\paula\OneDrive\Escritorio\Portables\eeg_wavelet\wICASSO\sub-CTR001_ses-V0_task-CE_desc-wavelet_eeg_wICAsso.fif"
    
raw = mne.io.read_raw_fif(sample_data_raw_file) # carga el archivo .fif en un objeto Raw de MNE

# Here we'll crop to 60 seconds and drop gradiometer channels for speed
raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True) # Recorta la señal a 60 segundos y selecciona los tipos de canales que se van a usar (magnetómetros, EEG, estimulación y EOG)
# Corregir nombres para que coincidan con los del montaje estándar
raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
raw.set_montage('standard_1020')
raw.load_data() # Carga los datos en memoria




# Cargar las matrices de separación y mezcla desde un archivo .npz
data = np.load(r"C:\Users\paula\OneDrive\Escritorio\Portables\eeg_wavelet\wICASSO\sub-CTR001_ses-V0_task-CE_desc-wavelet_eeg_wICAsso_raw_ICAdata.npz")
W = data['W']
A = data['A']
quality = data['quality']
ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=1000) # Crea un objeto ICA con 8 componentes, una semilla aleatoria y un máximo de 1000 iteraciones
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
ica.fit(filt_raw)

# Ingresar las matrices calculadas
ica._unmixing = W     # matriz de separación
ica._mixing = A    # matriz de mezcla (inversa)

label_components(filt_raw, ica, method='iclabel')
# Now, we can take a look at the components, which were modified in-place
# for the ICA instance.
print(ica.labels_)