# Se importa MNE y herramientas de MNE para realizar el ICA
# y una función que permite etiquetar automáticamente los componentes ICA
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components

# Carga un archivo de ejemplo de datos EEG disponible en MNE
# sample_data_folder = mne.datasets.sample.data_path() / "MEG" / "sample"
# sample_data_raw_file = sample_data_folder / "sample_audvis_raw.fif"
# raw = mne.io.read_raw_fif(sample_data_raw_file)


# Carga un archivo de ejemplo de datos EEG disponible en MNE
sample_data_raw_file = r"C:\Users\paula\OneDrive - Universidad de Antioquia\UdeA\JI\EEG_CE\sub01-CTR_ses-V0_task_CE_eeg.fif"
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)


# we'll crop to 60 seconds and drop MEG channels
# Recorta el archivo para analizar solo los primeros 60 segundos
raw.crop(tmax=60.0).pick_types(eeg=True, stim=True, eog=True) # filtra los tipos de canales: EEG, estímulosy electrooculograma
raw.load_data() # carga los datos en memoria

# pick some channels that clearly show heartbeats and blinks
# selecciona los canales que claramente muestran latidos del corazón y parpadeos
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=r"(EEG 00.)") # selecciona los canales EEG que contienen el patrón de artefacto mediante una expresión regular
print("artifact_picks:", artifact_picks)
print("raw.ch_names:", raw.ch_names)
# plot the raw data with the selected channels
raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)

# aplica un filtro pasa banda entre 1 y 100 Hz
filt_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
# se selecciona la referencia
filt_raw = filt_raw.set_eeg_reference("average")