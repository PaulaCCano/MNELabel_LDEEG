"""
Script: MNE_ICLabel_automatico.py
Editores: Paula Andrea C Cano, Sara Garcés, Juan Esteban Pineda
Fecha: 08-2025

Descripción:
Este script clasifica componentes independientes extraídos mediante ICA utilizando la
herramienta ICLabel de MNE-ICALabel. Se carga la señal EEG y los resultados previos
de una descomposición ICA (matrices W y A), para posteriormente etiquetar los componentes
en categorías como cerebro, músculo, ojo, corazón, línea eléctrica, canal defectuoso o ruido.

Entradas:
- Archivo .fif con la señal filtrada por un filtro wavelet y la descomposición ICA previa.
- Archivos .npz con las matrices W. y A, y la calidad de los componentes extraídos.

Salidas:
- Archivo .npy con sufijo anexo "_ICAlabel", en el que se almacenan las etiquetas de los componentes ICA.

Dependencias:
- Python 3.9 o mayor
- MNE-Python
- numpy
- pandas
- os
- mne
"""

# Se importa MNE y herramientas de MNE para realizar el ICA
# y una función que permite etiquetar los componentes ICA mediante una GUI
# Se importa la librería os para manejar rutas de archivos
import os
import mne
from mne.preprocessing import ICA
from mne_icalabel.gui import label_ica_components, _label_components
from mne_icalabel import label_components
import numpy as np

# Configuración de la ruta de los datos y la salida
# Asegúrate de que estas rutas sean correctas en tu sistema
path_data = 'C:/Users/paula/OneDrive/Escritorio/Portables/eeg_wavelet/wICASSO'
output_path = 'C:/Users/paula/OneDrive/Escritorio/Portables/ica_components'

for file in os.listdir(path_data):
    if file.endswith('.fif'):
        data_file = os.path.join(path_data, file) # Construye la ruta completa del archivo .fif
    
        raw = mne.io.read_raw_fif(data_file) # carga el archivo .fif en un objeto Raw de MNE

        # Here we'll crop to 60 seconds and drop gradiometer channels for speed
        raw.crop(tmax=60.0).pick_types(meg="mag", eeg=True, stim=True, eog=True) # Recorta la señal a 60 segundos y selecciona los tipos de canales que se van a usar (magnetómetros, EEG, estimulación y EOG)
        # Corregir nombres para que coincidan con los del montaje estándar
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'}) # Se redefine el nombre de los canales FP1 y FP2 para que coincidan con el estándar 10-20
        raw.set_montage('standard_1020') # Se establece el montaje estándar 10-20 para los canales EEG
        raw.load_data() # Carga los datos en memoria

    if file.endswith('.npz'):
        npzdata_path = os.path.join(path_data, file) # Construye la ruta completa del archivo .npz
        print(f"Labeling {npzdata_path}") 
        
        data = np.load(npzdata_path) # Carga el archivo .npz que contiene las matrices W, A y la calidad de los componentes
        W = data['W'] 
        A = data['A'] 
        quality = data['quality'] # La calidad de los componentes es un array que contiene la calidad de cada componente ICA
        ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=1000) # Crea un objeto ICA con 8 componentes, una semilla aleatoria y un máximo de 1000 iteraciones
        filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None) # Aplica un filtro pasa-bajos a la señal EEG para eliminar el ruido de alta frecuencia
        ica.fit(filt_raw) # Ajusta el objeto ICA a la señal filtrada

        # Ingresar las matrices calculadas
        ica._unmixing = W     # matriz de separación
        ica._mixing = A    # matriz de mezcla (inversa)

        label_components(filt_raw, ica, method='iclabel') # Etiqueta los componentes ICA utilizando la herramienta ICLabel de MNE
        # Now, we can take a look at the components, which were modified in-place
        # for the ICA instance.
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(f"Labels for components in {file}:") 
        print(ica.labels_)

        print(type(ica.labels_))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


        np.save(npzdata_path.replace('.npz', '_ICAlabel'), ica.labels_) # Guarda las etiquetas de los componentes en un archivo .npy con el sufijo "_ICAlabel"
