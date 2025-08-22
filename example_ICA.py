import mne
raw = mne.io.read_raw_fif(r"C:\Users\paula\OneDrive\Escritorio\Portables\eeg_wavelet\sample_audvis_raw.fif")


print(raw.info)

raw.crop(tmin=100, tmax=130)  # take 30 seconds for speed

# pick only EEG channels, muscle artifact is basically not picked up by MEG
# if you have a simultaneous recording, you may want to do ICA on MEG and EEG
# separately
raw.pick(picks="eeg", exclude="bads")

# ICA works best with a highpass filter applied
raw.load_data()
raw.filter(l_freq=1.0, h_freq=None)

ica = mne.preprocessing.ICA(
    n_components=15, method="fastica", max_iter="auto", random_state=97
)
ica2 = ica.fit(raw)

print(raw.info)

print(ica2)