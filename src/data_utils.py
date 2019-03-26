# =============================================================================
# Data utils
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, glob
import librosa
from scipy import signal
from scipy.io import wavfile
import librosa.display


# File directories
data_filedir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'data', 'external', 'train')

class DataUtils():
    def __init__(self, speaker):
        self.speaker = speaker
        self.speaker_path = os.path.join(data_filedir, self.speaker)
        self.speaker_png_path = os.path.join(self.speaker_path, 'spectrograms')
        self.speaker_mel_png_path = os.path.join(self.speaker_path , 'mel_spectrograms')

        
    def read_folder_content(self):
        available_files = glob.glob1(self.speaker_path, '*.wav')
        return available_files
    
    def original_spectrogram_conversion(self, file):
        filename = file.replace('.wav', '')
        sample_rate, samples = wavfile.read(os.path.join(self.speaker_path,file))
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        plt.imshow(np.log(spectrogram))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(os.path.join(self.speaker_png_path ,'{}.png'.format(filename)))
        
    def mel_spectrogram_conversion(self, file):
        filename = file.replace('.wav', '')
        sample_rate, samples = librosa.load(os.path.join(self.speaker_path,  file))
        spectrogram = librosa.feature.melspectrogram(y=sample_rate, sr=samples, n_mels = 128, fmax= 8000)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                                              y_axis='mel', 
                                                              fmax=8000,
                                                              x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig(os.path.join(self.speaker_mel_png_path +'{}.png'.format(filename)))
            

# =============================================================================
# Usage
# =============================================================================

# Defining speaker
utils = DataUtils('ritz')

# Spectrogram conversion
for i in utils.read_folder_content():
    utils.original_spectrogram_conversion(i)
