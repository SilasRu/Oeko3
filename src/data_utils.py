# =============================================================================
# Data utils
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, fnmatch
import librosa
from scipy import signal
from scipy.io import wavfile
import librosa.display

# File directories
data_filedir = os.path.abspath('..\\data')

class DataUtils():
    def __init__(self, speaker):
        self.speaker = speaker
        self.speaker_path = data_filedir + '\\external\\train\\'  + self.speaker + '\\'
        self.speaker_png_path = self.speaker_path + 'spectrograms\\'
        self.speaker_mel_png_path = self.speaker_path + 'mel_spectrograms\\'
        
    def read_folder_content(self):
        listOfFiles = os.listdir(self.speaker_path)
        pattern = '*.wav'
        available_files = {}
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                available_files.update({'{}'.format(entry): entry})
        return available_files
    
    def original_spectrogram_conversion(self, file):
        filename = file.replace('.wav', '')
        sample_rate, samples = wavfile.read(self.speaker_path + file)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        plt.imshow(np.log(spectrogram))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(self.speaker_png_path +'{}.png'.format(filename))
        
    def mel_spectrogram_conversion(self, file):
        filename = file.replace('.wav', '')
        sample_rate, samples = librosa.load(self.speaker_path + file)
        spectrogram = librosa.feature.melspectrogram(y=sample_rate, sr=samples, n_mels = 128, fmax= 8000)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                                              y_axis='mel', 
                                                              fmax=8000,
                                                              x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig(self.speaker_mel_png_path +'{}.png'.format(filename))
            

        

utils = DataUtils('berset')
utils.speaker_mel_png_path
utils.read_folder_content()

for i in utils.read_folder_content():
        utils.original_spectrogram_conversion(i)
        
        
sample_rate, samples = librosa.load('1_august_000.wav')
spectrogram = librosa.feature.melspectrogram(y=sample_rate, sr=samples, n_mels = 128, fmax= 8000)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(spectrogram,ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()