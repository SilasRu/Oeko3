# =============================================================================
# Data utils
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, fnmatch
from scipy import signal
from scipy.io import wavfile

# File directories
data_filedir = os.path.abspath('..\\data')

class DataUtils():
    def __init__(self, speaker):
        self.speaker = speaker
        self.speaker_path = data_filedir + '\\external\\train\\'  + self.speaker + '\\'
        self.speaker_png_path = self.speaker_path + 'spectrograms\\'
        
    def read_folder_content(self):
        listOfFiles = os.listdir(self.speaker_path)
        pattern = '*.wav'
        available_files = {}
        for entry in listOfFiles:
            if fnmatch.fnmatch(entry, pattern):
                available_files.update({'{}'.format(entry): entry})
        return available_files
    
    def spectrogram_conversion(self, file):
        filename = file.replace('.wav', '')
        sample_rate, samples = wavfile.read(self.speaker_path + file)
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
#        plt.pcolormesh(times, frequencies, np.log(spectrogram))
#        plt.imshow(np.log(spectrogram))
#        plt.ylabel('Frequency [Hz]')
#        plt.xlabel('Time [sec]')
#        plt.savefig(self.speaker_png_path +'{}.png'.format(filename))
        

utils = DataUtils('berset')
utils.speaker_png_path

utils.read_folder_content()
utils.spectrogram_conversion('1_august_000.wav')


sample_rate, samples = wavfile.read('D:\\GitHub\\Oeko3\\data\\external\\train\\berset\\1_august_000.wav')

for i in utils.read_folder_content():
        utils.spectrogram_conversion(i)