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
import pandas as pd
from datetime import datetime



# File directories
data_filedir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'data', 'train', 'audio_files')
data_filedir_test = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'data', 'test', 'mono', 'mono_rn')

class AudioConvert():
    def __init__(self, speaker):
        self.speaker = speaker
        if self.speaker == 'test':
            self.speaker_path = data_filedir_test
            self.speaker_png_path = os.path.join(data_filedir_test, os.pardir, os.pardir, os.pardir,
                                                 'train', 'audio_files', 'rest_noise')

        else:
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



class Utils():
    def __init__(self):
        pass

    # Create y_test for arena
    def create_y_test(self, speaker_list):
        speaker_df = pd.read_csv(speaker_list)

        speakers = {'Rytz Regula': 4,
                    'Projer Jonas': 2,
                    'Gössi Petra': 1,
                    'Berset Alain': 0,
                    'Rösti Albert': 3}

        y_test = list()
        # Loop trough speaker_list
        for row in range(len(speaker_df)):
            current_speaker = speaker_df['Wer'][row]
            # Parse the time strings for each row
            t1 = datetime.strptime(speaker_df['Von'][row], '%H:%M:%S')
            t2 = datetime.strptime(speaker_df['Bis'][row], '%H:%M:%S')
            # Calculate time delta
            delta = int((t2-t1).total_seconds())

            # Append speaker variable to y_test
            if current_speaker in speakers.keys():
                y_test.extend([speakers[current_speaker]]*delta)
            else:
                y_test.extend([5]*delta)
        return y_test


# y_test = create_y_test()



# =============================================================================
# Usage
# =============================================================================

# utils = DataUtils('test')
# utils.speaker_mel_png_path
# datalist = utils.read_folder_content()[3900:]
# len(utils.read_folder_content())
#
# # Spectrogram conversion
# for i in utils.read_folder_content():
#     utils.original_spectrogram_conversion(i)
#
# for i in datalist:
#     utils.original_spectrogram_conversion(i)
