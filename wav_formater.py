
import os
import pandas as pd
import numpy as np
# audio editing libs
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import wave
import contextlib

sound = AudioSegment.from_file('test_audio.wav')

duration = 0
new_num_cuts = 0

# this block of code performs necessary calculations to ensure frame lengths are 16 milliseconds long (Andy).
with contextlib.closing(wave.open('test_audio.wav','r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    new_num_cuts = duration/0.016
    print(new_num_cuts)
    print(duration)
    print(duration/new_num_cuts)

frame_file_path = "frames1-forth/"
np_frame_file_path = "npframes1-forth/"
#num_cuts = 8135 // 4
n_mels = 320
n_fft = 2048
hop_length = 100
try:
    os.stat(frame_file_path)
except:
    os.mkdir(frame_file_path)

try:
    os.stat(np_frame_file_path)
except:
    os.mkdir(np_frame_file_path)


# new_num_cuts replaces num_cuts. new_num_cuts is calculated from the length of the audio file to generate 16 ms long frames (Andy).
#size_frame = len(sound) // num_cuts
size_frame = len(sound) // new_num_cuts
#step_size = len(sound) / num_cuts
step_size = len(sound) / new_num_cuts
sound_set = []
center = size_frame
center_true = step_size
#for i in range(num_cuts):
for i in range(int(new_num_cuts)):
    start = center - size_frame
    stop = center + size_frame
    # sanity check
    if start < 0:
        start = 0
    if stop > len(sound):
        stop = len(sound)
    sound_set.append(sound[start:stop])
    center_true = center_true + step_size
    center = int(center_true)

f_num = []
for i, frame in enumerate(sound_set):
    f_num.append(i)
    frame.export(frame_file_path + "{}.wav".format(i),format="wav")

for i in f_num:
    wav = "{}.wav".format(i)
    # here kaiser_fast is a technique used for faster extraction
    audio, sample_rate = librosa.load(frame_file_path + wav, res_type='kaiser_fast')
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40
    S = librosa.feature.melspectrogram(audio, sr=sample_rate, n_fft=n_fft)
    # if i == 100:
    #     librosa.display.specshow(S, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.show()

   ##########################
   #  DATA PROCESSING TIME  #
   ##########################

# load the image
img = np.load("audio1.npy")

count = 0
for im in os.listdir("npframes1-forth"):
    count += 1

count = count - 1
# convert to numpy array
img = np.asarray(img)
# save shape to pass to convolution
img_shape = list(img.shape)
img_shape.reverse()
img_shape.append(count)
img_shape.reverse()
print(img_shape)
if img_shape[0] < 1:
    img_shape[0] = img_shape[0]*-1
data = np.zeros(img_shape)
data.shape

for i, im in enumerate(list(range(count))):
    img = np.load("npframes1-forth/mel_spec.npy".format(im))
    if img.shape == (320,8):
        img = np.reshape(img, (320, 8,1))
        data[i,:,:,:] = img
    else:
        print("skipped ", im)

np.save("audio1.npy", data)
#data2 = np.load("audio1.npy")
#data2[8128,:,:,:]
#data[50,:,:,:]

hop_length = (hop_length, n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
S_nb = np.asmatrix(S_DB)
np.save(np_frame_file_path + "mel_spec".format(i), S_nb)

# print(mel_spec)
# print(mel_db)

#print out entire mel_spec array at once. The output of the entire array all at once is pink.
# Intelligble output is shown when picking one image and showing it.
# I've printed out one frame on line 80

# librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.show()

# trash # zone # trash # zone # trash # zone # trash # zone # trash # zone # trash # zone
# sound_set[0].get_array_of_samples()
# temp = train.apply(parser, axis=1)
# temp.columns = ['feature', 'label']
#
# def parser(row):
#    # function to load files and extract features
#    file_name = os.path.join(os.path.abspath(data_dir), 'Train', str(row.ID) + '.wav')
#    # handle exception to check if there isn't a file which is corrupted
#    try:
#       # here kaiser_fast is a technique used for faster extraction
#       X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
#       # we extract mfcc feature from data
#       mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
#    except Exception as e:
#       print("Error encountered while parsing file: ", file)
#       return None, None
#    feature = mfccs
#    label = row.Class
#    return [feature, label]

