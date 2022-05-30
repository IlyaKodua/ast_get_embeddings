import torchaudio
import numpy as np
import torch
import torch.nn.functional
from models import ASTModel
import glob
import pickle
import os
import time

batch_size = 32


def wav2fbank(filename, filename2=None):
    # mixup
    if filename2 == None:
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
    # mixup
    else:
        waveform1, sr = torchaudio.load(filename)
        waveform2, _ = torchaudio.load(filename2)

        waveform1 = waveform1 - waveform1.mean()
        waveform2 = waveform2 - waveform2.mean()

        if waveform1.shape[1] != waveform2.shape[1]:
            if waveform1.shape[1] > waveform2.shape[1]:
                # padding
                temp_wav = torch.zeros(1, waveform1.shape[1])
                temp_wav[0, 0:waveform2.shape[1]] = waveform2
                waveform2 = temp_wav
            else:
                # cutting
                waveform2 = waveform2[0, 0:waveform1.shape[1]]

        # sample lambda from uniform distribution
        #mix_lambda = random.random()
        # sample lambda from beta distribtion
        mix_lambda = np.random.beta(10, 10)

        mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
        waveform = mix_waveform - mix_waveform.mean()

    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

    target_length = 998
    n_frames = fbank.shape[0]

    p = target_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if filename2 == None:
        return fbank, 0
    else:
        return fbank, mix_lambda


def get_tensor(filename):


    fbank, mix_lambda = wav2fbank(filename)


    return fbank[None,:,:]


def to_batch(files, n = 64):
    new_array = []
    id_start = 0
    id_end = n
    while id_end < len(files):
        new_array.append(files[id_start:id_end])
        id_start += n
        id_end += n
    new_array.append(files[id_start:len(files)])

    return new_array


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ast_mdl = ASTModel(label_dim=7, input_tdim=998, imagenet_pretrain=True, audioset_pretrain=True)
ast_mdl = ast_mdl.to(device)

data_dir = "/home/liya/datasets/dcase_2022_data/"
dir_out = "out"

os.mkdir(dir_out)

list_files = glob.glob(data_dir + '**/*.wav', recursive=True)

list_files = to_batch(list_files, batch_size)
persent = 0
timer = time.time()
print("Start!!!")
ast_mdl.eval()
for i, filename in enumerate(list_files):
    cycl_perent = int((i+1)/len(list_files)*100)
    
    if(cycl_perent != persent):
        persent = cycl_perent
        print(persent, " %")
        print(time.time() - timer, " s")
        timer = time.time()
    
    fbank = torch.zeros(len(list_files[i]),  998, 128)
    for i in range(len(list_files[i])):
        fbank[i,:,:] = get_tensor(filename[i])
    
    with torch.no_grad():
        embeddings_batch = ast_mdl(fbank.to(device))
    
    for i in range(batch_size):
        embedding = embeddings_batch[i].detach().cpu().numpy()
        file_name_split = filename[i].split("/")
        
        wav_name = file_name_split[-1]
        name = file_name_split[-1].split(".")[0]
        with open(dir_out + "/" +file_name_split[-3] +"_" +file_name_split[-2] +"_"+ name +".pkl", 'wb') as f:
            pickle.dump(embedding, f)






