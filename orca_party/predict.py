#!/usr/bin/env python3
"""
Module: predict.py
Authors: Christian Bergler, Manuel Schmitt
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 03.03.2022
"""

import os
import argparse

import numpy as np
import scipy.io.wavfile
import data.transforms as T
import data.signal as signal

import torch
import torch.nn as nn

from os import listdir
from math import ceil, floor
from os.path import isfile, join
from utils.logging import Logger
from collections import OrderedDict

from models.unet_model import UNet
from visualization.cm import viridis_cm
from visualization.utils import spec2img, flip
from data.audiodataset import DefaultSpecDatasetOps, StridedAudioDataset, SingleAudioFolder

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--denoiser",
    type=str,
    default=None,
    help="Path to the trained denoiser pickle")

parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    help="Print additional training and model information.",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Path to a model.",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint. If provided the checkpoint will be used instead of the model.",
)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="The directory to store the logs."
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="The directory to store the output."
)

parser.add_argument(
    "--sequence_len",
    type=float,
    default=2,
    help="Sequence length in [s]."
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="The number of images per batch."
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--visualize",
    dest="visualize",
    action="store_true",
    help="Additional visualization of the noisy vs. denoised spectrogram",
)

parser.add_argument(
    "--audio",
    dest="audio",
    action="store_true",
    help="Additional generation of (denoised --- optional) masked audio signal spectrograms",
)

parser.add_argument(
    "--jit_load",
    dest="jit_load",
    action="store_true",
    help="Load model via torch jit (otherwise via torch load).",
)

parser.add_argument(
    "--min_max_norm",
    dest="min_max_norm",
    action="store_true",
    help="activates min-max normalization instead of default 0/1-dB-normalization.",
)

parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Input file could either be a directory with multiple audio files or just one single audio file"
)

parser.add_argument(
    "--data_opts_path",
    type=str,
    default=None,
    help="Path to the data option file used for generation the overlapped network training data"
)

parser.add_argument(
    "--class_info",
    type=str,
    help="The path to the automatic generated class.info file, build by the generate_overlap_dataset.py.",
)

ARGS = parser.parse_args()

log = Logger("PREDICT", ARGS.debug, ARGS.log_dir)


"""
Main function to compute prediction by using a trained model together with the given input
"""
if __name__ == "__main__":

    debug = ARGS.debug
    log_dir = ARGS.log_dir
    cuda = ARGS.cuda
    denoiser = ARGS.denoiser
    jit_load = ARGS.jit_load
    visualize = ARGS.visualize
    batch_size = ARGS.batch_size
    model_path = ARGS.model_path
    input_file = ARGS.input_file
    class_info = ARGS.class_info
    output_dir = ARGS.output_dir
    num_workers = ARGS.num_workers
    sequence_len = ARGS.sequence_len
    audio = ARGS.audio
    min_max_norm = ARGS.min_max_norm
    data_opts_path = ARGS.data_opts_path
    checkpoint_path = ARGS.checkpoint_path

    log.info(f"Logging Level: {debug}")
    log.info(f"GPU-Cuda support: {cuda}")
    log.info(f"Directory of the Logging Output: {log_dir}")
    log.info(f"Input Data (File or Directory): {input_file}")
    log.info(f"File of the Sound Source Separation Model (.pk): {model_path}")
    log.info(f"Loading Models with JIT: {jit_load}")
    log.info(f"File of the Denoising Model (.pk): {denoiser}")
    log.info(f"Visualize class-specific (denoised) separated spectrograms: {visualize}")
    log.info(f"Batch Size: {batch_size}")
    log.info(f"Class Info File including Number and Labels of each Class: {class_info}")
    log.info(f"Directory of the entire output: {ARGS.output_dir}")
    log.info(f"Number of workers for initial data loading: {num_workers}")
    log.info(f"Sequence length (window-size) for frame-wise evaluation (step-size is equal to one half time window-size): {sequence_len}")
    log.info(f"Generation of class-specific sound separated audio results: {audio}")
    log.info(f"Min-Max-Normalization activated: {min_max_norm}")
    log.info(f"Data options used for data generation in order to train the sound separation model: {data_opts_path}")
    log.info(f"Loading the sound separation model from a given checkpoint: {checkpoint_path}")

    with open(class_info) as file:
        classes = [line.rstrip('\n') for line in file]
    num_classes = len(classes)

    if checkpoint_path is not None:
        log.info(
            "Restoring checkpoint from {} instead of using a model file.".format(
                checkpoint_path
            )
        )
        checkpoint = torch.load(checkpoint_path)
        model = UNet(1, num_classes, bilinear=False)
        model.load_state_dict(checkpoint["modelState"])
        log.warning(
            "Using default preprocessing options. Provide Model file if they are changed"
        )
        dataOpts = DefaultSpecDatasetOps
    else:
        if jit_load:
            extra_files = {}
            extra_files['dataOpts'] = ''
            model = torch.jit.load(model_path, _extra_files=extra_files)
            unetState = model.state_dict()
            dataOpts = eval(extra_files['dataOpts'])
            log.debug("Source separation model successfully load via torch jit: " + str(model_path))
        else:
            model_dict = torch.load(model_path)
            model = UNet(1, num_classes, bilinear=False)
            model.load_state_dict(model_dict["unetState"])
            model = nn.Sequential(
                OrderedDict([("denoiser", model)])
            )
            dataOptsModel = model_dict["dataOpts"]
            log.debug("Source separation model successfully load via torch load: " + str(model_path))

    dataOptsGen = {}
    with open(data_opts_path, "r") as file:
        for line in file:
            info = line.split("=")
            dataOptsGen[info[0].strip()] = info[1].strip()

    log.info("Data Options Network Training Data Generation: " + str(dataOptsGen))

    log.info("Data Options Source Separation Network Model: " + str(dataOptsModel))

    log.info(model)

    if audio:
        makeAudio = True
    else:
        makeAudio = False

    if visualize:
        sp = signal.signal_proc()
    else:
        sp = None

    if torch.cuda.is_available() and cuda:
        model = model.cuda()

    model.eval()

    #Data Options Data Generation Overlapping Network Input Data
    sr = int(dataOptsGen['sr'])
    fmin = int(dataOptsGen["fmin"])
    fmax = int(dataOptsGen["fmax"])
    n_fft = int(dataOptsGen["n_fft"])
    ref_level_db = int(dataOptsGen["refDB"])
    min_level_db = int(dataOptsGen["minDB"])
    hop_length = int(dataOptsGen["hop_length"])
    n_freq_bins = int(dataOptsGen["n_freq_bins"])
    min_max_norm = dataOptsGen["min_max_norm"]
    freq_cmpr = dataOptsGen["freq_compression"]

    if min_max_norm:
        log.debug("Init min-max-normalization activated")
    else:
        log.debug("Init 0/1-dB-normalization activated")

    sequence_len = int(ceil(sequence_len * sr))

    hop = int(sequence_len / 2)

    hop_factor = sequence_len/hop

    input_file = input_file

    if os.path.isdir(input_file):

        log.debug("Init Single Folder Audio Dataset - Predicting Files")
        log.debug("Audio folder to process: "+str(input_file))
        audio_files = [f for f in listdir(input_file) if isfile(join(input_file, f))]

        dataset = SingleAudioFolder(
            file_names=audio_files,
            working_dir=input_file,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_freq_bins=n_freq_bins,
            freq_compression=freq_cmpr,
            f_min=fmin,
            f_max=fmax,
            center=True,
            min_max_normalize=min_max_norm
        )

        log.info("number of files to predict={}".format(len(audio_files)))
        log.info("files will be entirely denoised without subsampling parts and/or padding")
        concatenate = False
    elif os.path.isfile(input_file):

        log.debug("Init Strided Audio Dataset - Predicting Files")
        log.debug("Audio file to process: "+str(input_file))

        dataset = StridedAudioDataset(
             file_name=input_file.strip(),
             log=log,
             sequence_len=sequence_len,
             hop=hop,
             sr=sr,
             fft_size=n_fft,
             fft_hop=hop_length,
             n_freq_bins=n_freq_bins,
             f_min=fmin,
             f_max=fmax,
             freq_compression=freq_cmpr,
             center=True,
             min_max_normalize=min_max_norm
        )

        log.info("size of the file(samples)={}".format(dataset.n_frames))
        log.info("size of hop(samples)={}".format(hop))
        stop = int(max(floor(dataset.n_frames / hop), 1))
        log.info("stop time={}".format(stop))
        concatenate = True
    else:
        raise Exception("Not a valid data format - neither folder nor file")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    t_decompr_f = T.Decompress(f_min=fmin, f_max=fmax, n_fft=n_fft, sr=sr)

    # Load the denoiser - see also: https://github.com/ChristianBergler/ORCA-CLEAN
    if denoiser is not None:
        log.info("Denoising Option enabled - Denoiser loading...")
        if jit_load:
            extra_files = {}
            extra_files['dataOpts'] = ''
            denoiser_model = torch.jit.load(denoiser, _extra_files=extra_files)
            unetState = denoiser_model.state_dict()
            dataOpts = eval(extra_files['dataOpts'])
            log.debug("Denoiser successfully load via torch jit: " + str(denoiser))
        else:
            denoiser_model_dict = torch.load(denoiser)
            denoiser_model = UNet(1, 1, bilinear=False)
            denoiser_model.load_state_dict(denoiser_model_dict["unet"]) #unetState
            denoiser_model = nn.Sequential(
                OrderedDict([("denoiser", denoiser_model)])
            )
            dataOpts = model_dict["dataOpts"]
            log.debug("Denoiser successfully load via torch load: " + str(model_path))

    if torch.cuda.is_available() and cuda:
        denoiser_model = denoiser_model.cuda()
    denoiser_model.eval()

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    total_audio_proc = []

    for i in classes:
        total_audio_proc.append(None)
    total_audio_proc.append(None)
    total_audio = total_audio_proc.copy()

    long_class = [[] for _ in range(num_classes + 1)]

    with torch.no_grad():

        for i, input in enumerate(data_loader):

            if os.path.isdir(input_file):
                total_audio = total_audio_proc.copy()
                audio_gen = True
            else:
                if i % hop_factor == 0:
                    audio_gen = True
                else:
                    audio_gen = False

            sample_spec_orig, input, spec_cmplx, filename = input

            time_dim = input.shape[2]
            freq_dim = input.shape[3]

            log.info("current file in process, " + str(i) + "-iterations: " + str(filename[0]))

            if torch.cuda.is_available() and cuda:
                input = input.cuda()

            #Denoising if available and required
            if denoiser_model is not None:
                input = denoiser_model(input)

            input2 = input.clone()

            #Separation output
            party_output = model(input)

            party_output_proc = party_output.clone().cpu()

            #Visualization
            if sp is not None:

                if os.path.isfile(input_file):
                    temp_input = input.squeeze(dim=0).squeeze(dim=0)[:time_dim, :].cpu()

                    long_class[0].append(temp_input)

                    for j in range(num_classes):
                        temp = party_output.squeeze(dim=0).squeeze(dim=0)[j][:time_dim, :].cpu()
                        temp = temp * temp_input
                        long_class[j+1].append(temp)
                else:

                    sp.plot_spectrogram(spectrogram=input.squeeze(dim=0), title="",
                                        output_filepath=output_dir+"/"+filename[0].split("/")[-1]+"_"+str(i).zfill(3)+"_"+"0_orig.png",
                                        sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title="original")

                    for j in range(num_classes):

                        sp.plot_spectrogram(spectrogram=input.squeeze(dim=0) * party_output.squeeze(dim=0)[j], title="",
                                            output_filepath=output_dir+"/"+filename[0].split("/")[-1]+"_"+str(i).zfill(3)+"_"+classes[j]+"_1_partyout.png",
                                            sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, show=False, ax_title=classes[j])
            #Denoised audio output
            if makeAudio and audio_gen:

                decompressed_net_out = t_decompr_f(input)

                spec_cmplx = spec_cmplx.squeeze(dim=0)

                decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)

                audio_spec = decompressed_net_out * spec_cmplx

                window = torch.hann_window(n_fft)

                audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)

                detected_spec_cmplx = spec_cmplx.squeeze(dim=0).transpose(0, 1)

                audio_out_denoised = torch.istft(audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)

                if concatenate:
                    if total_audio[0] is None:
                        total_audio[0] = audio_out_denoised
                    else:
                        total_audio[0] = torch.cat((total_audio[0], audio_out_denoised), 0)
                else:
                    total_audio[0] = torch.istft(audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)

                for j in range(num_classes):
                    party_out_channel = party_output_proc[0][j].unsqueeze(0).unsqueeze(0)
                    decompressed_net_out = t_decompr_f(party_out_channel)
                    spec_cmplx = spec_cmplx.squeeze(dim=0)
                    decompressed_net_out = decompressed_net_out.unsqueeze(dim=-1)
                    audio_spec = decompressed_net_out * spec_cmplx
                    audio_spec = audio_spec.squeeze(dim=0).transpose(0, 1)

                    window = torch.hann_window(n_fft)

                    audio_out_denoised = torch.istft(audio_spec, n_fft, hop_length=hop_length, onesided=True, center=True, window=window)

                    if total_audio[j + 1] is None:
                        total_audio[j + 1] = audio_out_denoised
                    else:
                        total_audio[j + 1] = torch.cat((total_audio[j + 1], audio_out_denoised), 0)

                scipy.io.wavfile.write(output_dir + "/" + filename[0].split("/")[-1].split(".")[0] + "_original.wav", sr, total_audio[0].numpy().T)

                for i, c in enumerate(classes):
                    scipy.io.wavfile.write(output_dir + "/" + filename[0].split("/")[-1].split(".")[0] + "_" + c + ".wav", sr, total_audio[i + 1].numpy().T)

    if sp is not None and os.path.isfile(input_file):
        hop = ceil(time_dim/2)
        temp1 = np.zeros([(len(long_class[0])-1)*hop+time_dim, freq_dim])
        temp2 = np.zeros([(len(long_class[0])-1)*hop+time_dim, freq_dim])

        for i, fram in enumerate(long_class[0]):
            to = fram.shape[0]
            if i % hop_factor == 0:
                temp1[i*hop: i*hop+to, :freq_dim] = fram[:to, :freq_dim]
            else:
                temp2[i*hop: i*hop+to, :freq_dim] = fram[:to, :freq_dim]

        temp1 = torch.from_numpy(temp1)
        temp2 = torch.from_numpy(temp2)
        original = torch.add(temp1, temp2)

        original = flip(original, dim=-1)
        img = spec2img(original.unsqueeze(0), cm=viridis_cm)
        img = torch.transpose(img, 0, 2)

        plt.subplot(num_classes+1, 1, 1)

        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = 200, 8
        plt.rcParams.update({'font.size': 3})
        plt.grid(b=None)
        plt.title("original", fontsize=6, pad=1.5)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, origin='lower')

        for j in range(num_classes):
            plt.subplot(num_classes+1, 1, j+2)

            temp1 = np.zeros([(len(long_class[0])-1)*hop+time_dim, freq_dim])
            temp2 = np.zeros([(len(long_class[0])-1)*hop+time_dim, freq_dim])

            for i, fram in enumerate(long_class[j+1]):
                to = fram.shape[0]
                if i % hop_factor == 0:
                    temp1[i*hop: i*hop+to, :256] = fram[:to, :freq_dim]
                else:
                    temp2[i*hop: i*hop+to, :256] = fram[:to, :freq_dim]

            temp1 = torch.from_numpy(temp1)
            temp2 = torch.from_numpy(temp2)
            temp = torch.add(temp1, temp2)
            temp = flip(temp, dim=-1)

            img = spec2img(temp.unsqueeze(0), cm=viridis_cm)
            img = torch.transpose(img, 0, 2)

            plt.rcParams['figure.figsize'] = 200, 8
            plt.rcParams.update({'font.size': 3})
            plt.title(classes[j], fontsize=6, pad=1.5)
            plt.tight_layout()
            plt.imshow(img, origin='lower')
            plt.axis('off')

        filename = input_file.split("/")[-1]
        plt.savefig(output_dir+"/"+filename+".png", dpi=600, pad_inches=0.0, bbox_inches=0.0)
        plt.close()

    log.debug("Finished proccessing")

    log.close()

