"""
Module: overlap_signal_generator.py
Authors: Christian Bergler, Manuel Schmitt
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 03.03.2022
"""

import sys
import pickle
import random
import pathlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils.logging import *

from models.unet_model import UNet

from collections import OrderedDict

from data.audiodataset import Dataset

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


"""
Check whether the required garbage class is within the pool of classes
"""
def check_partitioning(split_class_fname, logger):
    for partition_name in split_class_fname:
        partition = split_class_fname[partition_name]
        classes = list(partition.keys())
        is_in = "garbage" in (cl_el.lower() for cl_el in classes)
        if is_in:
            logger.info(
                "Partition: " + partition_name + " - Garbage class is present within the class labels. Valid data pre-processing and signal generation of overlapping signals is ensured!")
        else:
            logger.error(
                "Missing Garbage class in Partition: " + partition_name + " - This does not ensure a correct data pre-processing and signal generation of overlapping signals. Insert a Garbage class!")
            log.close()
            sys.exit()

"""
Check and return the needed "garbage" class label
"""
def get_garbage_class_label(classes):
    for cl_el in classes:
        if cl_el.lower() == "garbage":
            return cl_el
    return None


"""
Denoise given input files using an existing and trained version of ORCA-CLEAN and save the denoised version (tensor) within a pickle file for any kind of post-processing procedures
"""
def build_dataset(
        cuda,
        logger,
        input_dir,
        data_output_dir,
        sr=44100,
        fmin=500,
        fmax=10000,
        n_fft=4096,
        denoiser=None,
        noise_files=[],
        hop_length=441,
        n_freq_bins=256,
        ref_level_db=20,
        min_level_db=-100,
        sequence_len=1280,
        augmentation=False,
        min_max_normalize=False,
        freq_compression="linear"
):

    if denoiser is not None:
        denoiser_model_dict = torch.load(denoiser)

        denoiser_model = UNet(1, 1, bilinear=False)
        denoiser_model.load_state_dict(denoiser_model_dict["unet"])  # unetState = in final version
        denoiser_model = nn.Sequential(OrderedDict([("denoiser", denoiser_model)]))
        dataOpts = denoiser_model_dict["dataOpts"]

        if torch.cuda.is_available() and cuda:
            denoiser_model = denoiser_model.cuda()

        denoiser_model.eval()

        logger.info("Loading pre-trained denoising model!")
        logger.info(denoiser)

        logger.info("Loading data options from the current denoising model!")
        logger.info("Denoiser: " + str(dataOpts))

    else:
        logger.info("No pre-trained denoising model! Running with noisy spectrograms!")
        denoiser_model = None

    audio_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])

    logger.debug("Processed audio files: " + str(audio_files))

    sequence_len = int(float(sequence_len) / 1000 * sr / hop_length)

    dataset = Dataset(
        sr=sr,
        cuda=cuda,
        f_min=fmin,
        f_max=fmax,
        n_fft=n_fft,
        orca_detection=True,
        seq_len=sequence_len,
        hop_length=hop_length,
        perc_of_max_signal=1.0,
        file_names=audio_files,
        n_freq_bins=n_freq_bins,
        noise_files=noise_files,
        min_level_db=min_level_db,
        ref_level_db=ref_level_db,
        augmentation=augmentation,
        denoiser_model=denoiser_model,
        freq_compression=freq_compression,
        min_max_normalize=min_max_normalize
    )

    if not os.path.isdir(data_output_dir):
        os.makedirs(data_output_dir)

    file_count = 0

    for sample, label in dataset:
        file_count += 1
        fname = label["file_name"].split("/")[-1].replace("-", "_")
        log.debug("Number of files=" + str(file_count) + ", filename=" + str(fname) + ", tensor shape=" + str(sample.shape))

        with open(data_output_dir + "/" + fname + ".p", "wb") as f:
            pickle.dump(sample, f)


"""
Generate training, validation, and test split for each category-specific type according to the given split ratios
"""
def create_data_split(data_output_dir, target_path, logger):
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    if visualization_dir is not None and not os.path.isdir(visualization_dir):
        os.makedirs(visualization_dir)

    class_fname = {}

    classes = os.listdir(data_output_dir)
    for class_element in classes:
        if class_element not in class_fname.keys():
            class_fname[class_element] = []

        for fname in os.listdir(data_output_dir + "/" + class_element):
            class_fname[class_element].append(fname)

    split_fracs = {"train": .7, "val": .15, "test": .15}

    split_class_fname = {}

    split_class_fname["train"] = {}
    split_class_fname["val"] = {}
    split_class_fname["test"] = {}

    for call in class_fname.keys():
        fnames = class_fname[call]

        random_sample = random.sample(fnames, int(len(fnames) * float(split_fracs["train"])))
        split_class_fname["train"][call] = random_sample
        fnames = [x for x in fnames if x not in random_sample]

        random_sample = random.sample(fnames, int(len(fnames) * 0.5))
        split_class_fname["val"][call] = random_sample
        fnames = [x for x in fnames if x not in random_sample]

        split_class_fname["test"][call] = fnames

        logger.info(
            "Label: " + call + ", Num Train: " + str(len(split_class_fname["train"][call])) + ", Num Val: " + str(
                len(split_class_fname["val"][call])) + ", Num Test: " + str(len(split_class_fname["test"][call])))

    logger.info("Generation data split done!")

    logger.info(split_class_fname)

    check_partitioning(split_class_fname=split_class_fname, logger=logger)

    return split_class_fname, split_fracs


"""
Generate overlapping combinations of class-specific vocalization types according to the given training, validation, and test split

Combinatorial Variety: N-Classes including the needed garbage class, amount of vocalization classes = (N-1), (N-1) * number per label (self-overlap),
(N-1) * number per label (non-overlap), binomial coefficient N=(N over K=2) * number of label (every class wise combination ignoring the ordering)

Example (same as in Paper):
-------------------------------
-Number of Classes: N=8 (7 Vocalizations Classes + 1 Garbage Class)
-Number of spectral pairs per Class: S=2000, K=2

Calculate total number of overlapping samples: 
--------------------------------------------------
Binomial Coefficient ( (N over K) * S + (N-1 * S) (=non overlapping signals, except garbage class) + (N-1 * S) (=self overlap, except garbage class)

(8 over 2) * 2000 + (7 * 2000) + (7 * 2000) = 84,000 (=Final Number of Samples) ---> See also ORCA-PARTY Paper
"""
def generate_overlapping_data(datasplit, split_fracs, data_output_dir, target_path, number_per_label, logger, dataOpts, spec_time_dim, spec_freq_dim, visualization_dir=None, offset_max=64):

    all_classes = set()
    num_smpls_partition = {"train": 0, "val": 0, "test": 0}

    for frac in datasplit.keys():
        total_frac_file_ctr = 0
        frac_fname_list = []
        classes = list(datasplit[frac].keys())
        all_classes.update(classes)

        if "garbage" in (cl_el.lower() for cl_el in classes):
            gb_lbl = get_garbage_class_label(classes)
            classes.append(classes.pop(classes.index(gb_lbl)))

        frac_num_per_label = int(number_per_label * split_fracs[frac])
        if frac_num_per_label == 0:
            frac_num_per_label = 1

        for i in range(len(classes) - 1):

            #if the number of existing category-specific files is larger then the amount of randomly generated overlapping signals process the data, cause each sample is just used once
            if len(datasplit[frac][classes[i]]) >= frac_num_per_label:

                cat_frac_file_ctr = 0

                sample_list_first_part = random.sample(datasplit[frac][classes[i]], frac_num_per_label)

                logger.info("--------------------------")
                logger.info("Partition: " + str(frac) + ", Label: " + str(
                    classes[i]) + ", Number of Files for Non-Overlap Generation: " + str(
                    len(sample_list_first_part)) + " --- START")
                logger.info("--------------------------")

                for count, l1 in enumerate(sample_list_first_part):

                    fn_wo_ext = l1.split(".")[0]

                    with open(data_output_dir + "/" + classes[i] + "/" + l1, "rb") as f:
                        f1 = pickle.load(f)

                    f1 = f1.detach().numpy().squeeze(0)

                    offset = random.randint(0, offset_max)

                    mixed = np.zeros([offset + spec_time_dim, spec_freq_dim])

                    name = fn_wo_ext + "__NOOVERLAP"
                    if count >= frac_num_per_label / 2:
                        mixed[:f1.shape[0], :f1.shape[1]] = f1
                    else:
                        name = "NOOVERLAP__" + fn_wo_ext
                        mixed[offset:spec_time_dim + offset, :f1.shape[1]] += f1

                    frac_fname_list.append(name + ".p")

                    mixed -= mixed.min()
                    if not mixed.max().item() == 0.0:
                        mixed /= mixed.max()

                    random_pick = random.randint(0, mixed.shape[0] - spec_time_dim)

                    overlapped_signal = np.zeros([spec_time_dim, spec_freq_dim])
                    overlapped_signal[:spec_time_dim, :spec_freq_dim] = mixed[random_pick:spec_time_dim + random_pick, :spec_freq_dim]

                    f1_gt = np.zeros([spec_time_dim, spec_freq_dim])

                    # make sure that the sequence changes, so that not always the vocalization-specific sound types is the first element of the spectral pair
                    if count >= frac_num_per_label / 2:
                        f1_gt[0:spec_time_dim - random_pick, :spec_freq_dim] = f1[random_pick:spec_time_dim, :spec_freq_dim]
                    else:
                        f1_gt[(offset - random_pick):spec_time_dim, :spec_freq_dim] = f1[0:(spec_time_dim - (offset - random_pick)), :spec_freq_dim]

                    f2_gt = np.zeros([spec_time_dim, spec_freq_dim])
                    f2 = f2_gt

                    if visualization_dir is not None:
                        save_imgs(visualization_dir + "/" + name, overlapped_signal, f1_gt, f2_gt, f1, f2)

                    with open(target_path + "/" + name + ".p", "wb") as f:
                        pickle.dump((overlapped_signal, f1_gt, f2_gt, offset, random_pick, l1, "self-zero"), f)

                    logger.info("Generated Data Sample=" + target_path + "/" + name)

                    cat_frac_file_ctr += 1
                    total_frac_file_ctr += 1

            logger.info("--------------------------")
            logger.info("Partition: " + str(frac) + ", Label: " + str(classes[i]) + ", Number of category-specific Files (=" + str(cat_frac_file_ctr) + "), Number of cummulated generated category-specific Files (=" +
                        str(total_frac_file_ctr) + ") for Non-Overlap Generation --- DONE")
            logger.info("--------------------------")

            for j in range(i, len(classes)):

                # if the number of existing category-specific files is larger then the amount of randomly generated overlapping signals process the data, cause each sample is just used once
                if (len(datasplit[frac][classes[i]]) >= frac_num_per_label) and (len(datasplit[frac][classes[j]]) >= frac_num_per_label):

                    cat_frac_file_ctr = 0

                    sample_list_first_part = random.sample(datasplit[frac][classes[i]], frac_num_per_label)
                    sample_list_second_part = random.sample(datasplit[frac][classes[j]], frac_num_per_label)

                    logger.info("--------------------------")
                    logger.info("Partition: " + str(frac) + ", Label First Class: " + str(
                        classes[i]) + ", Number of Files for Overlap Generation - First Class: " + str(
                        len(sample_list_first_part)) + ", Label Second Class: " + str(
                        classes[j]) + ", Number of Files for Overlap Generation - Second Class: " + str(
                        len(sample_list_second_part)) + " --- START")
                    logger.info("--------------------------")

                    for count, (l1, l2) in enumerate(zip(sample_list_first_part, sample_list_second_part)):

                        fn1_wo_ext = l1.split(".")[0]
                        fn2_wo_ext = l2.split(".")[0]

                        with open(data_output_dir + "/" + classes[i] + "/" + l1, "rb") as f:
                            f1 = pickle.load(f)

                        with open(data_output_dir + "/" + classes[j] + "/" + l2, "rb") as f:
                            f2 = pickle.load(f)

                        f1 = f1.detach().numpy().squeeze(0)
                        f2 = f2.detach().numpy().squeeze(0)

                        name = fn1_wo_ext + "__" + fn2_wo_ext

                        # make sure that the sequence changes, so that not always one of the two vocalization-specific sound types is the first element of the spectral pair
                        if count >= frac_num_per_label / 2:
                            temp = f1
                            f1 = f2
                            f2 = temp
                            temp = l1
                            l1 = l2
                            l2 = temp
                            name = fn2_wo_ext + "__" + fn1_wo_ext

                        frac_fname_list.append(name + ".p")

                        offset = random.randint(0, offset_max)

                        mixed = np.zeros([offset + spec_time_dim, spec_freq_dim])
                        mixed[:f1.shape[0], :f1.shape[1]] = f1
                        mixed[offset:spec_time_dim + offset, :f2.shape[1]] += f2
                        mixed[offset:spec_time_dim + offset, :f2.shape[1]] += f2

                        mixed -= mixed.min()
                        if not mixed.max().item() == 0.0:
                            mixed /= mixed.max()

                        random_pick = random.randint(0, mixed.shape[0] - spec_time_dim)

                        overlapped_signal = np.zeros([spec_time_dim, spec_freq_dim])
                        overlapped_signal[:spec_time_dim, :spec_freq_dim] = mixed[random_pick:spec_time_dim + random_pick, :spec_freq_dim]

                        f1_gt = np.zeros([spec_time_dim, spec_freq_dim])
                        f2_gt = np.zeros([spec_time_dim, spec_freq_dim])
                        if i != j:
                            f1_gt[0:spec_time_dim - random_pick, :spec_freq_dim] = f1[random_pick:spec_time_dim, :spec_freq_dim]
                            f2_gt[(offset - random_pick):spec_time_dim, :spec_freq_dim] = f2[0:(spec_time_dim - (offset - random_pick)), :spec_freq_dim]
                        else:
                            f1_gt = overlapped_signal

                        if visualization_dir is not None:
                            save_imgs(visualization_dir + "/" + name, overlapped_signal, f1_gt, f2_gt, f1, f2)

                        with open(target_path + "/" + name + ".p", "wb") as f:
                            pickle.dump((overlapped_signal, f1_gt, f2_gt, offset, random_pick, l1, l2), f)

                        logger.info("Generated Data Sample=" + target_path + "/" + name)

                        cat_frac_file_ctr += 1
                        total_frac_file_ctr += 1

                logger.info("--------------------------")
                logger.info("Partition: " + str(frac) + ", Label: " + str(classes[i]) + ", Number of category-specific Files (=" + str(
                    cat_frac_file_ctr) + "), Number of cummulated generated category-specific Files (=" + str(total_frac_file_ctr) + ") for Non-Overlap Generation --- DONE")
                logger.info("--------------------------")

        random.shuffle(frac_fname_list)
        with open(target_path + "/" + frac, "w") as f:
            f.write(frac + ".csv")

        with open(target_path + "/" + frac + ".csv", "w") as f:
            for fname in frac_fname_list:
                f.write(fname + "\n")

        with open(target_path + "/" + "classes.info", "w") as f:
            for class_element in all_classes:
                f.write(class_element + "\n")

        with open(target_path + "/" + "data_gen.opts", "w") as f:
            for data_key in dataOpts:
                data_val = dataOpts[data_key]
                f.write(str(data_key) + "=" + str(data_val) + "\n")

        logger.info("Final " + frac + ".csv with the respective number of overlapping signals has been written to: " + str(target_path + "/" + frac) + "\n")

        num_smpls_partition[frac] = total_frac_file_ctr

    logger.info("Number of generated overlapping train signals: " + str(num_smpls_partition["train"]))
    logger.info("Number of generated overlapping validation signals: " + str(num_smpls_partition["val"]))
    logger.info("Number of generated overlapping test signals: " + str(num_smpls_partition["test"]))
    logger.info("Number of generated overlapping signals across all partitions: " + str(sum(num_smpls_partition.values())))


"""
Store results of the overlapping signal generation procedure --- (1) overlapped signal, (2) ground truth first part, (3) ground truth second part, (4) original image first part, (5) original image second part
"""
def save_imgs(filepath, img_1, img_2, img_3, img_4, img_5):
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = 12, 7
    plt.grid(b=None)
    plt.tight_layout()

    plt.subplot(2, 3, 1)
    plt.imshow(img_1.transpose(), origin='lower')
    plt.subplot(2, 3, 2)
    plt.imshow(img_2.transpose(), origin='lower')
    plt.subplot(2, 3, 3)
    plt.imshow(img_3.transpose(), origin='lower')
    plt.subplot(2, 3, 4)
    plt.imshow(img_4.transpose(), origin='lower')
    plt.subplot(2, 3, 5)
    plt.imshow(img_5.transpose(), origin='lower')

    plt.savefig(filepath, dpi=100, pad_inches=0.0, bbox_inches=0.0)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Log additional training and model information.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory where all label-based audio files are stored within sub-directories, named as the corresponding class names",
    )

    parser.add_argument(
        "--data_output_dir",
        type=str,
        default=None,
        help="Directory where the pickle files (generated spectrograms) for each data sample are saved",
    )

    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        help="Directory where the final machine-generated overlapped signals are saved as pickle",
    )

    parser.add_argument(
        "--visualization_dir",
        type=str,
        default=None,
        help="Directory where the final overlapped spectrograms are saved as images for visual inspection",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="The directory to store the logs."
    )

    parser.add_argument(
        "--noise_dir",
        type=str,
        default=None,
        help="The directory including noise files for noise augmentation."
    )

    parser.add_argument(
        "--denoiser",
        type=str,
        default=None,
        help="Path to the denoising model as an additional option for data enhancemnt (e.g. ORCA-CLEAN.pk)",
    )

    parser.add_argument(
        "--number_per_label",
        type=int,
        default=2000,
        help="Number of machine-generated overlapping signals for each label"
    )

    parser.add_argument(
        "--no_cuda",
        default=False,
        help="Do not use cuda to apply the denoising model",
    )

    parser.add_argument(
        "--freq_compression",
        type=str,
        default="linear",
        help="Frequency compression to reduce GPU memory usage. Options: `'linear'` (default), '`mel`'",
    )

    parser.add_argument(
        "--sequence_len",
        type=int,
        default=1280,
        help="Sequence length in ms."
    )

    parser.add_argument(
        "--n_freq_bins",
        type=int,
        default=256,
        help="Number of frequency bins after compression.",
    )

    parser.add_argument(
        "--n_fft",
        type=int,
        default=4096,
        help="FFT size.")

    parser.add_argument(
        "--hop_length",
        type=int,
        default=441,
        help="FFT hop length.")

    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sampling Rate.")

    parser.add_argument(
        "--fmin",
        type=int,
        default=500,
        help="F-Min frequency represented in the spectrogram.")

    parser.add_argument(
        "--fmax",
        type=int,
        default=10000,
        help="F-Max frequency represented in the spectrogram.")

    parser.add_argument(
        "--minDB",
        type=int,
        default=-100,
        help="Minimum dB value for 0/1-dB-normalization (can not be used together with --min_max_norm).")

    parser.add_argument(
        "--refDB",
        type=int,
        default=20,
        help="Reference dB value for 0/1-dB-normalization (can not be used together with --min_max_norm).")

    parser.add_argument(
        "--min_max_norm",
        dest="min_max_norm",
        action="store_true",
        help="activates min-max normalization instead of default 0/1-dB-normalization.",
    )

    parser.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        help="Whether to augment the input data. Validation and test data will not be augmented.",
    )

    parser.add_argument(
        "--only_overlap",
        dest="only_overlap",
        action="store_true",
        help="Initial building of stand-alone spectrograms for subsequent overlap generation is not performed.",
    )

    ARGS = parser.parse_args()

    ARGS.cuda = torch.cuda.is_available() and ARGS.cuda

    ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

    log = Logger("OVERLAP_SIGNAL_GENERATOR", ARGS.debug, ARGS.log_dir)

    dataOpts = {}

    for arg, value in vars(ARGS).items():
        dataOpts[arg] = value

    input_dir = ARGS.input_dir
    data_output_dir = ARGS.data_output_dir
    target_path = ARGS.target_path
    visualization_dir = ARGS.visualization_dir
    log_dir = ARGS.log_dir
    denoiser = ARGS.denoiser
    num_per_label = ARGS.number_per_label
    cuda = ARGS.no_cuda
    noise_dir = ARGS.noise_dir
    freq_compression = ARGS.freq_compression
    sequence_len = ARGS.sequence_len
    n_freq_bins = ARGS.n_freq_bins
    n_fft = ARGS.n_fft
    hop_length = ARGS.hop_length
    sr = ARGS.sr
    fmin = ARGS.fmin
    fmax = ARGS.fmax
    minDB = ARGS.minDB
    refDB = ARGS.refDB
    min_max_norm = ARGS.min_max_norm
    augmentation = ARGS.augmentation
    only_overlap = ARGS.only_overlap

    log.info(f"Input Directory: {ARGS.input_dir}")
    log.info(f"Directory where the pickle files (generated spectrograms) for each data sample are saved: {ARGS.data_output_dir}")
    log.info(f"Directory where the final machine-generated overlapped signals are saved as pickle: {ARGS.target_path}")
    log.info(f"Directory where the final overlapped spectrograms are saved as images for visual inspection: {ARGS.visualization_dir}")
    log.info(f"Directory of the Logging Output: {ARGS.log_dir}")
    log.info(f"Directory including noise files for noise augmentation: {ARGS.noise_dir}")
    log.info(f"Path to the denoising model (e.g. ORCA-CLEAN.pk): {ARGS.denoiser}")
    log.info(f"Number of machine-generated overlapping signals for each label: {ARGS.number_per_label}")
    log.info(f"Do not use cuda to apply the denoising model: {ARGS.no_cuda}")
    log.info(f"Directory with noise files for additional noise augmentation: {ARGS.noise_dir}")
    log.info(f"Frequency compression: {ARGS.freq_compression}")
    log.info(f"Spectrogram sequence length: {ARGS.sequence_len}")
    log.info(f"Number of frequency bins: {ARGS.n_freq_bins}")
    log.info(f"Number of FFT window in samples: {ARGS.n_fft}")
    log.info(f"Number of HOP length in samples: {ARGS.hop_length}")
    log.info(f"Sampling Rate: {ARGS.sr}")
    log.info(f"Minimum frequency represented within spectrogram: {ARGS.fmin}")
    log.info(f"Maximum frequency represented within spectrogram: {ARGS.fmax}")
    log.info(f"Minimum dB-value for normalization: {ARGS.minDB}")
    log.info(f"Reference dB-value for normalization: {ARGS.refDB}")
    log.info(f"Min-Max-Normalization activated: {ARGS.min_max_norm}")
    log.info(f"Spectrogram augmentation: {ARGS.augmentation}")
    log.info(f"Initial building of stand-alone spectrograms for subsequent overlap generation is not performed: {ARGS.only_overlap}")

    """
    noise files for augmentation
    """
    if noise_dir is not None:
        noise_files = [str(p) for p in pathlib.Path(noise_dir).glob("*.wav")]
    else:
        noise_files = []

    sub_dirs = next(os.walk(input_dir))[1]

    spec_time_dim = int(float(sequence_len) / 1000 * sr / hop_length)
    spec_freq_dim = n_freq_bins

    """
    build dataset of single stand-alone and category-specifc spectrograms
    """
    if not only_overlap:
        for sub_dir in sub_dirs:
            sub_input_dir = input_dir + "/" + sub_dir
            log.info("Current data in process: " + sub_input_dir)
            data_output_dir_class_specific = data_output_dir + "/" + sub_dir
            log.info("Write denoised data to: " + data_output_dir_class_specific)
            if not os.path.isdir(data_output_dir_class_specific):
                os.makedirs(data_output_dir_class_specific)

            build_dataset(input_dir=sub_input_dir, data_output_dir=data_output_dir_class_specific, denoiser=denoiser, logger=log, cuda=cuda,
                          sr=sr, fmin=fmin, fmax=fmax, n_fft=n_fft, noise_files=noise_files, hop_length=hop_length, n_freq_bins=n_freq_bins,
                          ref_level_db=refDB, min_level_db=minDB, sequence_len=sequence_len, min_max_normalize=min_max_norm,
                          freq_compression=freq_compression, augmentation=augmentation)

    """
    generating data split 
    """
    datasplit, split_frac = create_data_split(data_output_dir=data_output_dir, target_path=target_path, logger=log)

    """
    generating spectral overlapping pairs of spectrograms according to the given (labeled) input data, save & store pickle, as well as data split .csv files
    """
    generate_overlapping_data(datasplit=datasplit, split_fracs=split_frac, data_output_dir=data_output_dir, target_path=target_path, number_per_label=num_per_label, spec_time_dim=spec_time_dim, spec_freq_dim=spec_freq_dim, logger=log, dataOpts=dataOpts, visualization_dir=visualization_dir, offset_max=64)

    log.close()
