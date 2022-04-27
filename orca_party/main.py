#!/usr/bin/env python3
"""
Module: main.py
Authors: Christian Bergler, Manuel Schmitt
License: GNU General Public License v3.0
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 26.04.2022
"""

import os
import math
import pathlib
import argparse
import glob

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    OverlappedPreGeneratedDataset,
    OverlappedPreGeneratedDatasetCsvSplit,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict

from models.L2Loss import L2Loss
from models.unet_model import UNet

parser = argparse.ArgumentParser()

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

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir",
    type=str,
    default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--start_from_scratch",
    dest="start_from_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--jit_save",
    dest="jit_save",
    action="store_true",
    help="Save model via torch.jit save functionality.",
)

parser.add_argument(
    "--max_train_epochs",
    type=int,
    default=500,
    help="The number of epochs to train for the classifier."
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
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
    default=4,
    help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1",
    type=float,
    default=0.5,
    help="beta1 for the adam optimizer."
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set."
)

parser.add_argument(
    "--class_info",
    type=str,
    help="The path to the automatic generated class.info file, build by the generate_overlap_dataset.py.",
)

""" Input parameters """
ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""
def get_audio_files():
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to .bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files


def get_pickle_files(input_data, path):
    pickle_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))

        pickle_files = glob.glob(os.path.join(ARGS.data_dir, "**", "*.p"), recursive=True)
        pickle_files = map(lambda p: pathlib.Path(p), pickle_files)
        base = pathlib.Path(path)

        pickle_files = list(map(lambda p: str(p.relative_to(base)), pickle_files))

        log.info("Found {} audio files for training.".format(len(pickle_files)))

        if len(pickle_files) == 0:
            log.close()
            exit(1)

    return pickle_files

"""
Save the trained model and corresponding options either via torch.jit and/or torch.save.
"""
def save_model(unet, dataOpts, path, model, use_jit=False):
    unet = unet.cpu()
    unet_state_dict = unet.state_dict()
    save_dict = {
        "unetState": unet_state_dict,
        "dataOpts": dataOpts,
    }
    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    if use_jit:
        example = torch.rand(1, 1, 128, 256)
        extra_files = {}
        extra_files['dataOpts'] = dataOpts.__str__()
        model = torch.jit.trace(model, example)
        torch.jit.save(model, path, _extra_files=extra_files)
        log.debug("Model successfully saved via torch jit: " + str(path))
    else:
        torch.save(save_dict, path)
        log.debug("Model successfully saved via torch save: " + str(path))


"""
Main function to compute data preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":

    dataOpts = {}

    for arg, value in vars(ARGS).items():
        dataOpts[arg] = value

    ARGS.cuda = torch.cuda.is_available() and ARGS.cuda

    debug = ARGS.debug
    batch_size = ARGS.batch_size
    lr = ARGS.lr
    early_stopping_patience_epochs = ARGS.early_stopping_patience_epochs
    epochs_per_eval = ARGS.epochs_per_eval
    class_info = ARGS.class_info
    jit_save = ARGS.jit_save
    max_train_epochs = ARGS.max_train_epochs
    cuda = ARGS.cuda
    data_dir = ARGS.data_dir
    summary_dir = ARGS.summary_dir
    log_dir = ARGS.log_dir
    cache_dir = ARGS.cache_dir
    lr_decay_factor = ARGS.lr_decay_factor
    model_dir = ARGS.model_dir
    num_workers = ARGS.num_workers
    beta1 = ARGS.beta1
    lr_patience_epochs = ARGS.lr_patience_epochs
    checkpoint_dir = ARGS.checkpoint_dir
    start_from_scratch = ARGS.start_from_scratch

    log.info(f"Logging Level: {debug}")
    log.info(f"Directory of the Input Data: {data_dir}")
    log.info(f"Directory of all the Summaries: {summary_dir}")
    log.info(f"Directory of the Logging Output: {log_dir}")
    log.info(f"Directory of the Cached Output: {cache_dir}")
    log.info(f"Directory of the Checkpoint Output: {checkpoint_dir}")
    log.info(f"Directory of the Final Model (.pk): {model_dir}")
    log.info(f"Number of Workers during Data Loading: {num_workers}")
    log.info(f"Initial Learning Rate: {ARGS.lr}")
    log.info(f"Factor of Learning Rate Decay: {lr_decay_factor}")
    log.info(f"Number of Epochs after (divided by the --epochs_per_eval value): {lr_patience_epochs}")
    log.info(f"Number of Epochs for Evaluation on the Validation Set: {epochs_per_eval}")
    log.info(f"Early Criterion/Patience in Epochs (divided by the --epochs_per_eval value): {early_stopping_patience_epochs}")
    log.info(f"Batch Size: {batch_size}")
    log.info(f"GPU-Cuda support: {cuda}")
    log.info(f"Beta1 Value Optimizer: {beta1}")
    log.info(f"Saving the model with JIT save: {jit_save}")
    log.info(f"Number of Maximum Training Epochs: {max_train_epochs}")
    log.info(f"Class Info File including Number and Labels of each Class: {class_info}")
    log.info(f"Starts Model Training from Scratch or Existing Checkpoint: {start_from_scratch}")

    prefix = "ORCA-PARTY"

    device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

    lr *= batch_size

    patience_lr = math.ceil(lr_patience_epochs / epochs_per_eval)

    patience_lr = int(max(1, patience_lr))

    split_fracs = {"train": .7, "val": .15, "test": .15}

    input_data = OverlappedPreGeneratedDatasetCsvSplit(split_fracs, working_dir=data_dir, split_per_dir=True)

    pickle_files = get_pickle_files(input_data, data_dir)

    with open(class_info) as file:
        classes = [line.rstrip('\n') for line in file]

    log.info("Setting up model")

    log.info("Training ORCA-PARTY on "+str(len(classes))+" classes")

    log.info("Class Labels: " + str(classes))

    unet = UNet(n_channels=1, n_classes=len(classes), bilinear=False)

    log.debug("Model: " + str(unet))

    model = nn.Sequential(OrderedDict([("unet", unet)]))

    datasets = {
        split: OverlappedPreGeneratedDataset(
            file_names=input_data.load(split, pickle_files),
            working_dir=data_dir,
            dataset_name=split
        )
        for split in split_fracs.keys()
    }

    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False if split == "val" or split == "test" else True,
            pin_memory=True,
        )
        for split in split_fracs.keys()
    }

    trainer = Trainer(
        model=model,
        logger=log,
        prefix=prefix,
        checkpoint_dir=checkpoint_dir,
        summary_dir=summary_dir,
        n_summaries=1,
        start_scratch=start_from_scratch,
        classes=classes,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, 0.999)
    )

    metric_mode = "min"
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        patience=patience_lr,
        factor=lr_decay_factor,
        threshold=1e-3,
        threshold_mode="abs",
    )

    L2Loss = L2Loss(reduction="sum")

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=L2Loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=max_train_epochs,
        val_interval=epochs_per_eval,
        patience_early_stopping=early_stopping_patience_epochs,
        device=device,
        val_metric="loss"
    )

    path = os.path.join(model_dir, prefix+".pk")

    save_model(unet, dataOpts, path, model, use_jit=jit_save)

    log.close()
