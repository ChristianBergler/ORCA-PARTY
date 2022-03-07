# ORCA-PARTY
ORCA-PARTY: An Automatic Killer Whale Sound Type Separation Toolkit Using Deep Learning

## General Description
ORCA-PARTY, is a deep sound type separation network designed for separation of overlapping (bioacoustic) killer whale (<em>Orcinus Orca</em>) vocalization types, not requiring any human-annotated overlapping ground truth data, irrespective of speaker, sound source location, and recording conditions, in order to enhance downstream sound type recognition. <br>ORCA-PARTY was trained exclusively on overlapping killer whale signals of 7 known vocalization types plus a "unknown" category (8 classes), resulting in a significant signal enhancement and improvement of subsequent vocalization type classification. To show and prove the transferability, robustness and generalization of ORCA-PARTY even more, deep sound type separation was also conducted for bird sounds (<em>Myiopsitta monachus</em>).<br><br>

## Reference
If ORCA-PARTY is used for your own research please cite the following publication: ORCA-PARTY: An Automatic Killer Whale Sound Type Separation Toolkit Using Deep Learning (tbd)

```
@inproceedings{Bergler-OP-2022,
  author={Christian Bergler, Manuel Schmitt, Maier Andreas, Rachael Xi Cheng, Volker Barth, Elmar Nöth},
  title={ORCA-PARTY: An Automatic Killer Whale Sound Type Separation Toolkit Using Deep Learning},
  year = {2022},
  month = {May}
  booktitle = {International Conference on Acoustics, Speech, and Signal Processing, Proceedings (ICASSP)},
  pages={tbd},
  doi={tbd},
  url={tbd}
}
```
## License
GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 (GNU GPLv3)

## General Information
Manuscript Title: <em>ORCA-PARTY: An Automatic Killer Whale Sound Type Separation Toolkit Using Deep Learning</em>. Within the <em>docs/audiovisual_data</em> folder several audio examples of various overlapping killer whale signals and corresponding spectral visualizations (spectrograms) are stored. All examples belong to the unseen <em>ORCA-PARTY Overlapping Dataset (OPOD)</em> test set mentioned in our manuscript.

## Python, Python Libraries, and Version
ORCA-PARTY is a deep learning algorithm which was implemented in Python (Version=3.8) (Operating System: Linux) together with the deep learning framework PyTorch (Version=1.8.1, TorchVision=0.9.1, TorchAudio=0.8.1). Moreover it requires the following Python libraries: Pillow, MatplotLib, Librosa, TensorboardX, Matplotlib, Soundfile, Scikit-image, Six, Resampy, Opencv-python (recent versions). ORCA-PARTY is currently compatible with Python 3.8 and PyTorch (Version=1.9.0, TorchVision=0.10.0, TorchAudio=0.9.0)

## Required Filename Structure for Training
In order to properly load and preprocess your data to train the network you need to prepare the filenames of your audio data clips to fit the following template/format:

Filename Template: LABEL-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav

1st-Element: LABEL = a placeholder for any kind of string which describes the label of the respective sample, e.g. call-N9, orca, echolocation, etc.

2nd-Element: ID = unique ID (natural number) to identify the audio clip

3rd-Element: YEAR = year of the tape when it has been recorded

4th-Element: TAPENAME = name of the recorded tape (has to be unique in order to do a proper data split into train, devel, test set by putting one tape only in only one of the three sets

5th-Element: STARTTIME = start time of the audio clip in milliseconds with respect to the original recording (natural number)

6th-Element: ENDTIME = end time of the audio clip in milliseconds with respect to the original recording(natural number)

Due to the fact that the underscore (_) symbol was chosen as a delimiter between the single filename elements please do not use this symbol within your filename except for separation.

Examples of valid filenames:

call-Orca-A12_929_2019_Rec-031-2018-10-19-06-59-59-ASWMUX231648_2949326_2949919

Label Name=call-Orca-A12, ID=929, Year=2019, Tapename=Rec-031-2018-10-19-06-59-59-ASWMUX231648, Starttime in ms=2949326, Starttime in ms=2949919

orca-vocalization_2381_2010_101BC_149817_150055.wav

Label Name=orca-vocalization, ID=2381, Year=2010, Tapename=101BC, Starttime in ms=149817, Starttime in ms=150055

In case the original annotation time stamps, tape information, year information, or any filename-specific info is not available, also artificial/fake names/timestamps can be chosen. It only needs to be ensured that the filenames follow the given structure.

## Generation of Overlapping Data Samples
Based on the given training material, the first step involves a machine-driven fully-automated overlapping ground truth data generation for training the model. For this purpose it is necessary to provide the input data (audio files as ".wav" format) within the following data structure.

├── Input Folder (see --input_dir option below)
│   ├── CategoryA
│   │   ├── categoryA-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── categoryA-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── (...)
│   ├── CategoryB
│   │   ├── categoryB-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── categoryB-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── (...)
│   ├── CategoryC
│   │   ├── categoryC-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── categoryC-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── (...)
│   ├── CategoryD
│   │   ├── categoryD-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── categoryD-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── (...)
│   ├── Garbage
│   │   ├── Garbage-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── Garbage-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav
│   │   ├── (...)

Each class-specific content has to be stored within a separate sub-folder within the acutal input directory path. The name of the sub-folders have to be coherent with the "LABEL" of all corresponding filenames below. The filename structure has to follow the descriptions within the previous chapter (see <em>Required Filename Structure for Training</em>). Independent of the training dataset, each corpus has to have an additional <em>Garbage</em> class, modelling all events (e.g. noise, unkown call patterns, etc.) which can not be assigned to a specific category.

Once the animal-specific dataset is ready following the above directory- and file-structure, the <em>generate_overlap_dataset.py</em> script can be executed. For a detailed description about each possible script option we refer to the usage/code in <em>generate_overlap_dataset.py</em> (usage: <em>generate_overlap_dataset.py -h</em>). This is just an example command in order to start the overlapping dataset generation procedure:

```generate_overlap_dataset.py --debug --input_dir path_to_input_dir --data_output_dir path_to_output_dir --target_path path_to_target_dir --denoiser path_to_denoiser_model_pickle --number_per_label number_of_overlapping_samples_per_label --log_dir path_to_log_dir```

In a first step, the module will set up the same directory structre as shown above, as provided via the <em>--data_output_dir</em> option and generates for every given audio file (.wav) the corresponding spectrogram, according to the chosen options (fft-size, step-size, sampling rate, & sequence length -- see <em>generate_overlap_dataset.py -h</em>). Each category-specific spectrogram will be stored as pickle file within the respective sub-directory of the selected <em>--data_ouptut_dir</em> option. The initial spectrogram generation provides also the functionality of signal denoising/enhancement, by using the <em>--denoiser</em> option, togehter with a previously trained ORCA-CLEAN model stored as pickle file (see ORCA-CLEAN https://github.com/ChristianBergler/ORCA-CLEAN). If the <em>--denoiser</em> option is not in use, the original noisy data samples will be used.

├── Output Folder (see --data_ouptut_dir option below)
│   ├── CategoryA
│   │   ├── categoryA-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav.p
│   │   ├── categoryA-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav.p
│   │   ├── (...)
│   ├── CategoryB
│   │   ├── categoryB-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav.p
│   │   ├── categoryB-XXX_ID_YEAR_TAPENAME_STARTTIME_ENDTIME.wav.
│   │   ├── (...)
(...)

Once the entire audio (.wav) pool has been converted into spectrograms, the final overlapping ground truth data will be generated. According to the selected number of overlapping samples per cateogry (see option <em>--number_per_label</em>), together with the respective data split (default: 70% training, 15% validation, 15% test), the final overlapping signals will be machine-generated for each combination. The combinatorial complexity can be calculated as follows: N = number of classes, K = 2 = overlapping signals, consist always two spectrograms ---> [N over K + (N-1) + (N-1)] x number_per_label. Every category-based combination, plus every class with itself except the garbage class, plus every class without any overlap except the garbage class. According to the ORCA-PARTY paper K=8, N=2 --> [8 over 2 + (8-1) + (8-1)] x 2,000 = [28 + 7 + 7] x 2,000 = 84,000 overlapping signals. IMPORTANT: the total amount of samples per category, has to be equal or larger than the chosen fraction-specific number of overlapping events per label (see option <em>--number_per_label</em>). If this is not the case all overlapping events involving this specific category will be skipped/ignored (e.g. 2,000 overlapping samples -> 70% in train = 1,400 samples, so each category-specific datapool has to be larger than this number).

The final overlapping samples will be stored as pickle files at the location defined by the <em>--target_path</em> option. Next to all category-specific overlapping data samples, an additional <em>classes.info</em> and <em>data_gen.opts</em> file are stored. Within the <em>classes.info</em> file all automatically identified categories/classes are listed, which is equal to the final number and naming of the respective segmentation output masks. Within the <em>data_gen.opts</em> file, all relevant and needed data options of the entire data generation process are stored.

Finally, also the data split files are stored within this directory (see <em>--target_path</em>), including train.csv, val.csv, test.csv, following the same structure as in ORCA-SPOT (see https://github.com/ChristianBergler/ORCA-SPOT) and ORCA-CLEAN (see https://github.com/ChristianBergler/ORCA-CLEAN), listing all the filenames stored within the respective data partition. 

## Network Training
Once the initial overlapping data pool generation is done, network training can be conducted. For a detailed description about each possible training option we refer to the usage/code in main.py (usage: <em>main.py -h</em>). This is just an example command in order to start network training:

```main.py --debug --class_info path_to_classes_info_file --lr 10e-4 --batch_size 8 --num_workers 6 --data_dir path_to_overlapping_data_folder --model_dir path_to_model_output --log_dir path_to_log_dir --checkpoint_dir path_to_checkpoint_dir --early_stopping_patience_epochs 20 --summary_dir path_to_summary_dir --max_train_epochs 150```

The input for the <em>--class_info</em> and <em>--data_dir</em> (including data split files) are the result of the previously illustrated data generation approach.

## Network Testing and Evaluation
During training ORCA-PARtY will be verified on an independent validation set. In addition ORCA-PARTY will be automatically evaluated on the test set. As evaluation criteria the validation loss is utilized. All documented results/images and the entire training process could be reviewed via tensorboard and the automatic generated summary folder:

```tensorboard --logdir /directory_to_model/summaries/```

There exist also the possibility to evaluate your model on either a folder including unseen audio recordings (.wav) of different length, or a stand-alone audio tape. The prediction script (<em>predict.py</em>) reads, denoises (if provided with a trained ORCA-CLEAN denoising model - see https://github.com/ChristianBergler/ORCA-CLEAN), and separate incoming audio input (.wav) into the respective category-specific segmentation outputs. ORCA-PARTY and the <em>predict.py</em> provides also the opportunity to visualize (<em>--visualize</em>) input and output spectrograms, next to audio reconstruction (<em>--audio</em>) after source separation. For a detailed description about each possible option we refer to the usage/code in predict.py (usage: <em>predict.py -h</em>). This is just an example command in order to start the prediction:

Example Command:

```predict.py --debug --model_path path_to_orca_party_model_pickle --output_dir path_to_output_dir --denoiser path_to_denoiser_model_pickle --input_file path_to_folder_with_audio_files_OR_entire_recording --data_opts_path path_to_data_gen_opts_file --class_info path_to_classes_info_file --sequence_len 1.28 --visualize --audio```