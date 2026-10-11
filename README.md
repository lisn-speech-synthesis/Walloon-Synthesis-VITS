# Walloon Speech Synthesis with VITS

Code for the first text-to-speech system for Walloon, an under-resourced Romance language of Belgium, developed at LISN (CNRS / Université Paris-Saclay).

**Paper:** José Felipe Espinosa Orjuela, Philippe Boula de Mareüil, Marc Evrard. *Speech synthesis for Walloon, an under-resourced minority language.* 13th Speech Synthesis Workshop (SSW 2025), pp. 189–195. [doi:10.21437/SSW.2025-29](https://doi.org/10.21437/SSW.2025-29)

## What we did

- Trained VITS on Walloon recordings from two speakers, in a low-data (18 minutes) and a full-data (156 minutes) condition.
- Compared grapheme vs. phoneme input, and training from scratch vs. fine-tuning a French-pretrained model.
- Evaluated with objective speech metrics and a perceptual test with 23 Walloon speakers. Best systems (fine-tuned from French, grapheme input): **4.10/5 MOS** for both voices, against 4.48–4.55 for natural speech.

## Try the models

Interactive demos on Hugging Face Spaces:

- [Walloon, phoneme input](https://huggingface.co/spaces/Pipe1213/VITS_Walloon_Phonemes)
- [Walloon, grapheme input](https://huggingface.co/spaces/Pipe1213/VITS_Walloon_Graphemes)
- [French base model](https://huggingface.co/spaces/Pipe1213/VITS_French)

Analysis notebooks (bootstrap confidence intervals, speech and silence measurements): [Walloon-Synthesis-Results](https://github.com/lisn-speech-synthesis/Walloon-Synthesis-Results).

## Citation

```bibtex
@inproceedings{orjuela25_ssw,
  title     = {Speech synthesis for Walloon, an under-resourced minority language},
  author    = {Espinosa Orjuela, Jos{\'e} Felipe and Boula de Mare{\"u}il, Philippe and Evrard, Marc},
  booktitle = {13th ISCA Speech Synthesis Workshop (SSW 2025)},
  pages     = {189--195},
  year      = {2025},
  doi       = {10.21437/SSW.2025-29}
}
```

---

## Training VITS on your own data

This repository is based on the original [VITS implementation](https://github.com/jaywalnut310/vits) by Jaehyeon Kim et al. The steps below explain how to train it from scratch on a custom dataset and run inference.


### 1. Clone repository and dependencies
Clone this repository onto your system:

``git clone https://github.com/lisn-speech-synthesis/Walloon-Synthesis-VITS.git``

### 2. Enviroment setup

Create a new environment with Python version 3.6, then install all dependencies from the `requirements.txt` file. Note that some dependencies are available via `conda install`, while others can only be installed using `pip`. Therefore, I recommend installing them one by one.

Additionally, it is advised to install the appropriate CUDA driver for your system. You can find the suitable driver at the following link: https://pytorch.org/get-started/locally/

### 3. Monotonic Alignment Search
Build Monotonic Alignment Search module:

```sh
# Cython-version Monotonoic Alignment Search

cd monotonic_align
python setup.py build_ext --inplace

```

### 4. Preprocess Dataset
Download datasets:

1. Download and extract the LJ Speech dataset, then rename or create a symbolic link to the dataset folder: 
    `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
   
2. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a symbolic link to the dataset folder: 
    `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
    
You can also use your own dataset, but it must have the same format as the ones mentioned above. Below, I will provide more details on how to prepare your dataset, along with some tools.

If your dataset is already organized in the required format (LJ Speech for single speaker and VCTK for multi-speaker), use the following commands to preprocess the text files containing the different subsets (train, validation, test).

```sh
# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.

# LJ Speech
python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt

# VCTK
python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```
### 5. Symbols and Clearner files

In the `text` folder, there are two important files: `symbols.py` and `cleaners.py`.

- **`symbols.py`** contains all the alphabet characters, numbers, and symbols used in the dataset. Ensure that all characters present in your dataset are included in this file to avoid errors. Additionally, include any potential characters you plan to use during inference, as this cannot be changed after training. Failing to do so will require you to preprocess the input text before inference.

- **`cleaners.py`** provides default text preprocessing routines that are applied to the training, validation, and test data before training, as well as during inference. Modify these routines according to your specific requirements.

### 6. Configuration file

In the `configs` folder, there are example configuration files. Review them and modify anything necessary to adapt to your training setup.

Some of the most important parameters you should change include:

- **Training and validation file paths**
- **Text cleaners**
- **Number of speakers** (speaker IDs start counting from zero, but the number of speakers is counted normally. For example, with 2 speakers, the IDs are 0 and 1, and the number of speakers is 2)
- **Number of epochs**
- **Batch size** (adjust this according to your hardware)

### 7. Training
Run the following command depending on the type of training to be performed:

```sh
# LJ Speech (single speaker)
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK (multi-speaker)
python train_ms.py -c configs/vctk_base.json -m vctk_base
```
### 8. Inference

Use the `inference.ipynb` file from the original repository, or the [multi-speaker inference](inference_example_ms.py) file provided, to test inference. To do this, you need to specify the following in the file:

- The model path
- The configuration file path
- The folder where the generated audio files will be stored
- The path to the trained model (after training, this is stored in the `logs` folder)

## Dataset Formatting

To utilize the VITS configuration, the dataset must be formatted in a specific way. The audio files should have a sampling frequency of 22 kHz and be in WAV mono format. Additionally, audio files need to be segmented by phrases for two main reasons: longer audio files can reduce alignment quality, and the model automatically excludes files that are too long. To accommodate longer files, you can modify `train.py` at line 70 to increase the threshold for what constitutes a "long" file.

### Transcripts Formatting

Transcripts should be organized into three text files (`train`, `test`, `validation`), each in TXT format. Each line in these files should start with the audio file path, followed by the speaker ID (for multi-speaker configuration), and the corresponding transcription.

#### For Single-Speaker:
`
DUMMY/path_to_your_audio_folder/audio_001.wav|transcription_audio_001 
DUMMY/path_to_your_audio_folder/audio_999.wav|transcription_audio_999
`
#### For Multi-Speaker:
`
DUMMY/path_to_your_audio_folder/audio_001.wav|speaker_id|transcription_audio_001 
DUMMY/path_to_your_audio_folder/audio_999.wav|speaker_id|transcription_audio_999
`

Examples of how the alignment file should look can be found in the `filelists` folder of this repository.

#### Remarks:

- Each audio file must correspond to a transcription, so ensure you split the audio correctly beforehand.
- All audio files should have a sample rate of **22050 Hz** for optimal results. If any audio is in stereo, the system will not work, so you must convert them to mono.
- Speaker IDs start from 0, so keep this in mind when assigning the IDs.

## Contributing
Contributions are welcome. Please open an issue to discuss your ideas.
