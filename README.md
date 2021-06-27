# CMT-pytorch
Source code and generated samples for CMT (Chord Conditioned Melody Transformer) model introduced in 
["Chord Conditioned Melody Generation with Transformer Based Decoders"](https://ieeexplore.ieee.org/abstract/document/9376975)


## Requirements
- matplotlib >= 3.3.1
- numpy >= 1.19.1
- pretty_midi >= 0.2.9
- pytorch >= 1.0.0
- yaml >= 0.2.5

## File descriptions
  * `hparams.yaml` : specifies hyperparameters and paths to load data or save results.
  * `preprocess.py` : makes instance pkl files from two track midi files
  * `dataset.py` : loads preprocessed pkl data
  * `layers.py` : self attention block and relative multi-head attention layers
  * `model.py` : implementation of CMT
  * `loss.py` : defines loss functions
  * `trainer.py` : utilities for loading, training, and saving models 
  * `run.py` : main code to train CMT
  * `generated samples.zip` : zip file containing 15 generated samples 
  which are used for subjective listening test

## Preparing data
To train CMT, midi files containing melody and chords are necesary. 
Each midi file should have two instruments: 
the first instrument playing melody, 
and the second instrument playing all chordal notes of chords whenever chord changes.

Instance pkl files are made from two track midi files
by executing the following command line:
```bash 
$ python preprocess.py 
--root_dir [ROOT_DIR]
--midi_dir [MIDI_DIR]
--num_bars [NUMBER_OF_BARS]
--frame_per_bar [FRAME_PER_BAR]
--pitch_range [PITCH_RANGE]
```

  * Midi files should be located under `$ROOT_DIR/MIDI_DIR`
  * `NUMBER_OF_BARS`: number of bars to generate. Default is 8
  * `FRAME_PER_BAR`: number of unit notes in a bar. Default is 16 (16th note, time signature 4/4)
  * `PITCH_RANGE`: MIDI pitch range. Default is 48 (4 octaves)
  
To shift the pitch of melody and chords in 12 different keys, 
add argument `--shift` to the command line above.

## Training CMT
```bash 
$ python run.py 
--idx [EXPERIMENT_INDEX] 
--gpu_index [GPU_INDEX]
--ngpu [NUMBER_OF_GPU]
--optim_name [OPTIMIZER]
--restore_epoch [RESTORE_EPOCH]
--seed [RANDOM_SEED]
```

  * `EXPERIMENT_INDEX`: arbitrary index to distinguish different experiment settings
  * `GPU_INDEX`: index of GPU
  * `NUMBER_OF_GPU`: number of GPUs to use. If not specified, use only CPU.
  * `OPTIMIZER`: Optimizer to use. One of `sgd`, `adam`, `rmsprop`, default is `adam`
  * `RESTORE_EPOCH`: which checkpoint to restore when continuing an experiment
 
### 1st phase
Train the rhythm decoder (RD) with pitch varied rhythm data.
In `hparams.yaml`, set the data_io path to directory containing pkl files with 12 different keys.


### 2nd phase
Retain RD from the 1st phase and train pitch decoder (PD) with single key data.
In experiment config of `hparams.yaml`, specify the experiment index and epoch to load RD from (for example, idx 1, epoch 100).
Execute `run.py` with additional `--load_rhythm` argument.
