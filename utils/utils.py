import os
import datetime
import numpy as np
import pretty_midi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import logger
from tensorboardX import SummaryWriter


def get_optimizer(params, lr, config, name='adam'):
    name = name.lower()
    if name == 'sgd':
        optimizer = optim.sgd(params, lr=lr, **config[name])
    elif name == 'adam':
        optimizer = optim.Adam(params, lr=lr, **config[name])
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, **config[name])
    else:
        raise RuntimeError("%s is not available." % name)

    return optimizer


def make_save_dir(save_path, config):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'model'), exist_ok=True)
    config.save(os.path.join(save_path, "hparams.yaml"))


def get_tfwriter(asset_path):
    now = datetime.datetime.now()
    folder = "run-%s" % now.strftime("%m%d-%H%M%S")
    writer = SummaryWriter(logdir=os.path.join(asset_path, 'tensorboard', folder))

    return writer


def print_result(losses, metrics):
    for name, val in losses.items():
        logger.info("%s: %.4f" % (name, val))
    for name, val in metrics.items():
        logger.info("%s: %.4f" % (name, val))


def tensorboard_logging_result(tf_writer, epoch, results):
    for tag, value in results.items():
        if 'img' in tag:
            tf_writer.add_image(tag, value, epoch)
        elif 'hist' in tag:
            tf_writer.add_histogram(tag, value, epoch)
        else:
            tf_writer.add_scalar(tag, value, epoch)


def pitch_to_midi(pitch, chord, frame_per_bar=16, save_path=None, basis_note=60):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name='melody')
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second

    on_pitch = {}
    for t, idx in enumerate(pitch):
        if idx in range(48):
            if bool(on_pitch):
                note = pretty_midi.Note(start=on_pitch['time'],
                                        end=t * unit_time,
                                        pitch=on_pitch['pitch'],
                                        velocity=100)
                instrument.notes.append(note)
                on_pitch = {}
            on_pitch['pitch'] = basis_note + idx
            on_pitch['time'] = t * unit_time
        elif idx == 49 and bool(on_pitch):
            note = pretty_midi.Note(start=on_pitch['time'],
                                    end=t * unit_time,
                                    pitch=on_pitch['pitch'],
                                    velocity=100)
            instrument.notes.append(note)
            on_pitch = {}

    if bool(on_pitch):
        note = pretty_midi.Note(start=on_pitch['time'],
                                end=t * unit_time,
                                pitch=on_pitch['pitch'],
                                velocity=100)
        instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    midi_data.instruments.append(chord_to_instrument(chord, frame_per_bar=frame_per_bar))
    if save_path is not None:
        midi_data.write(save_path)

    return midi_data.instruments


def chord_to_instrument(chord_array, frame_per_bar=16):
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second
    instrument = pretty_midi.Instrument(program=0, name='chord')
    chord = chord_array[0]
    prev_t = 0
    for t in range(chord_array.shape[0]):
        if not (chord_array[t] == chord).all():
            chord_notes = chord.nonzero()[0]
            for pitch in chord_notes:
                note = pretty_midi.Note(start=prev_t * unit_time, end=t * unit_time, pitch=48 + pitch, velocity=70)
                instrument.notes.append(note)
            prev_t = t
            chord = chord_array[t]
    chord_notes = chord.nonzero()[0]
    for pitch in chord_notes:
        note = pretty_midi.Note(start=prev_t * unit_time, end=chord_array.shape[0] * unit_time, pitch=48 + pitch, velocity=70)
        instrument.notes.append(note)
    return instrument


def save_instruments_as_image(filename, instruments, frame_per_bar=16, num_bars=8):
    melody_inst = instruments[0]
    timelen = frame_per_bar * num_bars
    frame_per_second = (frame_per_bar / 4) * 2
    unit_time = 1 / frame_per_second

    piano_roll = melody_inst.get_piano_roll(fs=frame_per_second)
    if piano_roll.shape[1] < timelen:
        piano_roll = np.pad(piano_roll, ((0, 0), (0, timelen - piano_roll.shape[1])),
                            mode="constant", constant_values=0)
    for note in melody_inst.notes:
        note_len = note.end - note.start
        note.end = note.start + min(note_len, unit_time)
    onset_roll = melody_inst.get_piano_roll(fs=frame_per_second)
    if onset_roll.shape[1] < timelen:
        onset_roll = np.pad(onset_roll, ((0, 0), (0, timelen - onset_roll.shape[1])),
                            mode="constant", constant_values=0)

    if (num_bars) // 16 > 1:
        rows = (num_bars) // 16
        lowest_pitch = 36
        highest_pitch = 96
        C_labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    else:
        rows = 1
        lowest_pitch = 0
        highest_pitch = 128
        C_labels = ['C-1', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    fig = plt.figure(figsize=(8, 6))
    for row in range(rows):
        ax = fig.add_subplot(rows, 1, row + 1)
        ax.set_ylim([lowest_pitch, highest_pitch])
        ax.set_xticks(np.arange(frame_per_bar, timelen // rows, frame_per_bar))
        ax.set_xticklabels(np.arange(row*(timelen // rows) + 2 * frame_per_bar, (row+1)*(timelen // rows) + frame_per_bar, frame_per_bar))
        ax.set_yticks(np.arange(lowest_pitch, highest_pitch, 12))
        ax.set_yticklabels(C_labels)
        for C_idx in range(12 + lowest_pitch, highest_pitch, 12):
            ax.axhline(y=C_idx, color='b', linewidth=0.5)
        for i in range(num_bars // rows):
            ax.axvline(x=frame_per_bar*(i+1), color='r', linewidth=0.5)
        plot = plt.imshow((piano_roll + onset_roll)[:, row*(timelen // rows):(row+1)*(timelen // rows)] / 2, interpolation=None, cmap='gray_r')
    plt.savefig(filename)
    plt.close(fig)

root_note_list = [' C', 'C#', ' D', 'D#', ' E', ' F', 'F#', ' G', 'G#', ' A', 'A#', ' B']

def idx_list_to_symbol_list(idx_list):
    symbol_list = []
    for i, event_idx in enumerate(idx_list):
        if event_idx == 48:
            symbol = '%03d,  N' % (i + 1)
        elif event_idx == 49:
            symbol =  ''
        else:
            octave = event_idx // 12 + 3
            root_note = event_idx % 12
            symbol = '%03d,%s%d' % (i + 1, root_note_list[root_note], octave)
        symbol_list.append(symbol)
    return symbol_list


def chord_to_symbol_list(chord):
    symbol_list = []
    root_list = []
    for root in chord[0].nonzero()[0].tolist():
        root_list.append(root_note_list[root].replace(' ', ''))
    symbol = ','.join(root_list)
    symbol_list.append('001,'+symbol)
    for i in range(1, chord.shape[0]):
        if (chord[i] - chord[i-1]).tolist() == np.zeros(12).tolist():
            symbol = ''
        else:
            root_list = []
            for root in chord[i].nonzero()[0].tolist():
                root_list.append(root_note_list[root].replace(' ', ''))
            symbol = '-'.join(root_list)
            symbol = '%03d,%s' % (i + 1, symbol)
        symbol_list.append(symbol)
    return symbol_list

def rhythm_to_symbol_list(beat_list):
    symbol_list = []
    for i, beat in enumerate(beat_list):
        if beat == 2:
            symbol = '%03d, 2' % (i + 1)
        elif beat == 1:
            symbol = '1'
        else:
            symbol = ''
        symbol_list.append(symbol)
    return symbol_list

def pitch_to_symbol_list(pitch_list):
    symbol_list = []
    for i, pitch in enumerate(pitch_list):
        if pitch == 88:
            symbol = ''
        else:
            octave = (pitch + 20) // 12 - 2
            root_note = (pitch + 20) % 12
            symbol = '%03d,%s%d' % (i + 1, root_note_list[root_note], octave)
        symbol_list.append(symbol)
    return symbol_list


def chord_dict_to_array(chord_dict, max_len):
    chord = []
    next_t = max_len
    for t in sorted(chord_dict.keys(), reverse=True):
        chord_array = np.zeros(12)
        for note in chord_dict[t] % 12:
            chord_array[note] = 1
        chord_array = np.tile(chord_array, (next_t - t, 1))
        chord.append(chord_array)
        next_t = t
    if next_t != 0:
        chord.append(np.tile(np.zeros(12), (next_t, 1)))
    chord = np.concatenate(chord)[::-1]
    return chord

def chord_array_to_dict(chord_array):
    chord_dict = dict()
    chord = np.zeros(12)
    for t in range(chord_array.shape[0]):
        if not (chord_array[t] == chord).all():
            chord_dict[t] = chord_array[t].nonzero()[0]
            chord = chord_array[t]
    return chord_dict