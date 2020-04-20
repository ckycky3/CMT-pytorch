import glob
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from progress.bar import Bar
from scipy.sparse import csc_matrix

duplicate_set = {14, 74, 49, 353, 71, 313, 227, 73, 1328, 106, 134, 150, 317, 153, 340, 170, 311, 315,
                 1205, 334, 390, 403, 339, 413, 350, 421, 351, 383, 358, 359, 391, 395, 394, 407, 397,
                 440, 441, 418, 419, 422, 1034, 473, 1069, 442, 461, 525, 806, 531, 878, 533, 1007, 585,
                 1121, 601, 876, 616, 634, 637, 647, 644, 645, 664, 682, 829, 693, 694, 733, 825, 700, 751,
                 734, 746, 755, 790, 783, 784, 768, 1008, 794, 1006, 854, 956, 865, 1106, 868, 919, 928,
                 959, 929, 1199, 940, 1020, 945, 1087, 957, 958, 1136, 970, 1077, 1048, 1139, 1057, 1084,
                 1126, 1196, 1395, 1198, 1349, 1358, 1380, 1361, 1362}

def make_twotrack_midi(root_path, original_dir='cleansed_split_midi', midi_dir='cleansed_midi_twotrack'):
    os.makedirs(os.path.join(root_path, midi_dir), exist_ok=True)
    prefix = original_dir.replace('split_midi', '')
    for mldy_file in sorted(glob.glob(os.path.join(root_path, original_dir, '*_%schord_processed_melody.midi' % prefix))):
        if not os.path.exists(mldy_file.replace('_melody.midi', '_chord.midi')):
            continue
        midi_mldy = pm.PrettyMIDI(mldy_file)
        mldy_instrument = midi_mldy.instruments[0]
        mldy_instrument.name = 'melody'

        chord_file = mldy_file.replace('_melody.midi', '_chord.midi')
        midi_chord = pm.PrettyMIDI(chord_file)
        chord_instrument = midi_chord.instruments[0]
        chord_instrument.name = 'chord'

        new_midi = pm.PrettyMIDI()
        new_midi.instruments.append(mldy_instrument)
        new_midi.instruments.append(chord_instrument)
        new_midi.write(os.path.join(root_path, midi_dir,
                                    mldy_file.split('/')[-1].replace('_melody.midi', '.midi')))

def pad_pianorolls(pianoroll, timelen):
    if pianoroll.shape[1] < timelen:
        pianoroll = np.pad(pianoroll, ((0, 0), (0, timelen - pianoroll.shape[1])),
                           mode="constant", constant_values=0)
    return pianoroll


def make_instance_pkl_files(root_path, midi_dir, num_bars, frame_per_bar, pitch_range=48, augmentation=False, dictionary=False,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):
    if augmentation:
        instance_folder = '12aug_instance_pkl_%dbars_fpb%d_%dp' % (num_bars, frame_per_bar, pitch_range)
    else:
        instance_folder = 'instance_pkl_%dbars_fpb%d_%dp_ckey' % (num_bars, frame_per_bar, pitch_range)
    dir_name = os.path.join(root_path, 'pkl_files', instance_folder)
    os.makedirs(dir_name, exist_ok=True)

    if 'Wikifonia' in root_path:
        num_songs = 5029
    elif 'score2midi' in root_path:
        num_songs = 1397
    instance_len = frame_per_bar * num_bars
    stride = int(instance_len / 2)
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    # midi_files = glob.glob(os.path.join(root_path, midi_dir, '*.midi'))
    midi_files = glob.glob(os.path.join(root_path, 'midi', midi_dir, '*.midi'))
    num_eval = int(num_songs * data_ratio[1])
    num_test = int(num_songs * data_ratio[2])
    random.seed(0)
    eval_set = random.sample(set(range(num_songs)) - duplicate_set, num_eval)
    test_set = random.sample(set(range(num_songs)) - duplicate_set - set(eval_set), num_test)
    # eval_set = random.sample(range(num_songs), num_eval)
    # test_set = random.sample(set(range(num_songs)) - set(eval_set), num_test)

    for midi_file in Bar("Processing").iter(sorted(midi_files)):
        filename = int(midi_file.split('/')[-1].split('.')[0].split('_')[0])
        key_num = midi_file.split('/')[-1].split('.')[0].split('_')[1]

        if filename in eval_set:
            mode = 'eval'
        elif filename in test_set:
            mode = 'test'
        else:
            mode = 'train'

        os.makedirs(os.path.join(dir_name, mode, '%04d' % filename), exist_ok=True)

        if augmentation:
            pitch_shift = range(-5, 7)
        else:
            pitch_shift = [0]
        for k in pitch_shift:
            midi = pm.PrettyMIDI(midi_file)
            if len(midi.instruments) < 2:
                continue
            on_midi = pm.PrettyMIDI(midi_file)
            off_midi = pm.PrettyMIDI(midi_file)
            note_instrument = midi.instruments[0]
            onset_instrument = on_midi.instruments[0]
            offset_instrument = off_midi.instruments[0]
            for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
                if k != 0:
                    note.pitch += k
                    onset_note.pitch += k
                    offset_note.pitch += k
                note_length = offset_note.end - offset_note.start
                onset_note.end = onset_note.start + min(note_length, unit_time)
                offset_note.end += unit_time
                offset_note.start = offset_note.end - min(note_length, unit_time)
            pianoroll = note_instrument.get_piano_roll(fs=frame_per_second)
            onset_roll = onset_instrument.get_piano_roll(fs=frame_per_second)
            offset_roll = offset_instrument.get_piano_roll(fs=frame_per_second)

            chord_instrument = midi.instruments[1]
            timelen = min(pianoroll.shape[1], offset_roll.shape[1])
            for chord_note in chord_instrument.notes:
                if k != 0:
                    chord_note.pitch += k
                chord_note.end = chord_note.start + unit_time
            # chord_roll = chord_instrument.get_piano_roll(fs=frame_per_second)
            chord_onset = chord_instrument.get_piano_roll(fs=frame_per_second)

            pianoroll = pad_pianorolls(pianoroll, timelen)
            onset_roll = pad_pianorolls(onset_roll, timelen)
            offset_roll = pad_pianorolls(offset_roll, timelen)
            chord_onset = pad_pianorolls(chord_onset, timelen)

            pianoroll[pianoroll > 0] = 1
            onset_roll[onset_roll > 0] = 1
            offset_roll[offset_roll > 0] = 1
            chord_onset[chord_onset > 0] = 1

            for i in range(0, timelen - (instance_len + 1), stride):
                # cnt = 0
                pitch_list = []
                # comment out
                if dictionary:
                    chord_dict = dict()
                else:
                    chord_list = []

                pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
                onset_inst = onset_roll[:, i:(i+instance_len+1)]
                offset_inst = offset_roll[:, i:(i+instance_len+1)]
                chord_inst = chord_onset[:, i:(i + instance_len + 1)]

                if len(chord_inst.nonzero()[1]) < 4:
                    continue

                rhythm_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
                rhythm_idx = rhythm_idx.astype(int)
                # If more than 75% is not-playing, do not make instance
                if rhythm_idx.nonzero()[0].size < (instance_len // 4):
                    continue

                if pitch_range == 128:
                    base_note = 0
                else:
                    highest_note = max(onset_inst.T.nonzero()[1])
                    lowest_note = min(onset_inst.T.nonzero()[1])
                    base_note = 12 * (lowest_note // 12)
                    if highest_note - base_note >= pitch_range:
                        continue

                if not dictionary:
                    prev_chord = np.zeros(12)
                cont_rest = 0
                prev_onset = 0
                for t in range(instance_len+1):
                    if t in onset_inst.T.nonzero()[0]:
                        # TODO: pitch to idx, lowest or highest
                        pitch_list.append(onset_inst[:, t].T.nonzero()[0][0] - base_note)
                        # pitch_list.append(onset_inst[:, t].T.nonzero()[0][0])
                        if (t != onset_inst.T.nonzero()[0][0]) and abs(onset_inst[:, t].T.nonzero()[0][0] - base_note - prev_onset) > 12:
                            cont_rest = 30
                            break
                        else:
                            prev_onset = onset_inst[:, t].T.nonzero()[0][0] - base_note
                            cont_rest = 0
                    elif rhythm_idx[t] == 1:
                        pitch_list.append(pitch_range)
                    elif rhythm_idx[t] == 0:
                        pitch_list.append(pitch_range + 1)
                        cont_rest += 1
                        if cont_rest >= 30:
                            break
                    else:
                        print(filename, i, t, rhythm_idx[t], onset_inst.T.nonzero())

                    if len(chord_inst[:, t].nonzero()[0]) != 0:
                        if dictionary:
                            chord_dict.update({t: chord_inst[:, t].nonzero()[0][1:]})
                        else:
                            prev_chord = np.zeros(12)
                            for note in sorted(chord_inst[:, t].nonzero()[0][1:] % 12):
                                prev_chord[note] = 1
                    if not dictionary:
                        chord_list.append(prev_chord)

                if (dictionary and not chord_dict) or (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                    continue

                pitch_list = np.array(pitch_list)
                if dictionary:
                    chord_result = chord_dict
                else:
                    chord_result = csc_matrix(np.array(chord_list))
                result = {'pitch': pitch_list,
                          # 'rhythm': rhythm_idx,
                          'beat': rhythm_idx,
                          'chord': chord_result}
                if augmentation:
                    ps = ('%d' % k) if (k < 0) else ('+%d' % k)
                    pkl_filename = os.path.join(dir_name, mode, '%04d' % filename, '%04d_%s_%s_%02d.pkl' % (filename, ps, key_num, i // stride))
                else:
                    pkl_filename = os.path.join(dir_name, mode, '%04d' % filename, '%04d_%s_%02d.pkl' % (filename, key_num, i // stride))
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/data2/score2midi')
    parser.add_argument('--original_dir', type=str, default='cleansed_split_midi')
    parser.add_argument('--midi_dir', type=str, default='cleansed_midi_twotrack_ckey')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--frame_per_bar', type=int, default=16)
    parser.add_argument('--pitch_range', type=int, default=48)
    parser.add_argument('--aug', dest='aug', action='store_true')
    parser.add_argument('--dictionary', dest='dictionary', action='store_true')

    args = parser.parse_args()
    root_path = args.root_path
    original_dir = args.original_dir
    midi_dir = args.midi_dir
    num_bars = args.num_bars
    frame_per_bar = args.frame_per_bar
    pitch_range = args.pitch_range
    aug = args.aug
    dictionary = args.dictionary

    if not os.path.exists(os.path.join(root_path, 'midi', midi_dir)):
        make_twotrack_midi(root_path, original_dir, midi_dir)

    make_instance_pkl_files(root_path, midi_dir, num_bars, frame_per_bar, pitch_range, aug, dictionary)