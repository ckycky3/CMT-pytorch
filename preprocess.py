import glob
import os
import random
import argparse
import pickle
import numpy as np
import pretty_midi as pm
from progress.bar import Bar
from scipy.sparse import csc_matrix

duplicate_set = {480, 556, 525, 806, 531, 878, 533, 1007, 585, 1121, 616, 634, 637, 647,
                 644, 645, 664, 682, 829, 693, 694, 733, 825, 700, 751, 734, 746, 755, 790,
                 768, 1008, 783, 784, 794, 1006, 854, 956, 865, 1106, 868, 919, 928, 959, 940,
                 1020, 945, 1087, 957, 958, 1136, 970, 1077, 1048, 1139, 1057, 1084, 1126}

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


def make_instance_pkl_files(root_path, midi_dir, num_bars, frame_per_bar, pitch_range=48,
                            beat_per_bar=4, bpm=120, data_ratio=(0.8, 0.1, 0.1)):
    dir_name = os.path.join(root_path, 'pkl_files', 'uk_instance_pkl_%dbars_fpb%d_cleansed' % (num_bars, frame_per_bar))
    os.makedirs(dir_name, exist_ok=True)

    instance_len = frame_per_bar * num_bars
    stride = int(instance_len / 2)
    # Default : frame_per_second=8, unit_time=0.125
    frame_per_second = (frame_per_bar / beat_per_bar) * (bpm / 60)
    unit_time = 1 / frame_per_second

    # midi_files = glob.glob(os.path.join(root_path, midi_dir, '*.midi'))
    midi_files = glob.glob(os.path.join(root_path, 'midi', midi_dir, '*.midi'))
    num_eval = int(len(midi_files) * data_ratio[1])
    num_test = int(len(midi_files) * data_ratio[2])
    random.seed(0)
    eval_set = random.sample(set(range(464, 1140)) - duplicate_set, num_eval)
    test_set = random.sample(set(range(464, 1140)) - duplicate_set - set(eval_set), num_test)
    # eval_set = random.sample(range(len(midi_files)), num_eval)
    # test_set = random.sample(range(len(midi_files)), num_test)

    # for j, midi_file in enumerate(Bar("Processing").iter(sorted(midi_files))):
    for j, midi_file in enumerate(sorted(midi_files)):
        # filename = midi_file.split('/')[-1].replace('_cleansed_chord_processed.midi', '')
        filename = midi_file.split('/')[-1].split('.')[0].split('_')[0]

        if j in eval_set:
            mode = 'eval'
        elif j in test_set:
            mode = 'test'
        else:
            mode = 'train'

        os.makedirs(os.path.join(dir_name, mode, filename), exist_ok=True)

        # for k in range(-5, 7):
        midi = pm.PrettyMIDI(midi_file)
        if len(midi.instruments) < 2:
            continue
        on_midi = pm.PrettyMIDI(midi_file)
        off_midi = pm.PrettyMIDI(midi_file)
        note_instrument = midi.instruments[0]
        onset_instrument = on_midi.instruments[0]
        offset_instrument = off_midi.instruments[0]
        for note, onset_note, offset_note in zip(note_instrument.notes, onset_instrument.notes, offset_instrument.notes):
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
            chord_note.end = chord_note.start + unit_time
        chord_roll = chord_instrument.get_piano_roll(fs=frame_per_second)
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
            chord_dict = dict()
            # chord_list = []

            pianoroll_inst = pianoroll[:, i:(i+instance_len+1)]
            onset_inst = onset_roll[:, i:(i+instance_len+1)]
            offset_inst = offset_roll[:, i:(i+instance_len+1)]
            chord_inst = chord_onset[:, i:(i + instance_len + 1)]

            if len(chord_inst.nonzero()[1]) < 4:
                continue

            beat_idx = np.minimum(np.sum(pianoroll_inst.T, axis=1), 1) + np.minimum(np.sum(onset_inst.T, axis=1), 1)
            beat_idx = beat_idx.astype(int)
            # If more than 75% is not-playing, do not make instance
            if beat_idx.nonzero()[0].size < (instance_len // 4):
                continue

            highest_note = max(onset_inst.T.nonzero()[1])
            lowest_note = min(onset_inst.T.nonzero()[1])
            # base_note = 12 * (lowest_note // 12)
            base_note = 48

            if highest_note - base_note >= pitch_range:
                continue

            # prev_chord = np.zeros(12)
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
                elif beat_idx[t] == 1:
                    pitch_list.append(pitch_range)
                elif beat_idx[t] == 0:
                    pitch_list.append(pitch_range + 1)
                    cont_rest += 1
                    if cont_rest >= 30:
                        break
                else:
                    print(filename, i, t, beat_idx[t], onset_inst.T.nonzero())

                if len(chord_inst[:, t].nonzero()[0]) != 0:
                    chord_dict.update({t: chord_inst[:, t].nonzero()[0]})

                    # if t % int(frame_per_bar / 4) == int(frame_per_bar / 4) - 1:
                    #     chord_list.append(prev_chord)
                    #     prev_chord = np.zeros(12)
                    #     for note in sorted(chord_inst[:, t].nonzero()[0] % 12):
                    #         prev_chord[note] = 1
                    #     continue
                    # else:
                    #     prev_chord = np.zeros(12)
                    #     for note in sorted(chord_inst[:, t].nonzero()[0] % 12):
                    #         prev_chord[note] = 1
                    #     if t % int(frame_per_bar / 4) == 1:
                    #         chord_list[t - 1] = prev_chord

                    # prev_chord = np.zeros(12)
                    # for note in sorted(chord_inst[:, t].nonzero()[0] % 12):
                    #     prev_chord[note] = 1
            #     chord_list.append(prev_chord)
            # chord_list = np.array(chord_list)

            if (not chord_dict) or (cont_rest >= 30) or (len(set(pitch_list)) <= 5):
                print(filename, i // stride)
                continue

            # pitch_list = np.array(pitch_list)
            # result = {'beat': beat_idx,
            #           'pitch': pitch_list,
            #           'chord': chord_dict}
            #           # 'chord': csc_matrix(chord_list)}
            #
            # pkl_filename = os.path.join(dir_name, mode, filename, '%s_%02d.pkl' % (filename, i // stride))
            # # ps = ('%d' % k) if (k < 0) else ('+%d' % k)
            # # pkl_filename = os.path.join(dir_name, mode, filename, '%s_%s_%02d.pkl' % (filename, ps, i // stride))
            # with open(pkl_filename, 'wb') as f:
            #     pickle.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/data2/score2midi')
    parser.add_argument('--original_dir', type=str, default='cleansed_split_midi')
    parser.add_argument('--midi_dir', type=str, default='cleansed_midi_twotrack_ckey')
    parser.add_argument('--num_bars', type=int, default=8)
    parser.add_argument('--frame_per_bar', type=int, default=16)
    parser.add_argument('--pitch_range', type=int, default=48)

    args = parser.parse_args()
    root_path = args.root_path
    original_dir = args.original_dir
    midi_dir = args.midi_dir
    num_bars = args.num_bars
    frame_per_bar = args.frame_per_bar
    pitch_range = args.pitch_range

    if not os.path.exists(os.path.join(root_path, midi_dir)):
        make_twotrack_midi(root_path, original_dir, midi_dir)

    make_instance_pkl_files(root_path, midi_dir, num_bars, frame_per_bar, pitch_range)