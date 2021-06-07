import numpy as np
import pretty_midi
import os
import functools
from hyper_param import *
from PIL import Image
from math import ceil

remove_instruments = [i for i in range(113,129)]+[i for i in range(97,105)] 
def midi_to_array(notes,length):
    arr = np.zeros((MAX_NOTE-MIN_NOTE,length),dtype=np.int8)

    for note in notes:
        pitch,start = note
        arr[pitch-MIN_NOTE,start]=1
    row_sum = np.sum(arr,axis=0)
    zero_start = 0
    while row_sum[zero_start] == 0:
        zero_start += 1
    arr = arr[:,zero_start:]
    while arr.shape[1] < TIME_STEP+1:
        arr = np.concatenate([arr,np.copy(arr)],axis=1)
    return arr

def load_midi(file_name):
    midi_file = pretty_midi.PrettyMIDI(file_name)
    beats = midi_file.get_beats()
    # time_sig = midi_file.time_signature_changes[0].denominator
    # precision = 64/time_sig
    
    time_signatures = midi_file.time_signature_changes
    time_signature_idx = 0
    
    precision = NOTE_PRECISION/4
    if len(time_signatures) != 0:

        precision = NOTE_PRECISION/time_signatures[time_signature_idx].denominator
    divided_beats = []
    for i in range(beats.shape[0]-1):
        if time_signature_idx < len(time_signatures)-1:
            if beats[i] >= time_signatures[time_signature_idx].time:
                time_signature_idx+=1
                precision = NOTE_PRECISION/time_signatures[time_signature_idx].denominator

        for j in range(int(precision)):
            divided_beats.append((beats[i+1]-beats[i])/precision * j + beats[i])

    length = len(divided_beats)
    divided_beats = np.array(divided_beats)

    notes = []

    for instrument in midi_file.instruments:
        if instrument.is_drum:
            continue
        if instrument.program in remove_instruments:
            continue
        for note in instrument.notes:
            if note.pitch < MIN_NOTE or note.pitch>=MAX_NOTE:
                continue
            # start = midi_file.time_to_tick(note.start)
            dist = np.abs(note.start - divided_beats)
            closest = np.argsort(dist)

            notes.append((note.pitch,closest[0]))

    return notes,length

def load_all(load_array_from_file = True,save = True):
    if load_array_from_file:
        try:
            data = np.load("./midi_array/data.npy")
            print("successfully loaded", data.shape,data.dtype)
            return data
        except Exception as e:
            print(e)
            print("reload file")
    midis = []
    lengths = []
    num_data = 0
    i = 0
    for root,_,files in os.walk("./training_data/"):
        for f in files:
            try:
                # midi_file = pretty_midi.PrettyMIDI(root+f)
                notes,length = load_midi(root+f)
                # beat = midi_file.get_beats()
                # length = beat.shape[0]
                num_data += ceil(length/TIME_STEP)*2
                midis.append(notes)
                lengths.append(length)
                print(f,"loaded, len = ", length)
                i += 1
            except Exception as e:
                print("failed to load file "+f,e)
    data = np.zeros((num_data,TIME_STEP,MAX_NOTE-MIN_NOTE),dtype=np.int8)
    idx = 0
    for i, notes in enumerate(midis):
        if len(notes)==0:
            continue
        arr = midi_to_array(notes,lengths[i])
        # array_to_midi(arr*128,"foo")
        seg = preprocess(arr)
        data[idx:idx+seg.shape[0]] = seg
        idx += seg.shape[0]
        data[idx:idx+seg.shape[0]] = np.flip(seg,axis=2)
        idx += seg.shape[0]
    data = data[:idx]
    assert np.sum(data[-1])!=0
    if save:
        np.save("./midi_array/data.npy",data)
    return data

def preprocess(arr):
    cut=[]
    arr = np.transpose(arr)
    for j in range(0,arr.shape[0]-TIME_STEP,int(TIME_STEP)):
        cut.append(arr[j:j+TIME_STEP])
    cut = np.array(cut)
    return cut[:,:]
def array_to_midi(midi_array,file_name,tempo=30,threshold=64,dur=0.2,speed=0.5):
    midi_file = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(0)
    for i in range(midi_array.shape[0]):
        notes = []
        current_note = midi_array[i]
        has_note = np.squeeze(np.argwhere(midi_array[i]>threshold),axis=-1)
        if has_note.shape[0]>0:
            difference = has_note[1:]-has_note[0:-1]
            gap = np.squeeze(np.argwhere(difference>1),axis=-1)
            start = has_note[0]
            for idx,j in enumerate(gap):
                end = has_note[j]
                velocity = 127
                start = midi_file.tick_to_time(start)/speed
                x = pretty_midi.Note(velocity=velocity, pitch=i+MIN_NOTE, start=start, end=start+dur)
                for k in range(len(notes)):
                    if notes[k].start <= start and notes[k].end >= start:
                        collide = notes.pop(k)
                        collide.end = start-0.01
                        notes.append(collide)

                notes.append(x)
                start = has_note[j+1]
            end = has_note[-1]
            velocity = 127
            start = midi_file.tick_to_time(start)/speed
            x = pretty_midi.Note(velocity=velocity, pitch=i+MIN_NOTE, start=start, end=start+dur)
            for k in range(len(notes)):
                if notes[k].start <= start and notes[k].end >= start:
                    collide = notes.pop(k)
                    collide.end = start-0.01
                    notes.append(collide)
            notes.append(x)
            for note in notes:
                piano.notes.append(note)
    midi_file.instruments.append(piano)
    midi_file.write(file_name+".mid")

if __name__=="__main__":
    # data,length = load_midi("./akishimaiNoNakuKoroni.mid")
    # arr = midi_to_array(data,length)
    # print(arr.shape)
    # array_to_midi(arr*128,"foo",speed=0.2)
    # img = Image.fromarray(arr*128)
    # img = img.convert("RGBA")
    # img.save("foo.png")
    # print(arr.shape)
    data = load_all(load_array_from_file = False)

    # np.random.shuffle(data)
    
    