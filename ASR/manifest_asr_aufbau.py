from pathlib import Path
import pandas as pd
import os
import scipy
import csv
from scipy.io import wavfile
import json
import random

#absolute path generierung
csv_file = #path_to_csv
df = pd.read_csv(csv_file)
absolute_path_list = []
root_dir = #root_dir_path
for index, row in df.iterrows():
    paths = row["path"]
    paths = root_dir + str(paths)
    paths = paths.replace("/ ", "/")
    absolute_path_list.append(paths)

# länge audiofiles auf 3. nachkommastelle gerundet
duration_list = []
for file_name in absolute_path_list:
    if(os.path.isfile(file_name)):
        sample_rate, data = wavfile.read(file_name)
    elif(os.path.isfile(file_name.replace(".wav", " .wav"))):
        sample_rate, data = wavfile.read(file_name.replace(".wav", " .wav"))
    elif(os.path.isfile(file_name.replace(" .wav", "  .wav"))):
        sample_rate, data = wavfile.read(file_name.replace(" .wav", "  .wav"))
    elif(os.path.isfile(file_name.replace("  .wav", "   .wav"))):
        sample_rate, data = wavfile.read(file_name.replace(" .wav", "   .wav"))
    else:
      data = [0]
      sample_rate = 16
    len_data = len(data)  #länge numpy array
    t = len_data / sample_rate
    duration_list.append(round(t, 3)) #rundung auf 3. nachkommastelle

df["duration"] = duration_list

#Formattierung zu {"audio_filepath": path, "duration": länge, "sentence": transcript} formattierung
output_file = 'formatted_output.json'
with open(output_file, 'w', encoding='utf-8') as out_file:
    for index, row in df.iterrows():
        audio_path = row["path"]
        audio_path = str(audio_path)
        duration = float(row["duration"])
        transcript = str(row[" sentence "])
        metadata = {
            "audio_filepath": root_dir.rstrip(" ") + audio_path.lstrip(" "),
            "duration": duration,
            "text": transcript}
        json.dump(metadata, out_file)
        out_file.write(",")
        out_file.write('\n')

# Unterteilung manifest in training, validation, test datensatz


def split_text_file(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1): #80/10/10 ratio
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    indices = list(range(total_lines))
    random.shuffle(indices) #durchmischung der daten

    train_split = int(train_ratio * total_lines)
    val_split = int((train_ratio + val_ratio) * total_lines)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    def write_lines(output_file, indices):
        with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
            for idx in indices:
              f.write(lines[idx])

    write_lines('training.json', train_indices)
    write_lines('validation.json', val_indices)
    write_lines('test.json', test_indices)

# Example usage:
input_file = #path_to_input
output_directory = #path_to_output

split_text_file(input_file, output_directory)
