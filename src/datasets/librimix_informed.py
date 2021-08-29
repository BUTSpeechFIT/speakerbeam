# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from pathlib import Path
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
from asteroid.data import LibriMix
import random
import torch
import soundfile as sf

def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, 'r') as f:
        f.readline() # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(',')
            aux_it = iter(aux)
            aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[mix_id][utt_id] = aux
    return data

class LibriMixInformed(Dataset):
    def __init__(
        self, csv_dir, task="sep_clean", sample_rate=16000, n_src=2, 
        segment=3, segment_aux=3, 
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, segment)
        self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')

        if segment_aux is not None:
            max_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            self.seg_len_aux = int(segment_aux * sample_rate)
            self.data_aux = {m: {u:  
                [(path,length) for path, length in self.data_aux[m][u]
                    if length >= self.seg_len_aux
                    ]
                for u in self.data_aux[m]} for m in self.data_aux}
            new_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            print(
                f"Drop {max_len - new_len} utterances from {max_len} "
                f"(shorter than {segment_aux} seconds)"
            )
        else:
            self.seg_len_aux = None

        self.seg_len = self.base_dataset.seg_len

        # to choose pair of mixture and target speaker by index
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]

    def __len__(self):
        return len(self.data_aux_list)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            start = random.randint(0, length - seg_len)
            stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()

        mixture_path = row['mixture_path']
        self.mixture_path = mixture_path
        tgt_spk_idx = mix_id.split('_').index(utt_id)
        self.target_speaker_idx = tgt_spk_idx

        # read mixture
        start, stop = self._get_segment_start_stop(self.seg_len, row['length'])
        mixture,_ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        mixture = torch.from_numpy(mixture)

        # read source
        source_path = row[f'source_{tgt_spk_idx+1}_path']
        source,_ = sf.read(source_path, dtype="float32", start=start, stop=stop)
        source = torch.from_numpy(source)[None]

        # read enrollment
        enroll_path, enroll_length = random.choice(self.data_aux[mix_id][utt_id])
        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll,_ = sf.read(enroll_path, dtype="float32", start=start_e, stop=stop_e)
        enroll = torch.from_numpy(enroll)

        return mixture, source, enroll

    def get_infos(self):
        return self.base_dataset.get_infos()

