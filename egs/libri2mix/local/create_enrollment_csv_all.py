# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import sys
from collections import defaultdict

# Creates CSV mapping mixture-id and utterance-id into paths to enrollment audios.
# As input, CSV describing the mixtures in the dataset in required. This is prepared
# by Asteroid local/create_local_metadata.py script.

# This script assigns each mixture with the list of ALL possible enrollement utterances.
# This is useful for training data, where we want to choose a random enrollment utterance
# in each epoch of the training.

# Example of line of input CSV
# mixture_ID,mixture_path,source_1_path,source_2_path,noise_path,length
# 1578-6379-0038_6415-111615-0009,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/mix_both/1578-6379-0038_6415-111615-0009.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/s1/1578-6379-0038_6415-111615-0009.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/s2/1578-6379-0038_6415-111615-0009.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/noise/1578-6379-0038_6415-111615-0009.wav,53560.0

# Example of line of output CSV (here, 3 possible enrollment utterances are shown,
# but typically there would be more)
# mixture_id,utterance_id,enr_path1,length1,enr_path2,length2,...
# 1578-6379-0038_6415-111615-0009,1578-6379-0038,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/s1/1578-140049-0017_911-128684-0003.wav,92280.0,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/s2/3857-180923-0023_1578-140049-0025.wav,60880.0,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/train-100/s1/1578-6379-0024_8226-274371-0036.wav,61120.0

mix_csv = sys.argv[1]
out_enr_csv = sys.argv[2]

spk2utts = defaultdict(set)
utt2pathlen = {}
mix_ids = []
with open(mix_csv) as f:
    f.readline()
    for line in f:
        mix_id, _, s1_path, s2_path, _, length = line.strip().split(',')
        mix_ids.append(mix_id)
        utt1id, utt2id = mix_id.split('_')
        spk1, spk2 = utt1id.split('-')[0], utt2id.split('-')[0]
        spk2utts[spk1].add(utt1id)
        spk2utts[spk2].add(utt2id)
        utt2pathlen[utt1id] = (s1_path, length)
        utt2pathlen[utt2id] = (s2_path, length)

with open(out_enr_csv, 'w') as f:
    f.write('mixture_id,utterance_id,enr_path1,length1,enr_path2,length2,...\n')
    for mix_id in mix_ids:
        utt1, utt2 = mix_id.split('_')

        # 1st speaker
        f.write(f'{mix_id},{utt1},')
        enr_all = []
        for utt_id in spk2utts[utt1.split('-')[0]]:
            if utt_id == utt1:
                continue
            enr_utt, length = utt2pathlen[utt_id]
            enr_all.append(enr_utt)
            enr_all.append(length)
        f.write(','.join(enr_all))
        f.write('\n')

        # 2nd speaker
        f.write(f'{mix_id},{utt2},')
        enr_all = []
        for utt_id in spk2utts[utt2.split('-')[0]]:
            if utt_id == utt2:
                continue
            enr_utt, length = utt2pathlen[utt_id]
            enr_all.append(enr_utt)
            enr_all.append(length)
        f.write(','.join(enr_all))
        f.write('\n')
