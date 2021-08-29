# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import sys
from collections import defaultdict

# Creates CSV mapping mixture-id and utterance-id into paths to enrollment audios.
# As input, CSV describing the mixtures in the dataset in required. This is prepared
# by Asteroid local/create_local_metadata.py script. Second, a map of mixture-ids
# to enrollment utterance ids is required.

# This script assigns each mixture with one enrollment utterance for each speaker
# based on the provided map. This is useful for test and validation data, where
# we want to keep the mapping fixed across experiments or projects.

# Example of line of input CSV
# mixture_ID,mixture_path,source_1_path,source_2_path,noise_path,length
# 4077-13754-0001_5142-33396-0065,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/test/mix_both/4077-13754-0001_5142-33396-0065.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/test/s1/4077-13754-0001_5142-33396-0065.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/test/s2/4077-13754-0001_5142-33396-0065.wav,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/test/noise/4077-13754-0001_5142-33396-0065.wav,30160.0

# Example of lines of map of mixtures to enrollments
# mixture_ID target_utterance_ID enrollment_ID
# 4077-13754-0001_5142-33396-0065 4077-13754-0001 s1/4077-13754-0004_5142-36377-0020
# 4077-13754-0001_5142-33396-0065 5142-33396-0065 s1/5142-36377-0003_1320-122612-0014
# 6930-76324-0027_5683-32879-0011 6930-76324-0027 s2/4992-41797-0018_6930-75918-0017
# 6930-76324-0027_5683-32879-0011 5683-32879-0011 s1/5683-32866-0017_8455-210777-0024

# Example of a line of output CSV
# mixture_id,utterance_id,enr_path1,length1
# 4077-13754-0001_5142-33396-0065,4077-13754-0001,/mnt/matylda6/izmolikova/Corpora/LibriMix/Libri2Mix/wav8k/min/test/s1/4077-13754-0004_5142-36377-0020.wav,36240.0

mix_csv = sys.argv[1]
map_mix2enroll = sys.argv[2]
out_enr_csv = sys.argv[3]

utt2pathlen = {}
mix_ids = []
with open(mix_csv) as f:
    f.readline()
    for line in f:
        mix_id, _, s1_path, s2_path, _, length = line.strip().split(',')
        mix_ids.append(mix_id)
        utt2pathlen[f's1/{mix_id}'] = (s1_path, length)
        utt2pathlen[f's2/{mix_id}'] = (s2_path, length)

mix2enroll = {}
with open(map_mix2enroll) as f:
    for line in f:
        mix_id, utt_id, enroll_id = line.strip().split()
        mix2enroll[mix_id,utt_id] = enroll_id

with open(out_enr_csv, 'w') as f:
    f.write('mixture_id,utterance_id,enr_path1,length1\n')
    for mix_id in mix_ids:
        utt1, utt2 = mix_id.split('_')
        
        # 1st speaker
        enr_id = mix2enroll[mix_id,utt1]
        enr_utt, length = utt2pathlen[enr_id]
        f.write(f'{mix_id},{utt1},{enr_utt},{length}\n')

        # 2nd speaker
        enr_id = mix2enroll[mix_id,utt2]
        enr_utt, length = utt2pathlen[enr_id]
        f.write(f'{mix_id},{utt2},{enr_utt},{length}\n')
