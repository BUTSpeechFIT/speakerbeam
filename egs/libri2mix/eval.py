# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

# This code is based on Asteroid eval.py, 
# which is released under the following MIT license:
# https://github.com/asteroid-team/asteroid/blob/master/LICENSE

import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path

from asteroid.metrics import get_metrics
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates

from models.td_speakerbeam import TimeDomainSpeakerBeam
from datasets.librimix_informed import LibriMixInformed

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_dir", type=str, required=True, help="Test directory including the csv files"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="One of `enh_single`, `enh_both`, " "`sep_clean` or `sep_noisy`",
)

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the model (either best model or checkpoint, "
         "which is determined by parameter --from_checkpoint"
)

parser.add_argument(
    "--from_checkpoint",
    type=int,
    default=0,
    help="Model in model path is checkpoint, not final model. Default: 0"
)

parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
    help="Directory where the eval results" " will be stored",
)
parser.add_argument(
    "--use_gpu", type=int, default=0, help="Whether to use the GPU for model execution"
)
parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment root")

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]



def main(conf):
    compute_metrics = COMPUTE_METRICS
    if not conf["from_checkpoint"]:
        model = TimeDomainSpeakerBeam.from_pretrained(conf["model_path"])
    else:
        model = TimeDomainSpeakerBeam(**conf["train_conf"]["filterbank"],
                                   **conf["train_conf"]["masknet"],
                                   **conf["train_conf"]["enroll"]
                                   ) 
        ckpt = torch.load(conf["model_path"], map_location = torch.device('cpu')) 
        state_dict = {} 
        for k in ckpt['state_dict']: 
            state_dict[k.split('.',1)[1]] = ckpt['state_dict'][k]
        model.load_state_dict(state_dict)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = LibriMixInformed(
        csv_dir=conf["test_dir"],
        task=conf["task"],
        sample_rate=conf["sample_rate"],
        n_src=conf["train_conf"]["data"]["n_src"],
        segment=None,
        segment_aux=None,
    )  # Uses all segment length

    eval_save_dir = conf["out_dir"]
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, source, enroll = test_set[idx]
        mix, source, enroll = tensors_to_device([mix, source, enroll], device=model_device)
        est_source = model(mix.unsqueeze(0), enroll.unsqueeze(0))
        mix_np = mix.cpu().data.numpy()
        source_np = source.cpu().data.numpy()
        est_source_np = est_source.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            source_np,
            est_source_np,
            sample_rate=conf["sample_rate"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.mixture_path
        est_source_np_normalized = normalize_estimates(est_source_np, mix_np)
        series_list.append(pd.Series(utt_metrics))

        # Save some examples in a folder. Wav files and metrics as text.
        local_save_dir = os.path.join(conf["out_dir"], "out/")
        os.makedirs(local_save_dir, exist_ok=True)
        mixture_name = test_set.mixture_path.split('/')[-1].split('.')[0]
        sf.write(local_save_dir + f"{mixture_name}_"
                                  f"s{test_set.target_speaker_idx}.wav",
                 est_source_np_normalized[0], conf["sample_rate"])

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)



if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["sample_rate"] = train_conf["data"]["sample_rate"]
    arg_dic["train_conf"] = train_conf

    if args.task != arg_dic["train_conf"]["data"]["task"]:
        print(
            "Warning : the task used to test is different than "
            "the one from training, be sure this is what you want."
        )

    main(arg_dic)
