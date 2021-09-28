# SpeakerBeam for neural target speech extraction

This repository contains an implementation of SpeakerBeam method for target speech extraction, made public during Interspeech 2021 tutorial.

The code is based on the [Asteroid toolkit](https://github.com/asteroid-team/asteroid) for audio speech separation.

## Requirements

To install requirements:
```
pip install -r requirements.txt
```
The code was tested with Python 3.8.6.

## Running the experiments
The directory `egs` contains a recipe for [Libri2mix dataset](https://github.com/JorisCos/LibriMix). The recipe assumes that you already have the LibriMix dataset available. If not, please follow the instructions at the [LibriMix repository](https://github.com/JorisCos/LibriMix) to obtain it. 

Before running our recipe, modify `path.sh` file to contain path to the repository root. 
```
PATH_TO_REPOSITORY="<path-to-repo>"
```
Then follow the steps below in the recipe directory
```
cd egs/libri2mix
```

### Preparing data
To prepare lists of the data run
```
local/prepare_data.sh <path-to-libri2mix-data>
```
The `<path-to-libri2mix-data>` should contain `wav8k/min` subdirectories. The command will create `data` directory containing `csv` lists describing the data. The preparation of the data follows the data preparation from Asteroid. In addition, it creates a list mapping mixtures to enrollment utterances.

### Training SpeakerBeam
To train the SpeakerBeam model run
```
. ../../path.sh
python train.py --exp_dir exp/speakerbeam
```
The training script will by default use parameters in `local/conf.yml`. To run with different parameters, you can either change the `local/conf.yml` file or pass them directly as command-line arguments, e.g.
```
python train.py --exp_dir exp/speakerbeam_adaptlay15 --i_adapt_layer 15
```
to change the position of the adaptation layer in the network.

The training will create directory `exp/speakerbeam`. The final model after the training is finished is stored in `exp/speakerbeam/best_model.pth`. The training progress can be observed in Tensorboard using logs in `exp/speakerbeam/lightning_logs`.

### Decoding and evaluating the performance
To extract target speech signals on the test set with the trained model and evaluate performance, run
```
python eval.py --test_dir data/wav8k/min/test --task sep_noisy --model_path exp/speakerbeam/best_model.pth --out_dir exp/speakerbeam/out_best --exp_dir exp/speakerbeam --use_gpu=1
```
It is also possible to evaluate with an intermediate checkpoint, e.g.
```
python eval.py --test_dir data/wav8k/min/test --task sep_noisy --model_path exp/speakerbeam/checkpoints/epoch\=24-step\=115824.ckpt --from_checkpoint 1 --out_dir exp/speakerbeam/out_e24_s115824 --exp_dir exp/speakerbeam --use_gpu=1
```

In the output directory `exp/speakerbeam/out_best`, you can find the averaged results in `final_metrics.json` and the extracted audio files in `<out_dir>/out`.

## Reference
Please cite our works when using this code:
```
@ARTICLE{Zmolikova_Spkbeam_STSP19,
  author={Žmolíková, Kateřina and Delcroix, Marc and Kinoshita, Keisuke and Ochiai, Tsubasa and Nakatani, Tomohiro and Burget, Lukáš and Černocký, Jan},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures}, 
  year={2019},
  volume={13},
  number={4},
  pages={800-814},
  doi={10.1109/JSTSP.2019.2922820}}

@INPROCEEDINGS{delcroix_tdSpkBeam_ICASSP20,
  author={Delcroix, Marc and Ochiai, Tsubasa and Zmolikova, Katerina and Kinoshita, Keisuke and Tawara, Naohiro and Nakatani, Tomohiro and Araki, Shoko},
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Improving Speaker Discrimination of Target Speech Extraction With Time-Domain Speakerbeam}, 
  year={2020},
  volume={},
  number={},
  pages={691-695},
  doi={10.1109/ICASSP40776.2020.9054683}}
```
