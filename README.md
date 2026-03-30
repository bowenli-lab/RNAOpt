# 🧬 RNAOpt: Co-design of mRNA Functional Regions via Inference-Time Parallel Reasoning 🧬
Welcome to the official implementation of **RNAOpt**, an mRNA design platform with inference time reasoning over the massive space.

## 🚀 Setting up environment 🚀
Create new environment:

```

conda create -n rnaopt python=3.10.12
conda activate rnaopt

```

Then install the packages used:

```

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.55.2
pip install huggingface_hub
pip install packaging
pip install lightning==2.4.0
pip install h5py
pip install pandas
pip install biopython
pip install git+https://github.com/Benjamin-Lee/CodonAdaptationIndex.git

pip install mamba-ssm[causal-conv1d]


```


If the installation of mamba-ssm and casual-conv1d are not successful (frequently happens), you can choose to install manually by two methods:
1. Run following command on your own machine to get the information for versions, then look up for built wheels from the release of [mamba](https://github.com/state-spaces/mamba/releases) and [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) repository, then download the pre-built wheel to your machine.

```
import sys
import torch
print(f"🍱 Python Version: {sys.version.split()[0]}")
print(f"🍛 PyTorch Version: {torch.__version__}")
print(f"🧪 CUDA Version: {torch.version.cuda}")
print(f"🥨 Use CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}")

```
After downloading the prebuilt wheels to the machine, use `pip install` to install them.

If there is no pre-built wheel for your machine environment, you can download the code and build by yourself:

```

git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install .

git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
pip install .

```


## 🥘 Data for training
Our curated Stage 1 & Stage 2 and fine-tuning datasets are open-sourced on [Zenodo](https://doi.org/10.5281/zenodo.18805220)

## 🍳 Model Checkpoint
We provide pretrained, fine-tuned, and distilled weights on [HuggingFace](https://huggingface.co/ReaganGen/RNAOpt/tree/main)

## 🍜 RNAOpt_Evaluator Pretrain
To initiate or continue pre-training. Please use the RNAOpt_Evaluator/pretrain.py. Use the command:

```

python RNAOpt_Evaluator/pretrain.py --config <Path to pretrain config file>

```

Note: Ensure the paths for `train`, `val`, and `test` files, as well as `ckpt_dir`(checkpoint save path) and `log_dir`(log file save path), are correctly updated in your configuration file. We recommend maintaining the default training settings.


## 🍲 RNAOpt_Evaluator finetune
Use the RNAOpt_Evaluator/finetune.py to fine-tune the model with the command:

```

python RNAOpt_Evaluator/finetune.py --config <Path to finetune config file>

```

Note: Apart from paths for `train`, `val`, `test`, `ckpt_dir`, `log_dir` in the config files. It is also important to load the pretrained checkpoint path `pretrained_model_path` in the config file. Please also remember to keep the model architecture the same as the pretrained model.

## 🍰 RNAOpt_Evaluator inference
To perform high-throughput inference and score mRNA candidates, please use the RNAOpt_E/inference.py with the command:

```

python RNAOpt_Evaluator/inference.py --config <Path to inference config file>

```

If you need to customize the output format of the predictions, please modify RNAOpt_E/inference.py directly.

## 🍣 RNAOpt_Sequence_Generator 
We have included the Human Codon frequency table as an example, sourced from [Kazusa](https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1&style=N), please look for the species that you need from this repo. You need to import the codon frequency table by setting the path `human_codon_freq_path` in the RNAOpt_T.py. 

There are a few important parameters that could be adjusted：
- `beam_width` is used to desccribe how many candidates are kept for next step of reasoning
- `beta` is the number of sequences that are directly send to RNAOpt-E prediction in the diversity perservation
- `theta` is the weight of CAI in the definition of the fitnes score (setting it high will make the molecule design more CAI biased, recommend keep the original value)
- `patience` is the threshold of early stopping for no improvement of molecule fitness
- `batch_size` is the batch size used when perform inference
- `max_steps` the max number of steps for optimization
- `partial_sampling_percentage` the percentage used for sampling for the partial sampling strategy

To perform mRNA design, use the command as following:

```

python RNAOpt_Sequence_generator/RNAOpt_T.py --config <Path to RNAOpt_T search config file>

```


After getting the history of sequences that are checked, we select the candidates with the following script:

```

import pandas as pd

# Load the CSV file
csv_path = <Path to the result file from RNAOpt_Sequence_generator>
df = pd.read_csv(csv_path)

# Find the maximum step (i.e., the final round)
last_step = df['Step'].max()

# Filter rows corresponding to the final step
final_step_df = df[df['Step'] == last_step]

# Sort by Top_HL descending and take the top 3
top3_sequences = final_step_df.sort_values("Top_HL", ascending=False).head(3)

# Display or save
print(top3_sequences)

```

The selected sequence candidates are sent for wet lab experiments

## 🍞 Online GUI usage
We offer a ready-to-use [Colab notebook](https://colab.research.google.com/drive/1rUlp0QwX6QcOyYT1fS8SjgyFBcYLAK7l#scrollTo=v-3Kpxp_WP6W&forceEdit=true&sandboxMode=true) with an integrated GUI, enabling users to bypass complex local environment configurations and run optimizations directly in the browser.

## 🍟 Contributing
We greatly welcome contributions to RNAOpt. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using RNAOpt

## Acknowledgements
We sincerely thank the authors of following open-source projects:
- [Jamba](https://huggingface.co/ai21labs/Jamba-v0.1)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [mamba](https://github.com/state-spaces/mamba)
- [transformers](https://github.com/huggingface/transformers)

## Citing RNAOpt
Will update this part upon publication