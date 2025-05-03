#!/bin/bash
# Ensure if all base software result installed

# For project
sudo apt update
sudo apt upgrade
sudo apt install git gh -y

# Python (via conda)
CONDA_DIR=~/miniconda3
if [ ! -d "$CONDA_DIR" ]; then
	echo "[Installing Conda]"
	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm ~/miniconda3/miniconda.sh
	
	

	
else
	echo "[Conda just installed, SKIP]"
fi

# configure conda
source ~/miniconda3/bin/activate
conda init --all

# create environment MNLP (if not just exist)
ENV=~/miniconda3/envs/MNLP

if [[ ! -d $ENV  ]]; then
	conda create -n MNLP python=3.13.2 # latest release
fi

# activate environment

conda activate MNLP

# install required python packages for base operations
echo '[installing Base packages]'
pip3 install scipy numpy nltk pandas pytest
pip3 install matplotlib wandb
pip3 install requests-cache
pip3 install beautifulsoup4
pip3 install seaborn
pip3 install datasets
pip3 install lxml

# mnlp tools
pip3 install nltk

# install ml tools (and torch system)
echo '[installing ML tools]'
nvidia-smi >> /dev/null
if [[ -x "$(command -v nvidia-smi)" ]] ; then
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 #require install giga-byte
else
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi	

pip3 install scikit-learn
pip3 install torchmetrics

# Hugging Face dependecies
pip3 install transformers
pip3 install datasets
pip3 install evaluate 
pip3 install accelerate
pip3 install hf_xet
pip3 install 'accelerate>=0.26.0'

# install tools to access wikimedia project data and other modeling tools
echo '[installing Wikimedia Tools]'
pip install wikipedia
pip install networkx
