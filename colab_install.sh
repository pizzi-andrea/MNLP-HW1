#!/bin/bash
# light installer version for colab notebook (no require env)
# Ensure if all base software result installed


# install required python packages for base operations
echo '[installing Base packages]'
pip3 install scipy numpy nltk pandas pytest
pip3 install matplotlib wandb
pip3 install requests-cache
pip3 install beautifulsoup4
pip3 install seaborn
pip3 install datasets
pip3 install lxml

# install ml tools 
echo '[installing ML tools]'

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
