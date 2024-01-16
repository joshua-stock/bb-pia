# Black-Box Property Inference Attacks
Version 0.1
## Setup
* Locally (in virtual environment):
`pip install -r requirements.txt`
* Server:
```
conda create --name "pia-tf" python=3.10.9 ipython
conda activate pia-tf
conda install jupyter pip
pip install pandas numpy tensorflow keras-tuner
conda deactivate && python -m ipykernel install --user --name pia-tf
nano ~/.local/share/jupyter/kernels/pia-tf/kernel.json
```
(last command: add the kernel manually in case it has not been added to list)

### Create synthetic data for genereting model output
* see `generate-synthetic-data.py`
* No need to re-run, result is included as `data/syn_data.csv`
### Create shadow model outputs to train adversary
* No need to re-run, adversary exists (see next section)
* Training and using shadow models
* Run `generate-adv-input.py` -> might take a long time
* Finetuning of adversary training has been done in `tune_adversary.ipynb`
### Loading and using the adversary
* See `pia.ipynb`

---
### Coming up in next version 0.2:
* adversary training with less redundant training data
* synthetic data with proper header in file