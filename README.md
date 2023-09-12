# SSFCVAE
> Official PyTorch implementation of the paper "Boosting Speech Enhancement with Clean Self-Supervised Features via Conditional Variational Autoencoders"

## Introduction
Provide an in-depth overview of the project. Clearly explain the problem statement and how this project solves it. Reference key papers, benchmarks, or metrics that your model aims to improve.

### Prerequisites
* Download the VoiceBank-DEMAND dataset in advance [here](https://datashare.ed.ac.uk/handle/10283/2791)
* Prepare for using the BigVGAN vocoder [here](https://github.com/NVIDIA/BigVGAN). This code uses the pre-trained model in the `bigvgan_22khz_80band' folder, so please also download the pre-trained weights in advance.

## Installation
```bash
# Clone this repository
git clone https://github.com/YoonhyungLee94/SSFCVAE

# Navigate into the directory
cd SSFCVAE

# Install requirements
pip install -r requirements.txt
ln -s location/of/your/dataset/including/the/VoiceBank-DEMAND/folder Dataset
ln -s location/of/the/bigvgan/repo bigvgan
```

## Usage
Train the CVAE model using the below command.

```python
python train.py --gpu 0 --logdir ssf_cvae
```

## Audio samples
Audio samples are included in the folder `./audio_samples'[./audio_samples]

## License
This project is licensed under the [MIT License](LICENSE).
