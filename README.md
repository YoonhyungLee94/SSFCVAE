# SSFCVAE
This code provides the official PyTorch implementation of the paper "_Boosting Speech Enhancement with Clean Self-Supervised Features via Conditional Variational Autoencoders_", which is submitted to ICASSP 2024.

### Prerequisites
* Download the VoiceBank-DEMAND dataset [here](https://datashare.ed.ac.uk/handle/10283/2791)
* Prepare for using the BigVGAN vocoder [here](https://github.com/NVIDIA/BigVGAN), including the pre-trained model `bigvgan_22khz_80band`.

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
Audio samples are included in the folder ['./audio_samples'](./audio_samples)

## License
This project is licensed under the [MIT License](LICENSE).
