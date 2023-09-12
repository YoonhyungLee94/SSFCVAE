import os
import sys
sys.path.append('bigvgan/')

import torch
from torch.utils.data import DataLoader
from bigvgan import get_vocoder

import librosa
from utils import *
import yaml
from dataset import AudioSet, collate_fn
import numpy as np
from tqdm import tqdm
from modules import Model, Discriminator
import matplotlib.pyplot as plt


def visualize_data(target, pred, noisy, step, label):
    fig, axs = plt.subplots(3, figsize=(20,15))
    im0 = axs[0].imshow(target.detach().cpu().numpy(), origin='lower', aspect='auto')
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pred.detach().cpu().numpy(), origin='lower', aspect='auto')
    fig.colorbar(im1, ax=axs[1])
    im2 = axs[2].imshow(noisy.detach().cpu().numpy(), origin='lower', aspect='auto')
    fig.colorbar(im2, ax=axs[2])
    writer.add_figure(f'{label}', fig, global_step=step)

    
def train_model(model, disc_mel, vocoder, train_dataloader, optim_G, optim_D, device):
    global writer, iteration
    model.train()
    for batch in train_dataloader:
        ground_truth_list = batch[0]
        (mel_clean, mel_noisy, spec_noisy, spec_lengths,
         w2v_feats_clean, w2v_feats_noisy, w2v_lengths) = ([x.to(device) for x in batch[1:]])
        
        lr_warmup = 5*len(train_dataloader)
        if iteration < lr_warmup:
            optim_G.param_groups[0]['lr'] = ((iteration+1)/lr_warmup) * config['learning_rate']
            optim_D.param_groups[0]['lr'] = ((iteration+1)/lr_warmup) * config['learning_rate']
            
        mel_pred, kl_loss, mel_mask, w2v_mask, z_p = model(spec_noisy, spec_lengths,
                                                            w2v_feats_clean, w2v_feats_noisy, w2v_lengths)
        
        ####### Discriminator #######
        y, y_hat = disc_mel(torch.cat([mel_clean, mel_pred.detach()], dim=0), mel_mask.repeat(2, 1)).chunk(2, dim=0)
        loss_d = torch.masked_select( (1-y)**2 + y_hat**2, mel_mask).mean()

        optim_D.zero_grad(set_to_none=True)
        loss_d.backward()
        optim_D.step()

        ####### Model #######
        y_hat = disc_mel(mel_pred, mel_mask)
        loss_g = torch.masked_select((1-y_hat)**2, mel_mask).mean()
        loss_recon = torch.masked_select(torch.abs(mel_clean-mel_pred).mean(1), mel_mask).mean()
        
        kl_grads = torch.autograd.grad(kl_loss, z_p, retain_graph=True)[0]
        kl_grads = torch.masked_select(torch.norm(kl_grads, dim=1), w2v_mask).mean()

        recon_grads = torch.autograd.grad(loss_g + loss_recon, z_p, retain_graph=True)[0]
        recon_grads = torch.masked_select(torch.norm(recon_grads, dim=1), w2v_mask).mean()
        
        half_cycle = 25*len(train_dataloader) # single cycle lasts for 50 epochs
        anneal = min( 1.0, (iteration%(2*half_cycle))/(half_cycle) )
        alpha = anneal / (kl_grads/(recon_grads+1e-9)).item()
        
        optim_G.zero_grad(set_to_none=True)
        (loss_g + loss_recon + alpha*kl_loss).backward()
        optim_G.step()

        iteration += 1
        
        if iteration % 10 == 0:
            writer.add_scalar('train/loss_d', loss_d.item(), global_step=iteration)
            writer.add_scalar('train/loss_g', loss_g.item(), global_step=iteration)
            writer.add_scalar('train/loss_recon', loss_recon.item(), global_step=iteration)
            writer.add_scalar('train/kl_loss', kl_loss.item(), global_step=iteration)
            
        if iteration % 1000 == 0:
            with torch.no_grad():
                mel_pred = model.inference(spec_noisy[0:1, :, :spec_lengths[0]], spec_lengths[0:1],
                                           w2v_feats_noisy[0:1, :, :w2v_lengths[0]], w2v_lengths[0:1])
                visualize_data(mel_clean[0, :, :spec_lengths[0]],
                               mel_pred[0, :, :spec_lengths[0]],
                               mel_noisy[0, :, :spec_lengths[0]], iteration, 'train')
                
                wav_clean_16k, wav_noisy_16k = ground_truth_list[0]
                wav_pred = vocoder(mel_pred[:, :, :spec_lengths[0]].float())[0].detach().cpu().numpy()
                wav_pred_16k = librosa.resample(wav_pred, orig_sr=22050, target_sr=16000)
                
            writer.add_audio('train/wav_clean',  wav_clean_16k, iteration, 16000)
            writer.add_audio('train/wav_noisy', wav_noisy_16k, iteration, 16000)
            writer.add_audio('train/wav_pred', wav_pred_16k, iteration, 16000)


def validate_model(model, disc_mel, vocoder, val_dataloader, device):
    global writer, iteration
    (loss_d_tot, loss_g_tot, loss_recon_tot, kl_loss_tot) = 0, 0, 0, 0
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            ground_truth_list = batch[0]
            (mel_clean, mel_noisy, spec_noisy, spec_lengths,
             w2v_feats_clean, w2v_feats_noisy, w2v_lengths) = ([x.to(device) for x in batch[1:]])
            
            mel_pred, kl_loss, mel_mask, w2v_mask, z_p = model(spec_noisy, spec_lengths,
                                                               w2v_feats_clean, w2v_feats_noisy, w2v_lengths)

            ####### Discriminator #######
            y, y_hat = disc_mel(torch.cat([mel_clean, mel_pred], dim=0), mel_mask.repeat(2, 1)).chunk(2, dim=0)
            loss_d = torch.masked_select( (1-y)**2 + y_hat**2, mel_mask).mean()
            loss_g = torch.masked_select((1-y_hat)**2, mel_mask).mean()
            
            loss_recon = torch.masked_select(torch.abs(mel_clean-mel_pred).mean(1), mel_mask).mean()

            loss_d_tot += loss_d.item()
            loss_g_tot += loss_g.item()
            loss_recon_tot += loss_recon.item()
            kl_loss_tot += kl_loss.item()
            
            wav_clean_16k, wav_noisy_16k = ground_truth_list[0]

            mel_pred = model.inference(spec_noisy, spec_lengths, w2v_feats_noisy, w2v_lengths)
            wav_pred = vocoder(mel_pred[:, :, :spec_lengths[0]].float())[0].detach().cpu().numpy()
            wav_pred_16k = librosa.resample(wav_pred, orig_sr=22050, target_sr=16000)

            if i in [7, 30, 31, 45, 175, 247, 297, 418, 452, 477, 578, 736, 764, 777]:
                visualize_data(mel_clean[0], mel_pred[0], mel_noisy[0], iteration, f'val/sample{i}')
                writer.add_audio(f'val/sample{i}/wav_clean',  wav_clean_16k, iteration, 16000)
                writer.add_audio(f'val/sample{i}/wav_noisy', wav_noisy_16k, iteration, 16000)
                writer.add_audio(f'val/sample{i}/wav_pred', wav_pred_16k, iteration, 16000)
        
        writer.add_scalar('val/loss_d', loss_d_tot/len(val_dataloader.dataset), global_step=iteration)
        writer.add_scalar('val/loss_g', loss_g_tot/len(val_dataloader.dataset), global_step=iteration)
        writer.add_scalar('val/loss_recon', loss_recon_tot/len(val_dataloader.dataset), global_step=iteration)
        writer.add_scalar('val/kl_loss', kl_loss_tot/len(val_dataloader.dataset), global_step=iteration)


def main(config):
    # Load datasets
    global writer, iteration
    writer = get_writer(config['output_directory'], config['logdir'])
    
    train_dataset = AudioSet(config['train_dir'])
    val_dataset = AudioSet(config['val_dir'], train_dataset.w2v_clean_scaler, train_dataset.w2v_noisy_scaler)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=collate_fn,
                                  num_workers=6, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Instantiate the model
    model = Model(config['n_mel_channels'], config['n_spec_channels'], config['n_w2v_channels'],
                  config['hdim'], config['latent_dim'], config['n_head'], config['d_inner'],
                  config['n_layers'], config['n_flows'])
    disc_mel = Discriminator(config['n_mel_channels'])
    vocoder = get_vocoder("bigvgan_22khz_80band")

    # If a GPU is available, move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    disc_mel.to(device)
    vocoder.to(device)

    # Define the optimizer and loss function
    optim_G = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], betas=[0.5, 0.9])
    optim_D = torch.optim.AdamW(disc_mel.parameters(), lr=config['learning_rate'], betas=[0.5, 0.9])
    
    # Train the model
    iteration=0
    for epoch in range(config['epochs']):
        train_model(model, disc_mel, vocoder, train_dataloader, optim_G, optim_D, device)
        validate_model(model, disc_mel, vocoder, val_dataloader, device)
        
        if epoch>5:
            optim_G.param_groups[0]['lr'] = 0.99 * optim_G.param_groups[0]['lr']
            optim_D.param_groups[0]['lr'] = 0.99 * optim_D.param_groups[0]['lr']
        
    save_checkpoint(model, optim_G, config['learning_rate'], iteration, f"{config['output_directory']}/{config['logdir']}")
    writer.close()


if __name__ == "__main__":
    # Load the config from the YAML file
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # If argparse arguments are provided, use them to override the YAML configuration
    args = parse_arguments()
    args_dict = vars(args)
    config = {**config, **args_dict}
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Now you can use the config dictionary in your project
    main(config)
