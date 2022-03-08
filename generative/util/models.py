from pathlib import Path
from re import A
from typing import NewType, Optional, Union

import numpy as np
import torch
from torch import distributions, nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


JSON = NewType('JSON', dict)
EPS = 1e-12


def _phase_shuffle(audio_tensor, n):
    """ Performs a phase shuffle of `n` samples.
    See figure 3 of https://arxiv.org/pdf/1802.04208.pdf for details
    """

    shifted = torch.empty_like(audio_tensor)
    # Whether the shift is to the left or right
    direction = ( torch.rand(1)[0] > 0.5 ).item()
    if direction:
        # Shift samples to the left
        shifted[..., :-n] = audio_tensor[..., n:]
        filler = audio_tensor[..., -n-1:-1]
        # Mirror the filler values
        shifted[..., -n:] = torch.flip(filler, dims=(2,))
    else:
        # Shift samples to the right
        shifted[..., n:] = audio_tensor[..., :-n]
        filler = audio_tensor[..., 1:n+1]
        shifted[..., :n] = torch.flip(filler, dims=(2,))
    
    return shifted


class GerbilizerDiscriminator(nn.Module):
    def __init__(
        self, 
        config: JSON
    ):
        super().__init__()
        n_mics = config['num_microphones']
        multiplier = config['dimensionality_multiplier']

        self.nonlin = nn.LeakyReLU(negative_slope=0.2)

        kernel_size = config['conv_kernel_size']
        padding = config['conv_padding_size']
        self.convs = nn.ModuleList()
        self.nin = nn.ModuleList()

        self.convs.append(
            nn.Conv1d(
                n_mics,
                2 * multiplier,
                kernel_size=kernel_size,
                padding='same',
                stride=1
            )
        )
        self.nin.append(
            nn.Conv1d(
                2 * multiplier,
                n_mics,
                kernel_size=1,
                padding='same'
            )
        )

        for _ in range(9):
            self.convs.append(
                nn.Conv1d(
                    n_mics,
                    2 * multiplier,
                    kernel_size=kernel_size,
                    padding='same',
                    stride=1
                )
            )
            self.nin.append(
                nn.Conv1d(
                    2 * multiplier,
                    n_mics,
                    kernel_size=1,
                    padding='same'
                )
            )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(n_mics * 16, 1)

    def forward(self, x: Tensor) -> Tensor:
        working_audio = x
        for n, (conv, nin) in enumerate(zip( self.convs, self.nin )):
            convolved = conv(working_audio)
            working_audio = working_audio + self.nonlin(convolved)
            working_audio = nin(working_audio)
            if n < len(self.convs) - 1:
                working_audio = _phase_shuffle(working_audio, 2)
            working_audio = F.max_pool1d(working_audio, kernel_size=2, stride=2)
        reshaped = self.flatten(working_audio)
        output = self.dense(reshaped)
        return output
        

class GeneratorBlock(nn.Module):
    """https://arxiv.org/pdf/1909.11646.pdf ?
    """
    def __init__(self, in_channels: int, out_channels: int, filt_size: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.nonlin = nn.LeakyReLU(negative_slope=0.2)

        self.convs = nn.ModuleList([
            nn.Conv1d(out_channels, out_channels, filt_size, 1, dilation=1, padding='same'),
            nn.Conv1d(out_channels, out_channels, filt_size, 1, dilation=2, padding='same'),
            nn.Conv1d(out_channels, out_channels, filt_size, 1, dilation=4, padding='same'),
            nn.Conv1d(out_channels, out_channels, filt_size, 1, dilation=8, padding='same')
        ])

        # self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.skip_conv = nn.Identity
        # self.second_skip_conv = nn.Conv1d(out_channels, out_channels, 1)

        self.first_block = nn.Sequential(
            self.nonlin,
            self.upsample,
            self.convs[0],
            self.nonlin,
            self.convs[1]
        )

        self.second_block = nn.Sequential(
            self.nonlin,
            self.convs[2],
            self.nonlin,
            self.convs[3]
        )
    
    def forward(self, x):
        skip_conn = self.skip_conv(self.upsample(x))
        subblock_1 = self.first_block(x)
        block_out = skip_conn + subblock_1

        final_out = self.second_block(block_out) + block_out
        return final_out


class GerbilizerGenerator(nn.Module):
    def __init__(
        self,
        config: JSON
    ):
        """ Inspired by the WaveGAN architecture:
        https://arxiv.org/pdf/1802.04208.pdf
        """
        super().__init__()
        latent_size = config['latent_size']
        # Ensures the number of channels used is divisible by 256
        multiplier = config['dimensionality_multiplier']
        self.dense = nn.Linear(latent_size, 512 * multiplier)

        n_mics = config['num_microphones']
        filt_size = config['generator_conv_kernel_size']

        channel_sizes = [
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier,
            32 * multiplier
        ]
        self.starting_channels = channel_sizes[0]

        self.blocks = nn.ModuleList()
        for in_size, out_size in zip(channel_sizes[:-1], channel_sizes[1:]):
            self.blocks.append(GeneratorBlock(in_size, out_size, filt_size))

        self.final_conv = nn.Conv1d(channel_sizes[-1], n_mics, filt_size, dilation=2, padding='same')
    
    def forward(self, z: Tensor) -> Tensor:
        starting_audio = self.dense(z)
        # I think tensor.view might be applicable here
        working_audio = starting_audio.reshape((-1, self.starting_channels, 16))
        for block in self.blocks:
            working_audio = block(working_audio)
        output = torch.tanh(self.final_conv(working_audio))
        return output


class GerbilizerGAN:
    def __init__(
        self,
        config: JSON,
    ):
        # Lambda in eq. 3 of the paper
        self.gpu = config['device'].lower() == 'gpu'
        self.batch_size = config['batch_size']  # int
        self.gp_coeff = config['grad_penalty_weight']  # float
        self.latent_size = config['latent_size']  # int
        self.critic_steps = config['critic_steps_per_gen_step'] # int

        latent_prior_type = config['latent_prior']  # str
        if latent_prior_type == 'normal':
            # Assuming a spherical gaussian, so all variables in z are independent
            self.latent_sampler = lambda n: torch.randn((n, self.latent_size))
        elif latent_prior_type == 'uniform':
            self.latent_sampler = lambda n: torch.rand((n, self.latent_size,)) * 2 - 1
        else:
            raise ValueError(f'Unsupported prior distribution: {latent_prior_type}.')

        self.generator = GerbilizerGenerator(config)
        self.discriminator = GerbilizerDiscriminator(config)
        if self.gpu:
            self.generator.cuda()
            self.discriminator.cuda()

        learning_rate = config['learning_rate']
        betas = config['adam_beta_1'], config['adam_beta_2']
        self.disc_optimizer = Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=0
        )
        self.gen_optimizer = Adam(
            self.generator.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=0
        )
    
    def train(self):
        """ Relays the shift to the training state to both child models
        """
        self.discriminator.train()
        self.generator.train()
    
    def eval(self):
        """ Relays the shift to the evaluative state to both child models
        """
        self.discriminator.eval()
        self.generator.eval()

    def train_minibatch(self, data_iterator):
        self.train()
        critic_losses = self._discriminator_training_loop(
            data_iterator,
            report_gp_loss_term=True
        )

        gen_loss = self._gen_loss()
        mean_loss = torch.mean(gen_loss)
        self.gen_optimizer.zero_grad()
        mean_loss.backward()
        self.gen_optimizer.step()

        reported_losses = {
            'critic_losses': critic_losses,
            'generator_loss': mean_loss.detach().cpu().item()
        }
        # If fewer than `critic_steps` iterations were done, then the data loader
        # exhausted itself in this batch
        return reported_losses, len(critic_losses) < self.critic_steps
        
    def sample_from_generator(
        self, 
        n_samples: int, *, 
        latent: Union[np.ndarray, Tensor]=None, 
        as_numpy: bool=True) -> Tensor:
        """ Generates `n_samples` random samples and optionally, returns the
        result as a detached numpy array.
        """
        if latent is None:
            latent_batch = self.latent_sampler(n_samples)
        else:
            if not isinstance(latent, Tensor):
                latent_batch = torch.from_numpy(latent)
            else:
                latent_batch = latent
        if self.gpu:
            latent_batch = latent_batch.cuda()
        gen_data = self.generator(latent_batch)
        if as_numpy:
            return gen_data.detach().cpu().numpy()
        return gen_data

    def save_checkpoint(self, checkpoint_path):
        gen_state = self.generator.state_dict()
        disc_state = self.discriminator.state_dict()
        checkpoint_path = Path(checkpoint_path)
        parent_dir = checkpoint_path.parent

        gen_fname = checkpoint_path.stem + '_gen.pt'
        gen_path = parent_dir / Path(gen_fname)

        disc_fname = checkpoint_path.stem + '_disc.py'
        disc_path = parent_dir / Path(disc_fname)

        torch.save(gen_state, gen_path)
        torch.save(disc_state, disc_path)
    
    def _discriminator_training_loop(
        self, 
        data_iterator, *, 
        report_gp_loss_term: bool=False
    ) -> Optional[list]:
        gp_losses = list()
        # Iterate at most `critic_steps` times through data_iterator
        for _, x_real in zip(range(self.critic_steps), data_iterator):
            if len(x_real) < self.batch_size:
                # Don't process incomplete batches
                break
            latents = self.latent_sampler(self.batch_size)
            if self.gpu:
                latents = latents.cuda()
                x_real = x_real.cuda()
            x_gen = self.generator(latents)
            gp_loss = self._disc_gp_loss(x_real, x_gen)
            mean_loss = torch.mean(gp_loss)
            gp_losses.append(mean_loss.detach().cpu().item())
            self.disc_optimizer.zero_grad()
            mean_loss.backward()
            self.disc_optimizer.step()
        if report_gp_loss_term:
            return gp_losses
    
    def _gen_loss(self):
        """ Implements the first half of the generator training step described on 
        lines 11 and 12 of algorithm 1 in https://arxiv.org/abs/1704.00028
        """
        
        latents = self.latent_sampler(self.batch_size)
        if self.gpu:
            latents = latents.cuda()
        x_gen = self.generator(latents)
        objective = -self.discriminator(x_gen)
        return objective

    def _disc_gp_loss(self, x_real: Tensor, x_gen: Tensor) -> Tensor:
        """ Computes the gradient penalty, as defined by equation 3 of
        Improved Training of Wasserstein GANs
        https://arxiv.org/abs/1704.00028
        """
        
        # Start by sampling some epsilon, used to interpolate between the real
        # data and generated data
        eps = torch.rand(self.batch_size, 1, 1)
        if self.gpu:
            eps = eps.cuda()

        x_gen = eps * x_real + (1 - eps) * x_gen
        x_gen.requires_grad_()  # Gradients are takes w.r.t the interpolated data
        prob_xgen = self.discriminator(x_gen)

        # Should have shape: (batch_size, n_mics, n_samples)
        grad_output = torch.ones_like(prob_xgen)
        if self.gpu:
            grad_output = grad_output.cuda()
        
        disc_grad = torch.autograd.grad(
            prob_xgen,
            x_gen,
            grad_output,
            create_graph=True  # Necessary to compute grads w.r.t model parameters later
        )[0]
        disc_grad.requires_grad_()

        # Take the norm along the channels and samples axes,
        # leaving a vector of shape (batch_size,)
        disc_grad = disc_grad.view(self.batch_size, -1)
        # Adding EPS for stability of the gradient
        grad_norm = torch.sqrt( torch.sum(disc_grad ** 2, dim=1) + EPS )
        gp_term = self.gp_coeff * (grad_norm - 1) ** 2
        loss = torch.squeeze(prob_xgen) - torch.squeeze(self.discriminator(x_real)) + gp_term
        return loss
