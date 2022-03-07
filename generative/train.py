import argparse
import glob
import json
import logging
import os
from os import path
from typing import NewType

import numpy as np

from util import dataloaders as loaders, models


DEFAULT_CONFIG_PATH = 'configs/defaults.json'
JSON = NewType('JSON', dict)

def get_args():
    """ Gets arguments for the training run from the command line
    Current args:
    job_id (int): Determines where model parameters and training logs will be stored
    config (path): Location of the config file used to run the model (optional)
    hdf_path (path): Location of hdf file (Containing a contiguous array of vocalizations)
    """
    parser = argparse.ArgumentParser(description="A generative model for gerbil vocalizations")

    parser.add_argument(
        'job_id',
        type=int
    )

    parser.add_argument(
        '--config_path',
        type=str,
        required=False
    )

    parser.add_argument(
        '--hdf_path',
        type=str,
        required=False
    )

    args = parser.parse_args()

    verify_args(args)
    return args


def verify_args(args):
    if args.job_id < 1:
        raise ValueError(f'Job id must be a positive integer, received {args.job_id}')
    if (args.config_path is not None) and (not path.exists(args.config_path)):
        raise ValueError(f'Could not find requested config file {args.config_path}')



def find_data(args, config):
    """ Given command-line arguments and a config file, locates data,
    which may be specified in either.
    """
    if args.hdf_path is None:
        # Locate files within config
        if 'hdf_path' not in config:
            raise ValueError('Path to hdf5 dataset' \
                'should be provided as a command-line argument' \
                'or an entry in the config file.')
        return config['hdf_path']
    else:
        return args.hdf_path


def populate_config(config: JSON):
    """ Populates missing fields in `config` with those from the default
    config.
    """
    with open(DEFAULT_CONFIG_PATH, 'r') as ctx:
        default_config = json.load(ctx)
    
    for k,v in default_config.items():
        if k not in config:
            config[k] = v
    
    return config


def train(config, model, dataloader, checkpoint_dir, sample_dir):
    logging.info('Starting training:')
    n_epochs = config['num_epochs']
    # How often checkpoints should be saved
    checkpoint_freq = config['checkpoint_frequency']
    # Keep the latents used for visualization constant to see how
    # the model's generator evolves for the same input
    fixed_latents = model.latent_sampler(64)
    for n in range(n_epochs):
        logging.info(f'Training: Epoch {n + 1} of {n_epochs}')
        data_iter = iter(dataloader)
        
        losses, epoch_complete = model.train_minibatch(data_iter)
        minibatch_no = 0
        while not epoch_complete:
            logging.info(f'Discriminator losses (minibatch {minibatch_no + 1} of {len(data_iter)}')
            fmt_losses = ['{:.5f}'.format(loss) for loss in losses['critic_losses']]
            logging.info(fmt_losses)
            logging.info(f'Generator losses (minibatch {minibatch_no + 1} of {len(data_iter)}')
            logging.info('{:.5f}'.format(losses['generator_loss']))
            minibatch_no += model.critic_steps
            losses, epoch_complete = model.train_minibatch(data_iter)
            
        # Generate some samples to see how we are doing
        sample_fname = 'epoch_{:0>3d}_gen_sample.npy'.format(n + 1)
        sample_path = path.join(sample_dir, sample_fname)
        gen_sample = model.sample_from_generator(64, latent=fixed_latents)
        np.save(sample_path, gen_sample)

        # Save model state occasionally
        if (n + 1) % checkpoint_freq == 0:
            ckpt_fname = 'checkpoint_epoch_{:0>3d}.pt'.format(n + 1)
            ckpt_path = path.join(checkpoint_dir, ckpt_fname)
            model.save_checkpoint(ckpt_path)


def run():
    args = get_args()

    if args.config_path is None:
        config = dict()
    else:
        with open(args.config_path, 'r') as ctx:
            config = json.load(ctx)
    # This should be a mutating function, so the assignment might not be necessary
    config = populate_config(config)
    hdf_path = find_data(args, config)
    
    # Directory where everything related to this model will be saved
    model_dir = path.join('logs', '{:0>4d}'.format(args.job_id))
    checkpoint_dir = path.join(model_dir, 'checkpoints')
    generated_sample_dir = path.join(model_dir, 'generated_audio')
    if path.exists(model_dir):
        raise ValueError(f'Job id {args.job_id} is taken')
    os.makedirs(model_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(generated_sample_dir)

    train_log_path = path.join(model_dir, 'train.log')
    logging.basicConfig(
        filename=train_log_path,
        # encoding='utf-8',  # Apparently this argumnet doesn't exist? But it's there in the docs?
        level=logging.DEBUG
    )
    
    dataloader = loaders.create_dataloader(
        hdf_path,
        batch_size=config['batch_size']
    )

    model_wrapper = models.GerbilizerGAN(config)
    logging.info('Built models:')
    logging.info(str(model_wrapper.generator))
    logging.info(str(model_wrapper.discriminator))

    train(config, model_wrapper, dataloader, checkpoint_dir, generated_sample_dir)


if __name__ == '__main__':
    run()
