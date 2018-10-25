#-*- coding: utf-8 -*-
import argparse


parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')

# Environment
env_arg = add_argument_group('Environment')
env_arg.add_argument('--num_bins', type=int, default=5, help='number of bins')
env_arg.add_argument('--num_slots', type=int, default=5, help='number of slots per bin')
env_arg.add_argument('--num_descriptors', type=int, default=8, help='number of unique packets')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=64, help='agent LSTM num_neurons')
net_arg.add_argument('--num_stacks', type=int, default=3, help='agent LSTM num_stacks')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=128, help='batch size')
data_arg.add_argument('--min_length', type=int, default=2, help='minimum chain length')
data_arg.add_argument('--max_length', type=int, default=6, help='maximum chain length')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--num_epoch', type=int, default=6000, help='number of epochs')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='agent learning rate')
#train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='lr decay step')
#train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='lr decay rate')

# Performance
perf_arg = add_argument_group('Training')
perf_arg.add_argument('--enable_performance', type=str2bool, default=False, help='compare performance agains Gecode solver')

# Misc
misc_arg = add_argument_group('User options')

misc_arg.add_argument('--train_mode', type=str2bool, default=False, help='switch between training and testing')
misc_arg.add_argument('--save_model', type=str2bool, default=False, help='whether or not model is loaded')
misc_arg.add_argument('--load_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='save/model', help='saver sub directory')
misc_arg.add_argument('--load_from', type=str, default='save/model', help='loader sub directory')
misc_arg.add_argument('--log_dir', type=str, default='summary/repo', help='summary writer log directory')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


if __name__ == "__main__":
    
    config, _ = get_config()
