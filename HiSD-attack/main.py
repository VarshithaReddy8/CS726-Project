from core.utils import get_config
import argparse
import sys
import os
from data_loader import get_loader
from solver import Solver



def main(config):
	if not os.path.exists(config.results_dir):
		os.makedirs(config.results_dir)

	dataloader = get_loader(config.dataset_dir)
	solver = Solver(dataloader, config)

	if config.c_dim is not len(config.selected_attrs):
		print("Select correct attribubtes")
		return 0

	if config.mode=='test':
		solver.test()
	elif config.mode=='test_attack':
		solver.test_attack()
	else:
		print("Select mode")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='test_attack', choices=['test', 'test_attack'])
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--reference_images', type=str, default='reference_images')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--c_dim', type=int, default=2)
    parser.add_argument('--selected_attrs', '--list', nargs='+', default=['bangs', 'glasses'])
    
    config = parser.parse_args()
    main(config)