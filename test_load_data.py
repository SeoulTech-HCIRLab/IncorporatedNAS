from DataLoader import get_data
import os
import distutils.dir_util
import pprint, ast, argparse, logging
# import numpy as np
# import torch

if __name__ == '__main__':

    def parse_cmd_options(args):
        parser = argparse.ArgumentParser(description='Default command line parser.')
        parser.add_argument('--dataset', type=str, default='vgg-face2', help='name of the dataset')
        parser.add_argument('--mixup', type=bool, default=True)
        parser.add_argument('--random_erase',  type=bool, default=True)
        parser.add_argument('--auto_augment', type=bool, default=False)
        parser.add_argument('--no_data_augment', type=bool, default=False)
        parser.add_argument('--auto_augment_type', type=str, default='imagenet', help='type of auto_augment CIFA or imagenet')
        parser.add_argument('--shuffle_train_data', type=bool, default=True, help='shuffle sample data')

        parser.add_argument('--batch_size_per_gpu', default=64, type=int, help='batch size per GPU.')
        parser.add_argument('--auto_batch_size', action='store_true', help='allow adjust batch size smartly.')
        parser.add_argument('--num_cv_folds', type=int, default=None, help='Number of cross-validation folds.')
        parser.add_argument('--cv_id', type=int, default=None, help='Current ID of cross-validation fold.')
        parser.add_argument('--input_image_size', type=int, default=224, help='input image size.')
        parser.add_argument('--input_image_crop', type=float, default=0.875, help='crop ratio of input image')
        parser.add_argument('--independent_training', type=bool, default=True)
        parser.add_argument('--rank', type=int, default=1, help='crop ratio of input image')
        parser.add_argument('--world_size', type=int, default=1, help='crop ratio of input image'),
        parser.add_argument('--workers_per_gpu', type=int, default=6, help='crop ratio of input image'),
        parser.add_argument('--dataloader_testing', action='store_true', help='Testing data loader.')

        opt, _ = parser.parse_known_args(args)
        return opt


    # opt = {
    #     'dataset': 'vgg-face2',
    #     'batch_size_per_gpu': 64,
    #     'random_erase': True,
    #     'auto_augment': False,
    #     'input_image_size': 224,
    #     'input_image_crop': 0.875,
    #     'rank': 1,
    #     'world_size': 1,
    #     'workers_per_gpu': 6,
    #     'auto_augment_type': 'imagenet',
    #     'shuffle_train_data': True,
    #     'independent_training': True
    # }
    opt=parse_cmd_options(args=None)
    print(opt)
    data_vgg = get_data(opt, argv=None)
    print(data_vgg)