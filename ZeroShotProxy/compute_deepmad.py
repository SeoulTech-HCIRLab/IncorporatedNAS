# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.
# The DeepMAD score is from the paper https://arxiv.org/abs/2303.02165 of Alibaba.

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time
# from PlainNet import SuperResKXKX, SuperResK1KXK1, SuperResIDWEXKX

# from .builder import SCORES

# @SCORES.register_module(module_name = 'deepmad')
class ComputeDeepMadScore(metaclass=ABCMeta):

    def __init__(self, image_size = 224, multi_block_ratio = [1,1,1,1,8], ratio = 1, init_std = 1, init_std_act = 1, logger=None, **kwargs):
        self.init_std = init_std
        self.init_std_act = init_std_act
        self.resolution = image_size
        self.ratio_coef = multi_block_ratio
        self.ratio = ratio
        self.alpha1 = kwargs.get('alpha1', 1)
        self.alpha2 = kwargs.get('alpha2', 1)

        self.logger = logger or logging

    def ratio_score(self, stages_num, block_std_list):

        ## in nasscore don't care about ratio

        # if stages_num != len(self.ratio_coef):
        #     raise ValueError(
        #         'the length of the stage_features_list (%d) must be equal to the length of ratio_coef (%d)'
        #         % (stages_num, len(self.ratio_coef)))
        # self.logger.debug(
        #     'len of stage_features_list:%d, len of block_std_list:%d %s' %
        #     (stages_num, len(block_std_list), [std for std in block_std_list]))
        # self.logger.debug(
        #     'stage_idx:%s, stage_block_num:%s, stage_layer_num:%s' %
        #     (self.stage_idx, self.stage_block_num, self.stage_layer_num))

        nas_score_list = []
        for idx in range(stages_num):
            # if ratio == 0:
            #     nas_score_list.append(0.0)
            #     continue

            # compute std scaling
            nas_score_std = 0.0
            for idx1 in range(self.stage_block_num[idx]):
                # print("idx, idx1", (idx, idx1))
                nas_score_std += block_std_list[idx1]

            # larger channel and larger resolution, larger performance.
            resolution_stage = self.resolution // (2**(idx + 1))

            nas_score_feat = np.log(self.stage_channels[idx])
            nas_score_stage = nas_score_feat
            nas_score_stage = nas_score_stage + np.log(self.stage_feature_map_size[idx] ** 2)

            nas_score_stage = nas_score_stage * nas_score_std

            # print("stage:%d, mad_nas_score_stage:%.3f, score_feat:%.3f, log_std:%.3f, resolution:%d"%(
            #                     idx, nas_score_stage, nas_score_feat, nas_score_std, resolution_stage))
            self.logger.debug("stage:%d, mad_nas_score_stage:%.3f, score_feat:%.3f, log_std:%.3f, resolution:%d"%(
                                idx, nas_score_stage, nas_score_feat, nas_score_std, resolution_stage))
            nas_score_list.append(nas_score_stage)
        self.logger.debug("nas_score:%s"%(np.sum(nas_score_list)))
        # print(nas_score_list)

        return nas_score_list

    def __call__(self, model):
        info = {}
        timer_start = time.time()
        self.stage_idx, self.stage_block_num, self.stage_layer_num, self.stage_channels, self.stage_feature_map_size = model.get_stage_info(self.resolution)
        ##debug
        # print("debug of JinniPi for computing deepmad in ZenNAS")
        # print(model)
        # print("self.stage_idx", self.stage_idx)
        # print("self.stage_block_num", self.stage_block_num)
        # print("self.stage_layer_num", self.stage_layer_num)
        #
        # kwarg = {"alpha1":self.alpha1, "alpha2":self.alpha2}
        block_std_list = model.deepmad_forward_pre_GAP()
        nas_score_once = self.ratio_score(len(self.stage_idx), block_std_list)
        timer_end = time.time()
        #ratio = 1 for every blocks
        nas_score_once = np.array(nas_score_once)
        avg_nas_score = np.sum(nas_score_once)
        depth_penalty = model.get_depth_penalty(depth_penalty_ratio=10)
        avg_nas_score = avg_nas_score - depth_penalty
        info['avg_nas_score'] = avg_nas_score
        info['std_nas_score'] = avg_nas_score
        info['nas_score_list'] = nas_score_once
        info['depth_penalty'] = depth_penalty

        info['time'] = timer_end - timer_start
        self.logger.debug('avg_score:%s, consume time is %f ms' %
                          (avg_nas_score, info['time'] * 1000))

        return info['avg_nas_score']


def main():
    pass

#
# if __name__ == '__main__':
#     main()
#     pass
def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    parser.add_argument('--plainnet_struct', type=str, default=None, help='PlainNet structure string')
    parser.add_argument('--plainnet_struct_txt', type=str, default="/home/trangpi/Project/ZenNAS/ZeroShotProxy/init_plainnet_test2.txt", help='PlainNet structure file name')
    parser.add_argument('--num_classes', type=str, default=7, help='num_classes')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    import global_utils
    import Masternet
    from PlainNet import SuperResKXKX, SuperResK1KXK1
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)

    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=sys.argv, no_create=True)
    initial_structure_str = str(masternet)

    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=initial_structure_str,
                            no_create=False, no_reslink=True)
    # print(type(the_model))

    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)

    start_timer = time.time()
    print("the_model.get_efficient_score()", the_model.get_efficient_score(resolution=32))
    # the_deepmad_compute = ComputeDeepMadScore(image_size =224, multi_block_ratio = [1,1,1,1,1], ratio = 1, init_std = 1, init_std_act = 1, logger=None,
    #                                           kwargs={'alpha1': 1, 'alpha2': 1})
    # info = the_deepmad_compute(the_model)
    # time_cost = (time.time() - start_timer) / args.repeat_times
    # zen_score = info['avg_nas_score']
    # print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')