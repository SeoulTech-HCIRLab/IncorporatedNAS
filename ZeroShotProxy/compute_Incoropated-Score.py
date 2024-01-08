'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''



import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, ModelLoader, time
from compute_deepmad import ComputeDeepMadScore

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net

def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            output = model.forward_pre_GAP(input)
            mixup_output = model.forward_pre_GAP(mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
            nas_score = torch.mean(nas_score)

            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass
            nas_score = torch.log(nas_score) + log_bn_scaling_factor
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info

def compute_Combine(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    the_nas_core_info = compute_nas_score(model=model, gpu=gpu, resolution=resolution,mixup_gamma=mixup_gamma,
                                          batch_size=batch_size, repeat=repeat)
    the_nas_core = the_nas_core_info['avg_nas_score']

    the_compute_deepmad = ComputeDeepMadScore(image_size=resolution, multi_block_ratio=[1, 1, 1, 1, 1],
                                              ratio=1, init_std=1, init_std_act=1, logger=None,
                                              kwargs={'alpha1': 1, 'alpha2': 1})
    score_deepmad = the_compute_deepmad(the_model)
    the_nas_core = 0.9 * the_nas_core + 0.1 * np.sqrt(score_deepmad)

    return the_nas_core


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    parser.add_argument('--num_classes', type=str, default=100, help='num_classes')

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
    print(the_model)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_Combine(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info
    print(f'Izen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')

