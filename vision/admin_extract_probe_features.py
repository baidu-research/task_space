import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate, to_tensor, normalize
import numpy as np
from PIL import Image
from utils import use_gpu, load_ckpt_for_feature_extraction, extract
import pylab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=str)
    parser.add_argument('split', type=str,
                        help='path to an .npz of admin and agent data ids')
    parser.add_argument('-r', '--rotation', type=float, nargs='+',
                        help='one or multiple degrees to rotate the probe '
                        'images. Default to no rotation')
    parser.add_argument('save', type=str,
                        help='path to save the features.')
    return parser.parse_args()


def rotate_npy_img(np_img, angle):
    """
        np_img: numpy array: n_images x height x width x channel
    """
    pil_img = Image.fromarray(np_img)

    return rotate(pil_img, angle)


def probing_dataLoader(probe_ids, mean_std, batch_size=128, rot=None):
    """
        Loader for probing samples
        Parameters:

        probe_ids: index of probe data to read
        mean_std: mean and std to normalize the images
        batch_size: batch size
        rot: a list of degrees to rotate the image. Default to no rotation

        Return:
        An instance of DataLoader
    """
    full_set = torchvision.datasets.CIFAR100('images', True)
    probes = full_set.data[probe_ids]

    # rotate each image for specified degrees
    if rot is None:
        rot = [0]
    elif 0 not in rot:
        rot.insert(0, 0)

    batch = []
    for im in probes:
        for degree in rot:
            rot_probe = rotate_npy_img(im, degree)
            rot_probe = to_tensor(rot_probe)
            normalize_probe = normalize(rot_probe, mean_std[0], mean_std[1])
            batch.append(normalize_probe)
            if len(batch) == batch_size:
                yield torch.stack(batch)
                batch = []
    if len(batch) > 0:
        yield torch.stack(batch)


if __name__ == '__main__':
    args = parse_args()
    split_file = np.load(args.split)
    probe_ids = split_file['admin_probe_ids']
    net = load_ckpt_for_feature_extraction(args.ckpt_dir)
    mean_std = np.load(os.path.join(args.ckpt_dir, 'mean_std.npz'))
    mean_std = (mean_std['mean'], mean_std['std'])
    X_train, _ = extract(net,
                         probing_dataLoader(probe_ids, mean_std,
                                            rot=args.rotation)
                        )
    np.save(args.save, X_train)
