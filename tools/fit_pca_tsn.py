import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
import glob
from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model

from sklearn.decomposition import PCA
import pickle


def fit_pca_tsn():
    args = parse_args()
    # Set up modality-specific parameters
    args.is_rgb = args.modality == 'RGB'
    args.clip_len = 1 if args.is_rgb else 5
    args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)
    args.batch_size = 200  # max batch_size for one forward

        # Define the data pipeline for untrimmed videos
    data_pipeline = [
        dict(
            type='UntrimmedSampleFrames',
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            start_index=0),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # Define TSN R50 model as the feature extractor (no classification head)
    model_cfg = dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet',
            depth=50,
            in_channels=args.in_channels,
            norm_eval=False),
        cls_head=None,
        test_cfg=dict(average_clips=None,feature_extraction=True))
    model = build_model(model_cfg)

    # Load pretrained weights into the feature extractor
    state_dict = torch.load(args.ckpt)['state_dict']
    keys = list(model.state_dict().keys())
    new_dict = {k: state_dict[k] for k in keys}
    model.load_state_dict(new_dict)
    model = model.cuda()
    model.eval()

    # Read the list of videos to process
    data = open(args.data_list).readlines()
    data = [os.path.splitext(os.path.basename(x.strip()))[0] for x in data]
    data = data[args.part::args.total]

    # Progress bar for feature extraction
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    # feature extraction
    feats = []
    for item in data:
        frame_dir = item
        output_file = osp.basename(frame_dir) + '.npy'
        frame_dir = osp.join(args.data_prefix, frame_dir)
        length = len(glob.glob(os.path.join(frame_dir,'img_*.jpg' if args.is_rgb else 'flow_x_*.jpg')))
        output_file = osp.join(args.output_prefix, output_file)
        if  osp.exists(output_file):
            prog_bar.update()
            continue
        length = int(length)

        # Prepare a pseudo sample for the pipeline
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            filename_tmpl=args.f_tmpl,
            start_index=0,
            modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']
        shape = imgs.shape
        # Reshape for model input
        imgs = imgs.reshape((shape[0], 1) + shape[1:])
        imgs = imgs.cuda()

        def forward_data(model, data):
            # Chop large data into pieces and extract feature from them
            results = []
            start_idx = 0
            num_clip = data.shape[0]
            while start_idx < num_clip:
                with torch.no_grad():
                    part = data[start_idx:start_idx + args.batch_size]
                    feat = model.forward(part, return_loss=False)
                    results.append(feat)
                    start_idx += args.batch_size
            return np.concatenate(results)

        feat = forward_data(model, imgs)
        feats.append(feats)

    # fit pca
    all_feats = np.concatenate(feats, axis=0)  # shape: [total_samples, D]
    pca = PCA(n_components=args.pca_dim)
    pca.fit(all_feats)
    with open(args.pca_save_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"PCA fitted and saved to {args.pca_save_path}")

def parse_args():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Create PCA for TSN extraction.')
    parser.add_argument('--data-prefix', default='', help='dataset prefix')
    parser.add_argument(
        '--data-list',
        help='video list of the dataset, the format should be '
        '`frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=1,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument(
        '--total', type=int, default=1, help='how many parts exist')
    parser.add_argument('--pca-dim', type=int, required=True, help='Number of PCA components')
    parser.add_argument('--pca-save-path', type=str, required=True, help='Where to save the fitted PCA .pkl')
    parser.add_argument('--pca-fit-samples', type=int, default=10000, help='Number of feature vectors to use for PCA fitting')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    fit_pca_tsn()