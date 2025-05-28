import argparse
import glob
import os
import os.path as osp
from multiprocessing import Pool
from tqdm import tqdm
import subprocess

def extract_audio_wav(line):
    video_id, _ = osp.splitext(osp.basename(line))
    video_dir = osp.dirname(line)
    video_rel_dir = osp.relpath(video_dir, args.root)
    dst_dir = osp.join(args.dst_root, video_rel_dir)
    os.makedirs(dst_dir, exist_ok=True)
    try:
        out_path = f'{dst_dir}/{video_id}.wav'
        if osp.exists(out_path):
            return
        cmd = [
            'ffmpeg', '-i', line, '-map', '0:a', '-y', out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except BaseException:
        with open('extract_wav_err_file.txt', 'a+') as f:
            f.write(f'{line}\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Extract audios')
    parser.add_argument('root', type=str, help='source video directory')
    parser.add_argument('dst_root', type=str, help='output audio directory')
    parser.add_argument('--level', type=int, default=2, help='directory level of data')
    parser.add_argument('--ext', type=str, default='mp4', choices=['avi', 'mp4', 'webm'], help='video file extensions')
    parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
    args = parser.parse_args()
    return args

def initializer(global_args):
    global args
    args = global_args

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.dst_root, exist_ok=True)
    print('Reading videos from folder: ', args.root)
    print('Extension of videos: ', args.ext)
    pattern = args.root + '/*' * args.level + '.' + args.ext
    fullpath_list = glob.glob(pattern)
    done_fullpath_list = glob.glob(args.dst_root + '/*' * args.level + '.wav')
    print('Total number of videos found: ', len(fullpath_list))
    print('Total number of videos extracted finished: ', len(done_fullpath_list))

    with Pool(args.num_workers, initializer, (args,)) as pool:
        list(tqdm(pool.imap_unordered(extract_audio_wav, fullpath_list), total=len(fullpath_list)))