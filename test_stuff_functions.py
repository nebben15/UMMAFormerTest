import os
from collections import Counter

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import pandas as pd
import tempfile
import shutil
import subprocess
import torch
import torchaudio 
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def format_results_as_matrix(file_path):
    detection_data = []
    proposal_data = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("Detection:"):
            # Parse Detection results
            parts = line.split()
            detection_data.append({
                "average-mAP": float(parts[2]),
                "mAP@0.50": float(parts[4]),
                "mAP@0.55": float(parts[6]),
                "mAP@0.60": float(parts[8]),
                "mAP@0.65": float(parts[10]),
                "mAP@0.70": float(parts[12]),
                "mAP@0.75": float(parts[14]),
                "mAP@0.80": float(parts[16]),
                "mAP@0.85": float(parts[18]),
                "mAP@0.90": float(parts[20]),
                "mAP@0.95": float(parts[22]),
            })
        elif line.startswith("Proposal:"):
            # Parse Proposal results
            parts = line.split()
            proposal_data.append({
                "AR@10": float(parts[2]),
                "AR@20": float(parts[4]),
                "AR@50": float(parts[6]),
                "AR@100": float(parts[8]),
            })

    # Create DataFrames for Detection and Proposal
    detection_df = pd.DataFrame(detection_data)
    proposal_df = pd.DataFrame(proposal_data)

    return detection_df, proposal_df


def show_vid_with_segments(video_id, segments, scores, cfg, threshold=0.9):
    # Find all segments with scores above the threshold
    highlight_indices = np.where(scores > threshold)[0]
    highlight_segments = [segments[i] for i in highlight_indices]

    # Find the video file path (search in train, test, valid subfolders)
    video_root = cfg['dataset']['feat_folder'].replace('feats/tsn', 'videos')
    video_path = None
    for split in ['train', 'test', 'valid']:
        candidate = os.path.join(video_root, split, f"{video_id}.mp4")
        if os.path.isfile(candidate):
            video_path = candidate
            break
    if video_path is None:
        raise FileNotFoundError(f"Video file not found in train/test/valid: {video_id}.mp4")

    # Open the video to get fps and frame count
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Prepare video path relative to notebook root for HTML
    video_rel_path = video_path.lstrip('./') if video_path.startswith('./') else video_path

    # Prepare highlight bars for the progress bar
    highlight_divs = ""
    for seg in highlight_segments:
        start_time, end_time = seg
        seg_start_frac = (start_time / duration) * 100
        seg_end_frac = (end_time / duration) * 100
        bar_left = seg_start_frac
        bar_width = seg_end_frac - seg_start_frac
        highlight_divs += f"""
        <div style="
            position: absolute;
            left: {bar_left}%;
            width: {bar_width}%;
            top: 0;
            bottom: 0;
            background: red;
            opacity: 0.5;
            pointer-events: none;
        "></div>
        """

    # HTML and JS for video with custom progress bar
    html = f"""
    <div style="position: relative; width: 640px;">
        <video id="vid" width="640" controls style="display: block;">
            <source src="{video_rel_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <div id="progressbar-container" style="position: relative; width: 640px; height: 10px; background: #eee; margin-top: 2px;">
            {highlight_divs}
            <div id="progressbar-current" style="position: absolute; left: 0; top: 0; bottom: 0; width: 0%; background: #2196F3; opacity: 0.8;"></div>
        </div>
    </div>
    <script>
    (function() {{
        var video = document.getElementById('vid');
        var progress = document.getElementById('progressbar-current');
        var container = document.getElementById('progressbar-container');
        video.addEventListener('timeupdate', function() {{
            var percent = 100 * video.currentTime / video.duration;
            progress.style.width = percent + '%';
        }});
        // Allow clicking on the progress bar to seek
        container.addEventListener('click', function(e) {{
            var rect = container.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var percent = x / rect.width;
            video.currentTime = percent * video.duration;
        }});
    }})();
    </script>
    """
    display(HTML(html))

def show_vid(video_id, cfg):
    # Find the video file path (search in train, test, valid subfolders)
    video_root = cfg['dataset']['feat_folder'].replace('feats/tsn', 'videos')
    video_path = None
    for split in ['train', 'test', 'valid']:
        candidate = os.path.join(video_root, split, f"{video_id}.mp4")
        if os.path.isfile(candidate):
            video_path = candidate
            break
    if video_path is None:
        print(video_path)
        raise FileNotFoundError(f"Video file not found in train/test/valid: {video_id}.mp4")

    # Prepare video path relative to notebook root for HTML
    video_rel_path = video_path.lstrip('./') if video_path.startswith('./') else video_path

    # HTML for video with audio (audio is included by default in <video> tag if present in the file)
    html = f"""
    <div style="position: relative; width: 640px;">
        <video width="640" controls style="display: block;">
            <source src="{video_rel_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    display(HTML(html))

def get_sample_by_video_id(dataset, video_id):
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['video_id'] == video_id:
            return [sample]
    return None

def human_readable_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    elif num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"
    
def get_nparray_size(data):
    data_nbytes = data.nbytes
    size_str = human_readable_size(data_nbytes)
    return size_str

def get_mp4_size(path):
    """
    Returns the size of the mp4 video file (with audio) and the size without audio (video only),
    both as human-readable strings (e.g., KB, MB).
    Args:
        path (str): Path to the mp4 file.
    Returns:
        tuple: (size_with_audio_str, size_without_audio_str)
    """
    # Size with audio
    size_with_audio = os.path.getsize(path)

    # Remove audio using ffmpeg and save to a temp file
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "video_no_audio.mp4")
    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-c", "copy", "-an", temp_video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        size_without_audio = os.path.getsize(temp_video_path)
    finally:
        shutil.rmtree(temp_dir)

    return human_readable_size(size_with_audio), human_readable_size(size_without_audio)

def count_array_dims(folder_path):
    """
    Counts each dimensionality (shape) for all .npy arrays in a folder,
    prints the results ordered by number of occurrences (descending),
    counts the number of corrupted files, and returns a list of corrupted file names.
    """
    dim_counter = Counter()
    corrupted_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.npy'):
            fpath = os.path.join(folder_path, fname)
            try:
                arr = np.load(fpath)
                dim_counter[arr.shape] += 1
            except Exception as e:
                corrupted_files.append(fname)
    for shape, count in dim_counter.most_common():
        print(f"Shape {shape}: {count} files")
    print(f"\nNumber of corrupted files: {len(corrupted_files)}")
    if corrupted_files:
        print("Corrupted files:")
        for fname in corrupted_files:
            print(fname)
    return corrupted_files

def analyze_npy_folder(folder_path):
    """
    Given a folder path, calculates:
      - mean and median of the sizes (in bytes) of the .npy arrays in the folder
      - average of the first dimension (x) assuming all arrays are (x, y)
    Also visualizes the distributions of x and array size as histograms,
    and prints the mean and median in the visualization (with human-readable units).
    Ignores corrupted files.
    """
    sizes = []
    x_dims = []
    corrupted_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.npy'):
            try:
                arr = np.load(os.path.join(folder_path, fname))
                sizes.append(arr.nbytes)
                if arr.ndim == 2:
                    x_dims.append(arr.shape[0])
            except Exception as e:
                corrupted_files.append(fname)
    if not sizes:
        print("No valid .npy files found.")
        return None

    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    avg_x = np.mean(x_dims) if x_dims else None
    mean_x = np.mean(x_dims) if x_dims else None
    median_x = np.median(x_dims) if x_dims else None

    # Plot histogram for array sizes (in MB)
    sizes_mb = [s / 1024**2 for s in sizes]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sizes_mb, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(mean_size / 1024**2, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {human_readable_size(mean_size)}')
    plt.axvline(median_size / 1024**2, color='green', linestyle='dashed', linewidth=1, label=f'Median: {human_readable_size(median_size)}')
    plt.xlabel('Array Size (MB)')
    plt.ylabel('Count')
    plt.title('Distribution of Array Sizes')
    plt.legend()

    # Plot histogram for x dimension
    plt.subplot(1, 2, 2)
    plt.hist(x_dims, bins=30, color='orange', edgecolor='black')
    plt.axvline(mean_x, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_x:.1f}')
    plt.axvline(median_x, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_x:.1f}')
    plt.xlabel('x Dimension')
    plt.ylabel('Count')
    plt.title('Distribution of x Dimension')
    plt.legend()

    plt.tight_layout()
    plt.show()

    if corrupted_files:
        print(f"Ignored {len(corrupted_files)} corrupted file(s).")

    print(f"Mean array size: {human_readable_size(mean_size)}")
    print(f"Median array size: {human_readable_size(median_size)}")

    return {
        'mean_size_bytes': mean_size,
        'median_size_bytes': median_size,
        'average_x_dim': avg_x,
        'corrupted_files': corrupted_files
    }

def get_folder_size(folder_path):
    return human_readable_size(sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(folder_path)
        for f in files
        if os.path.isfile(os.path.join(dirpath, f)))
    )

def count_files_of_type(folder_path, extension):
    """
    Counts the total number of files with a given extension in a folder (including subfolders).
    Example: count_files_of_type('data/lavdf/byola/test', '.npy')
    """
    count = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if fname.endswith(extension):
                count += 1
    return count

def process_single_file(
    file_path,
    sample_rate,
    n_fft,
    hop_length,
    model_downsample_factor,
):
    try:
        waveform, sr = torchaudio.load(file_path)

        # Convert multi-channel to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        num_samples = waveform.shape[1]

        # Calculate number of STFT frames
        num_frames = 1 + (num_samples - n_fft) // hop_length
        num_frames = max(num_frames, 0)

        # Downsample for model output sequence length
        feature_seq_len = max(num_frames // model_downsample_factor, 1)

        return feature_seq_len
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def average_feature_sequence_length(
    root_folder,
    sample_rate=16000,
    n_fft=1024,
    hop_length=160,
    model_downsample_factor=8,
    max_workers=8,
):
    """
    Recursively calculates average sequence length of BYOL-A features
    extracted from all .wav files under root_folder using parallel processing.

    Args:
        root_folder (str): Root folder path containing .wav files (and subfolders).
        sample_rate (int): Audio sample rate (Hz).
        n_fft (int): FFT window length (samples).
        hop_length (int): Hop length between FFT windows (samples).
        model_downsample_factor (int): Downsampling factor from CNN (default 8).
        max_workers (int): Number of parallel workers.

    Returns:
        float: Average sequence length of features after model downsampling.
    """
    # Collect all wav files recursively
    wav_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file_name in filenames:
            if file_name.lower().endswith('.wav'):
                wav_files.append(os.path.join(dirpath, file_name))

    sequence_lengths = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file_path,
                sample_rate,
                n_fft,
                hop_length,
                model_downsample_factor,
            ): file_path for file_path in wav_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
            seq_len = future.result()
            if seq_len is not None:
                sequence_lengths.append(seq_len)

    if not sequence_lengths:
        return 0.0

    avg_length = sum(sequence_lengths) / len(sequence_lengths)
    return avg_length

def contraction_ratio(original_size_str, feature_size_str):
    """
    Calculates the contraction ratio (feature_size / original_size) given two human-readable size strings.
    Returns the ratio as a float and a formatted string (e.g., "0.25 (4x smaller)").
    """
    def parse_size(size_str):
        size_str = size_str.strip().upper()
        if size_str.endswith('BYTES'):
            return float(size_str.replace('BYTES', '').strip())
        elif size_str.endswith('KB'):
            return float(size_str.replace('KB', '').strip()) * 1024
        elif size_str.endswith('MB'):
            return float(size_str.replace('MB', '').strip()) * 1024**2
        elif size_str.endswith('GB'):
            return float(size_str.replace('GB', '').strip()) * 1024**3
        else:
            raise ValueError(f"Unknown size format: {size_str}")

    orig_bytes = parse_size(original_size_str)
    feat_bytes = parse_size(feature_size_str)
    if orig_bytes == 0:
        return None, "Original size is zero"
    ratio = feat_bytes / orig_bytes
    contraction = f"{ratio:.3f} ({1/ratio:.1f}x smaller)" if ratio < 1 else f"{ratio:.3f} ({ratio:.1f}x larger)"
    return ratio, contraction