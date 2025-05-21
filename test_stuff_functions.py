import os

import cv2
import numpy as np
from IPython.display import display, HTML
import pandas as pd

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