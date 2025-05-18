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