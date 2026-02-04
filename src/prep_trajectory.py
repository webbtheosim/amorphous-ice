import numpy as np
import re
import os
import json
from collections import defaultdict
from datetime import datetime

from config import scan_paths, mbpol_paths

import functools
print = functools.partial(print, flush=True)

# Global report file handle
report_file = None

def log_to_report(message):
    print(message)
    if report_file:
        report_file.write(message + '\n')
        report_file.flush()

def log_verbose(message):
    if report_file:
        report_file.write(message + '\n')
        report_file.flush()

def count_frames_in_file(path):
    if not os.path.exists(path):
        return 0

    try:
        with open(path, 'r') as f:
            frame_count = 0
            for line in f:
                if 'TIMESTEP' in line:
                    frame_count += 1
        return frame_count
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0

def analyze_files(file_paths):
    """First pass: analyze all files to understand available data."""
    file_info = {}

    log_to_report("=== FILE ANALYSIS PHASE ===")
    for state, file_set in file_paths.items():
        print(f"Analyzing {state} phase files...")
        log_verbose(f"\nAnalyzing {state} phase files...")
        file_info[state] = []

        for file_idx, filename in enumerate(file_set):
            if type(filename) is list:
                path = filename[0]
                limits = (filename[1], filename[2])
            else:
                path = filename
                limits = (-1, -1)

            # Count total frames
            total_frames = count_frames_in_file(path)

            # Calculate available frames after applying limits
            if limits[0] == -1:
                start_frame = 0
            else:
                start_frame = max(0, limits[0])

            if limits[1] == -1:
                end_frame = total_frames
            else:
                end_frame = min(total_frames, limits[1])

            available_frames = max(0, end_frame - start_frame)

            print({
                'index': file_idx,
                'path': path,
                'limits': limits,
                'total_frames': total_frames,
                'available_frames': available_frames,
                'exists': os.path.exists(path)
            })
            file_info[state].append({
                'index': file_idx,
                'path': path,
                'limits': limits,
                'total_frames': total_frames,
                'available_frames': available_frames,
                'exists': os.path.exists(path)
            })

            status = "Yes" if os.path.exists(path) else "No"
            log_verbose(f"  {status} File {file_idx}: {available_frames:,} frames available ({path})")

        total_available = sum(f['available_frames'] for f in file_info[state] if f['exists'])
        print(f"   Found {len([f for f in file_info[state] if f['exists']])} valid files, {total_available:,} total frames")

    return file_info

def extract_temperature_from_path(path):
    # For LDA quench files - no temperature
    if 'lda_quench' in path:
        return 'quench'

    # For LDA compression files - temperature in filename (lda_compression_60_1.dump)
    if 'lda_compression' in path:
        temp_match = re.search(r'lda_compression_(\d+)_', path)
        if temp_match:
            return int(temp_match.group(1))

    # For liquid files - temperature in filename (production_liquid_220.dump)
    if 'production_liquid' in path:
        temp_match = re.search(r'production_liquid_(\d+)', path)
        if temp_match:
            return int(temp_match.group(1))

    # For ice files - temperature in filename (production_IceIh_60.dump)
    if 'production_IceIh' in path:
        temp_match = re.search(r'production_IceIh_(\d+)', path)
        if temp_match:
            return int(temp_match.group(1))

    # For HDA files - temperature in directory structure (/60/, /80/, etc.)
    temp_match = re.search(r'/(\d+)/', path)
    if temp_match:
        return int(temp_match.group(1))

    return None

def group_files_by_temperature(files, state):
    temp_groups = {}

    for file_info in files:
        temp = extract_temperature_from_path(file_info['path'])
        if temp is not None:
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(file_info)
        else:
            # If we can't extract temperature, put in a generic group
            if 'unknown' not in temp_groups:
                temp_groups['unknown'] = []
            temp_groups['unknown'].append(file_info)

    return temp_groups

def calculate_lda_distribution(valid_files, target_per_state):
    log_verbose(f" Using LDA-specific distribution logic")

    # Group files
    temp_groups = group_files_by_temperature(valid_files, 'lda')

    # Separate quench files from temperature files
    quench_files = temp_groups.get('quench', [])
    temp_files = {k: v for k, v in temp_groups.items() if k != 'quench' and k != 'unknown'}
    unknown_files = temp_groups.get('unknown', [])

    log_verbose(f"Found {len(quench_files)} quench files")
    log_verbose(f"Found {len(temp_files)} temperature groups: {sorted(temp_files.keys())}")
    if unknown_files:
        log_verbose(f"Found {len(unknown_files)} unknown files")

    # Calculate target distributions
    quench_target = int(target_per_state * 0.25)  # 25% for quench files
    temp_target_total = target_per_state - quench_target  # 75% for temperature files

    if len(temp_files) > 0:
        temp_target_per_group = temp_target_total // len(temp_files)
        temp_remainder = temp_target_total % len(temp_files)
    else:
        temp_target_per_group = 0
        temp_remainder = 0

    log_verbose(f"Target distribution:")
    log_verbose(f"      Quench files: {quench_target:,} frames (25%)")
    log_verbose(f"      Per temperature: {temp_target_per_group:,} frames")

    file_distributions = []

    # Process quench files
    if quench_files:
        quench_available = sum(f['available_frames'] for f in quench_files)
        quench_actual = min(quench_target, quench_available)

        log_verbose(f"    🧊 Quench group: {quench_actual:,} frames from {len(quench_files)} files")

        for i, file_info in enumerate(quench_files):
            if i == len(quench_files) - 1:  # Last quench file gets remainder
                frames_to_take = quench_actual - sum(f['frames_to_extract'] for f in file_distributions if f['temperature'] == 'quench')
            else:
                frames_to_take = quench_actual // len(quench_files)

            frames_to_take = max(0, min(frames_to_take, file_info['available_frames']))

            file_distributions.append({
                'file_index': file_info['index'],
                'path': file_info['path'],
                'limits': file_info['limits'],
                'available_frames': file_info['available_frames'],
                'frames_to_extract': frames_to_take,
                'temperature': 'quench',
                'proportion': frames_to_take / target_per_state if target_per_state > 0 else 0
            })

            log_verbose(f"      File {file_info['index']}: {frames_to_take:,} frames")

    # Process temperature files
    for temp_idx, (temp, temp_files_list) in enumerate(sorted(temp_files.items())):
        # Add remainder to first few temperature groups
        temp_target = temp_target_per_group + (1 if temp_idx < temp_remainder else 0)

        temp_available = sum(f['available_frames'] for f in temp_files_list)
        temp_actual = min(temp_target, temp_available)

        log_verbose(f"{temp}K: {temp_actual:,} frames from {len(temp_files_list)} files")

        for i, file_info in enumerate(temp_files_list):
            if i == len(temp_files_list) - 1:  # Last file in temp group gets remainder
                frames_to_take = temp_actual - sum(f['frames_to_extract'] for f in file_distributions if f['temperature'] == temp)
            else:
                frames_to_take = temp_actual // len(temp_files_list)

            frames_to_take = max(0, min(frames_to_take, file_info['available_frames']))

            file_distributions.append({
                'file_index': file_info['index'],
                'path': file_info['path'],
                'limits': file_info['limits'],
                'available_frames': file_info['available_frames'],
                'frames_to_extract': frames_to_take,
                'temperature': temp,
                'proportion': frames_to_take / target_per_state if target_per_state > 0 else 0
            })

            log_verbose(f"      File {file_info['index']}: {frames_to_take:,} frames")

    return file_distributions

def calculate_optimal_distribution(file_info, target_per_state):
    """Calculate optimal frame distribution with phase-specific temperature balancing."""
    distribution = {}

    log_to_report("\n=== DISTRIBUTION CALCULATION ===")
    for state, files in file_info.items():
        print(f"Calculating {state} distribution...")
        log_verbose(f"\nCalculating distribution for {state} phase:")

        # Filter out files with no available frames
        valid_files = [f for f in files if f['exists'] and f['available_frames'] > 0]

        if not valid_files:
            print(f"No valid files found!")
            log_verbose(f"No valid files found for {state} phase!")
            distribution[state] = []
            continue

        total_available = sum(f['available_frames'] for f in valid_files)

        if total_available < target_per_state:
            print(f"Only {total_available:,} frames available (requested {target_per_state:,})")
            log_verbose(f"Only {total_available:,} frames available, but {target_per_state:,} requested")
            log_verbose(f"Using all available frames")
            target_per_state_actual = total_available
        else:
            target_per_state_actual = target_per_state

        # Use phase-specific distribution logic
        if state.lower() == 'lda':
            file_distributions = calculate_lda_distribution(valid_files, target_per_state_actual)
        else:
            # Standard temperature-based distribution for HDA, liquid, ice phases
            temp_groups = group_files_by_temperature(valid_files, state)

            if len(temp_groups) <= 1:
                log_verbose(f"No temperature grouping detected, using file-based balancing")
                temp_groups = {'all': valid_files}
            else:
                temps = [t for t in temp_groups.keys() if t != 'unknown' and t != 'quench']
                log_verbose(f"Using temperature-based balancing")
                log_verbose(f"Found {len(temp_groups)} temperature groups: {sorted(temps)}")

            # Calculate frames per temperature group (hierarchical balancing)
            frames_per_temp = target_per_state_actual // len(temp_groups)
            remainder_frames = target_per_state_actual % len(temp_groups)

            log_verbose(f"Target: {frames_per_temp:,} frames per temperature group")

            file_distributions = []

            for temp_idx, (temp, temp_files) in enumerate(sorted(temp_groups.items())):
                # Add remainder to first few temperature groups
                temp_target = frames_per_temp + (1 if temp_idx < remainder_frames else 0)

                temp_available = sum(f['available_frames'] for f in temp_files)
                temp_target_actual = min(temp_target, temp_available)

                if temp != 'unknown' and temp != 'all':
                    log_verbose(f"{temp}K: {temp_target_actual:,} frames from {len(temp_files)} files")
                elif temp == 'all':
                    log_verbose(f"All files: {temp_target_actual:,} frames from {len(temp_files)} files")
                else:
                    log_verbose(f"Unknown temp: {temp_target_actual:,} frames from {len(temp_files)} files")

                # Distribute frames within this temperature group
                remaining_temp_target = temp_target_actual

                for i, file_info_item in enumerate(temp_files):
                    if i == len(temp_files) - 1:  # Last file in temperature group gets remainder
                        frames_to_take = remaining_temp_target
                    else:
                        # Equal distribution within temperature group
                        frames_to_take = temp_target_actual // len(temp_files)

                    frames_to_take = max(0, min(frames_to_take, file_info_item['available_frames']))
                    remaining_temp_target -= frames_to_take

                    file_distributions.append({
                        'file_index': file_info_item['index'],
                        'path': file_info_item['path'],
                        'limits': file_info_item['limits'],
                        'available_frames': file_info_item['available_frames'],
                        'frames_to_extract': frames_to_take,
                        'temperature': temp,
                        'proportion': frames_to_take / target_per_state_actual if target_per_state_actual > 0 else 0
                    })

                    log_verbose(f"File {file_info_item['index']}: {frames_to_take:,} frames "
                          f"({frames_to_take/file_info_item['available_frames']*100:.1f}% of available)")

        distribution[state] = file_distributions
        total_planned = sum(f['frames_to_extract'] for f in file_distributions)
        print(f"Planned: {total_planned:,} frames")
        log_verbose(f"Total planned: {total_planned:,} frames")

        # Show temperature balance summary (verbose only)
        temp_summary = {}
        for file_dist in file_distributions:
            temp = file_dist['temperature']
            if temp not in temp_summary:
                temp_summary[temp] = 0
            temp_summary[temp] += file_dist['frames_to_extract']

        log_verbose(f"Temperature balance:")
        for temp in sorted(temp_summary.keys(), key=lambda x: (x != 'quench', x == 'all', x)):
            if temp == 'quench':
                log_verbose(f"      Quench: {temp_summary[temp]:,} frames")
            elif temp == 'all':
                log_verbose(f"      All files: {temp_summary[temp]:,} frames")
            elif temp != 'unknown':
                log_verbose(f"      {temp}K: {temp_summary[temp]:,} frames")
            else:
                log_verbose(f"      Unknown: {temp_summary[temp]:,} frames")

    return distribution

def condense(path, limits, max_frames=10000):
    if not os.path.exists(path):
        return np.array([]), f"File not found: {path}"

    try:
        # Compute frame length.
        with open(path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return np.array([]), "Empty file"

        frame_length = 0
        timestep_counter = 0
        for line in lines:
            frame_length += 1
            if 'TIMESTEP' in line:
                timestep_counter += 1
            if timestep_counter > 1:
                break
        frame_length -= 1

        if frame_length <= 0:
            return np.array([]), "Could not determine frame structure"

        # Extract relevant information from each frame
        frames = []
        total_frames = int(len(lines) / frame_length)

        for frame_idx in range(total_frames):
            try:
                # Get .lammpstrj lines corresponding to this frame.
                frame_lines = lines[frame_idx * frame_length : (frame_idx + 1) * frame_length]

                # Get simulation box limits.
                x_lims = frame_lines[5].strip().split()
                y_lims = frame_lines[6].strip().split()
                z_lims = frame_lines[7].strip().split()
                frame_ = [
                    [-1.0, -1.0, float(x_lims[0]), float(x_lims[1])],
                    [-1.0, -1.0, float(y_lims[0]), float(y_lims[1])],
                    [-1.0, -1.0, float(z_lims[0]), float(z_lims[1])],
                ]

                # Get coordinates.
                for line in frame_lines[9:]:
                    vals = line.strip().split()
                    if len(vals) >= 5:  # Ensure we have enough columns
                        frame_.append([
                            int(vals[1]),
                            float(vals[2]),
                            float(vals[3]),
                            float(vals[4])
                        ])

                # Save frame.
                frame_ = np.array(frame_)
                frames.append(frame_)

            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping malformed frame {frame_idx} in {path}: {e}")
                continue

        if len(frames) == 0:
            return np.array([]), "No valid frames found"

        # Limit selected frames based on appropriate conditions.
        frames = np.stack(frames, axis=0)
        start_idx = 0 if limits[0] == -1 else limits[0]
        end_idx = frames.shape[0] if limits[1] == -1 else limits[1]
        frames = frames[start_idx:end_idx]

        # If there are still too many frames, take a random subsample.
        if frames.shape[0] > max_frames:
            idx = [i for i in range(frames.shape[0])]
            chosen_idx = np.random.choice(idx, size=max_frames, replace=False)
            frames = frames[chosen_idx]

        return frames, "Success"

    except Exception as e:
        return np.array([]), f"Error processing file: {str(e)}"

def generate_report(distribution, results):
    report = {
        'summary': {},
        'details': {},
        'statistics': {},
        'temperature_balance': {}
    }

    print("\n" + "="*60)
    print("PROCESSING REPORT")
    print("="*60)

    total_planned = 0
    total_actual = 0

    for state in distribution:
        planned = sum(f['frames_to_extract'] for f in distribution[state])
        actual = sum(results[state][i]['frames_extracted'] for i in range(len(results[state])))

        total_planned += planned
        total_actual += actual

        report['summary'][state] = {
            'planned_frames': planned,
            'actual_frames': actual,
            'success_rate': actual / planned if planned > 0 else 0,
            'file_count': len(distribution[state])
        }

        print(f"\n{state.upper()} Phase:")
        print(f"  Planned: {planned:,} frames")
        print(f"  Actual:  {actual:,} frames")
        print(f"  Success: {actual/planned*100:.1f}%" if planned > 0 else "  Success: N/A")

        # Temperature balance analysis
        temp_planned = {}
        temp_actual = {}

        for i, file_dist in enumerate(distribution[state]):
            temp = file_dist.get('temperature', 'unknown')
            result = results[state][i]

            if temp not in temp_planned:
                temp_planned[temp] = 0
                temp_actual[temp] = 0

            temp_planned[temp] += file_dist['frames_to_extract']
            temp_actual[temp] += result['frames_extracted']

        # Show temperature balance
        if len(temp_planned) > 1:
            print(f"Temperature Balance:")
            for temp in sorted(temp_planned.keys(), key=lambda x: (x != 'quench', x)):
                if temp == 'quench':
                    print(f"      Quench: {temp_actual[temp]:,}/{temp_planned[temp]:,} frames")
                elif temp != 'unknown':
                    print(f"      {temp}K: {temp_actual[temp]:,}/{temp_planned[temp]:,} frames")
                else:
                    print(f"      Unknown: {temp_actual[temp]:,}/{temp_planned[temp]:,} frames")

        report['temperature_balance'][state] = {
            'planned': temp_planned,
            'actual': temp_actual
        }

        # Detailed file breakdown
        report['details'][state] = []
        for i, file_dist in enumerate(distribution[state]):
            result = results[state][i]
            file_report = {
                'file_index': file_dist['file_index'],
                'path': file_dist['path'],
                'temperature': file_dist.get('temperature', 'unknown'),
                'planned_frames': file_dist['frames_to_extract'],
                'actual_frames': result['frames_extracted'],
                'status': result['status']
            }
            report['details'][state].append(file_report)

            status_icon = "Yes" if result['status'] == "Success" else "No"
            temp = file_dist.get('temperature', 'unknown')
            if temp == 'quench':
                temp_str = "Quench"
            elif temp == 'all':
                temp_str = "All"
            elif temp != 'unknown':
                temp_str = f"{temp}K"
            else:
                temp_str = "?K"
            print(f"    {status_icon} File {file_dist['file_index']} ({temp_str}): "
                  f"{result['frames_extracted']:,}/{file_dist['frames_to_extract']:,} frames")
            if result['status'] != "Success":
                print(f"      Error: {result['status']}")

    print(f"\nOVERALL SUMMARY:")
    print(f"  Total planned: {total_planned:,} frames")
    print(f"  Total actual:  {total_actual:,} frames")
    print(f"  Overall success: {total_actual/total_planned*100:.1f}%" if total_planned > 0 else "  Overall success: N/A")

    return report

if __name__ == '__main__':
    np.random.seed(1)

    # Configuration
    SIZE_PER_STATE = 10000
    IDENTIFIER = 'mbpol'
    file_paths = mbpol_paths

    # Create output directory
    os.makedirs('../data/frames', exist_ok=True)

    print("ENHANCED TRAJECTORY PREPROCESSING")
    print("="*50)

    # Phase 1: Analyze all files
    file_info = analyze_files(file_paths)

    # Phase 2: Calculate optimal distribution
    distribution = calculate_optimal_distribution(file_info, SIZE_PER_STATE)

    # Phase 3: Process files according to distribution
    print("\n=== PROCESSING PHASE ===")
    results = defaultdict(list)

    for state, file_list in distribution.items():
        print(f"\nProcessing {state} phase...")

        for file_dist in file_list:
            print(f"  Processing file {file_dist['file_index']}: {file_dist['path']}")

            arr, status = condense(
                path=file_dist['path'],
                limits=file_dist['limits'],
                max_frames=file_dist['frames_to_extract']
            )

            if arr.size > 0:
                arr = arr.astype(np.float16)
                output_path = f'../data/frames/{IDENTIFIER}_{state}_{file_dist["file_index"]}.npy'
                np.save(output_path, arr)
                frames_extracted = arr.shape[0]
            else:
                frames_extracted = 0

            results[state].append({
                'frames_extracted': frames_extracted,
                'status': status,
                'output_path': output_path if arr.size > 0 else None
            })

    # Phase 4: Generate comprehensive report
    report = generate_report(distribution, results)

    # Save report to file
    with open('processing_report.json', 'w') as f:
        json.dump(report, f, indent=2)