# -*- coding: utf-8 -*-
"""
NILM Data Mixing Script v3 (Evenly-Spaced OFF-Period Injection)
================================================================
Synthetic event detection strategy:
  - Flatten .npy [N, 600, C] into a continuous sequence.
  - Use EVENT-DENSITY detection (Hart85-style edge counting) to find
    appliance-active periods. This is robust to:
      * Complex multi-phase appliances (washingmachine, dishwasher)
      * Low-power phases that drop below simple power thresholds
      * Internal vibrations that confuse gap-based merging
  - Active periods are distributed EVENLY across real OFF segments.
  - Aggregate is reconstructed as: real_background + synthetic_appliance.

Event Density Algorithm:
  1. Compute first-order difference (edges) of the power signal.
  2. Find timesteps where |delta| >= edge_threshold.
  3. For each timestep, count events in a sliding density window.
  4. Threshold the density to get raw "active" mask.
  5. Morphological closing: merge gaps < min_gap_steps.
  6. Filter: remove segments shorter than min_duration_steps.

Usage:
    python mix_training_data_multivariate_v3.py --appliance washingmachine \\
        --real_rows 200000 --synthetic_rows 20000 \\
        --suffix 200k+20k_event_even_v3
"""

import numpy as np
import pandas as pd
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG_PATH = 'Config/preprocess/preprocess_multivariate.yaml'


def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return None


CONFIG = load_config()

APPLIANCE_SPECS = {
    'kettle':         {'mean': 700,  'std': 1000, 'max_power': 3998},
    'microwave':      {'mean': 500,  'std':  800, 'max_power': 3969},
    'fridge':         {'mean': 200,  'std':  400, 'max_power':  350},
    'dishwasher':     {'mean': 700,  'std': 1000, 'max_power': 3964},
    'washingmachine': {'mean': 400,  'std':  700, 'max_power': 3999},
}

if CONFIG:
    for _app, _data in CONFIG['appliances'].items():
        APPLIANCE_SPECS[_app] = {
            'mean': _data['mean'], 'std': _data['std'], 'max_power': _data['max_power']
        }
    AGG_MEAN = CONFIG['normalization']['aggregate_mean']
    AGG_STD  = CONFIG['normalization']['aggregate_std']
else:
    AGG_MEAN, AGG_STD = 522, 814

WINDOW_SIZE = 600  # each .npy sample length


# ── Per-appliance Detection Config ──────────────────────────────────────────
# At 6s/sample:  10 steps = 1 min,  30 = 3 min,  100 = 10 min,  300 = 30 min
@dataclass
class DetectionConfig:
    strategy: str           # 'density' or 'threshold'
    edge_threshold: float   # [density] minimum |delta_power| to count as event
    density_window: int     # [density] sliding window length for event counting
    density_min_events: int # [density] events per window to be "active"
    power_threshold: float  # [threshold] fallback simple threshold
    min_gap_steps: int      # merge gaps shorter than this
    min_duration_steps: int # discard segments shorter than this
    max_event_length: int   # split events longer than this into chunks


DETECTION_CONFIG = {
    'washingmachine': DetectionConfig(
        strategy='threshold',
        edge_threshold=30,
        density_window=50,
        density_min_events=2,
        power_threshold=1800,   # Raised to 1800W for cleaner start
        min_gap_steps=10,       # VERY tight - won't bridge different cycles
        min_duration_steps=30,
        max_event_length=4000,
    ),
    'dishwasher': DetectionConfig(
        strategy='density',
        edge_threshold=50,
        density_window=50,
        density_min_events=2,
        power_threshold=10,
        min_gap_steps=120,
        min_duration_steps=80,
        max_event_length=800,
    ),
    'fridge': DetectionConfig(
        strategy='threshold',
        edge_threshold=20,
        density_window=30,
        density_min_events=1,
        power_threshold=50,
        min_gap_steps=60,
        min_duration_steps=20,
        max_event_length=300,
    ),
    'kettle': DetectionConfig(
        strategy='threshold',
        edge_threshold=100,
        density_window=20,
        density_min_events=1,
        power_threshold=200,
        min_gap_steps=5,
        min_duration_steps=5,
        max_event_length=200,
    ),
    'microwave': DetectionConfig(
        strategy='threshold',
        edge_threshold=100,
        density_window=20,
        density_min_events=1,
        power_threshold=200,
        min_gap_steps=5,
        min_duration_steps=5,
        max_event_length=100,
    ),
}


# ── Detection Helpers ───────────────────────────────────────────────────────

def detect_events_density(power: np.ndarray, cfg: DetectionConfig,
                          appliance_name: str) -> List[Tuple[int, int]]:
    """
    Hart85-style event density detection.

    1. First-order difference → edges
    2. Event indicator: |delta| >= edge_threshold
    3. Sliding-window density: count(events) per density_window steps
    4. Active mask: density >= density_min_events
    5. Morphological closing: merge gaps < min_gap_steps
    6. Duration filter: keep segments >= min_duration_steps
    """
    n = len(power)

    # Step 1-2: edges
    delta   = np.abs(np.diff(power, prepend=power[0]))
    is_edge = delta >= cfg.edge_threshold

    # Step 3: sliding density (convolution = fast sum over window)
    kernel  = np.ones(cfg.density_window)
    density = np.convolve(is_edge.astype(float), kernel, mode='same')

    # Step 4: active mask from density
    active = density >= cfg.density_min_events

    # Step 5: morphological closing — merge gaps < min_gap_steps
    # Dilate the active mask by min_gap_steps on each side, then erode back
    gap_kernel = np.ones(cfg.min_gap_steps * 2 + 1)
    dilated    = np.convolve(active.astype(float), gap_kernel, mode='same') > 0
    # Erosion: convolve again and threshold at full kernel sum
    eroded     = np.convolve(dilated.astype(float), gap_kernel, mode='same') >= gap_kernel.sum()
    # Use dilated to close gaps but keep original boundaries tight
    closed     = dilated  # dilation alone is sufficient for closing gaps

    # Step 6: extract segments and apply duration filter
    segs, i_seg = [], False
    for i, v in enumerate(closed):
        if v and not i_seg:
            s = i; i_seg = True
        elif not v and i_seg:
            if i - s >= cfg.min_duration_steps:
                segs.append((s, i))
            i_seg = False
    if i_seg and n - s >= cfg.min_duration_steps:
        segs.append((s, n))

    return segs


def detect_events_threshold(power: np.ndarray, cfg: DetectionConfig,
                             appliance_name: str) -> List[Tuple[int, int]]:
    """
    Simple power threshold detection with morphological closing.
    Better for short-burst appliances (kettle, microwave, fridge).
    """
    is_on  = power >= cfg.power_threshold
    kernel = np.ones(cfg.min_gap_steps * 2 + 1)
    closed = np.convolve(is_on.astype(float), kernel, mode='same') > 0

    segs, i_seg = [], False
    n = len(power)
    for i, v in enumerate(closed):
        if v and not i_seg:
            s = i; i_seg = True
        elif not v and i_seg:
            if i - s >= cfg.min_duration_steps:
                segs.append((s, i))
            i_seg = False
    if i_seg and n - s >= cfg.min_duration_steps:
        segs.append((s, n))

    return segs


def detect_synthetic_events(power: np.ndarray, appliance_name: str
                             ) -> List[Tuple[int, int]]:
    """Choose detection strategy per appliance and return (start, end) list."""
    cfg = DETECTION_CONFIG.get(appliance_name, DETECTION_CONFIG['kettle'])
    if cfg.strategy == 'density':
        segs = detect_events_density(power, cfg, appliance_name)
    else:
        segs = detect_events_threshold(power, cfg, appliance_name)

    if segs:
        avg_len = np.mean([e - s for s, e in segs])
        print(f"  [{appliance_name}] {len(segs)} events detected  "
              f"(strategy={cfg.strategy}, avg={avg_len:.0f}steps, "
              f"min={min(e-s for s,e in segs)}, max={max(e-s for s,e in segs)})")
    else:
        print(f"  [{appliance_name}] WARNING: No events detected!")
    return segs


# ── Synthetic Data Loading ──────────────────────────────────────────────────

def load_and_build_events(appliance_name: str):
    """
    Load .npy [N, 600, C], flatten to continuous sequence,
    run event detection, split long events into max_event_length chunks,
    return list of event dicts.
    """
    npy_path = Path('synthetic_data_multivariate') / f'ddpm_fake_{appliance_name}_multivariate.npy'
    if not npy_path.exists():
        npy_path = Path(f'OUTPUT/{appliance_name}_512/ddpm_fake_{appliance_name}_512.npy')

    print(f"  Loading: {npy_path}")
    data = np.load(npy_path)   # [N, 600, C]
    spec = APPLIANCE_SPECS[appliance_name]

    # Flatten to continuous sequence
    N            = data.shape[0]
    full_power   = (data[:, :, 0] * spec['max_power']).reshape(-1)
    full_time    = data[:, :, 1:].reshape(-1, data.shape[2] - 1)
    print(f"  Flattened: {len(full_power):,} steps ({N} windows × 600)")

    # Detect events using per-appliance strategy
    segs = detect_synthetic_events(full_power, appliance_name)

    cfg = DETECTION_CONFIG.get(appliance_name, DETECTION_CONFIG['kettle'])
    max_len = cfg.max_event_length

    # Convert to event dicts AND split any event exceeding max_event_length
    events = []
    for s, e in segs:
        # Tight Cropping: Ensure the extracted slice starts and ends AT the actual thresholded pulse
        evt_power = full_power[s:e]
        on_indices = np.where(evt_power >= cfg.power_threshold)[0]
        if len(on_indices) == 0: continue
        
        # New tight boundaries within the flattened sequence
        tight_s = s + on_indices[0]
        tight_e = s + on_indices[-1] + 1
        
        pos = tight_s
        actual_e = tight_e
        while pos < actual_e:
            chunk_end = min(pos + max_len, actual_e)
            chunk_len = chunk_end - pos
            events.append({
                'power':  full_power[pos:chunk_end],   # Watts
                'length': chunk_len,
            })
            pos = chunk_end

    total_rows = sum(ev['length'] for ev in events)
    print(f"  After chunking (max={max_len}): {len(events)} units, "
          f"total={total_rows:,} rows, "
          f"avg={total_rows//max(len(events),1)} steps/unit")
    return events


# ── Real Data OFF-Period Detection ─────────────────────────────────────────

def get_off_periods_mask(appliance_name: str, df: pd.DataFrame) -> np.ndarray:
    """True where the target appliance is strictly OFF in the real data."""
    spec  = APPLIANCE_SPECS[appliance_name]
    app_w = df.iloc[:, 1].values * spec['std'] + spec['mean']

    if CONFIG and appliance_name in CONFIG['appliances']:
        threshold = CONFIG['appliances'][appliance_name]['on_power_threshold']
        l_window  = CONFIG['algorithm1']['window_length']
    else:
        threshold, l_window = 15.0, 50

    is_on  = app_w >= threshold
    kernel = np.ones(l_window * 2 + 1)
    exp_on = np.convolve(is_on.astype(float), kernel, mode='same') > 0
    return ~exp_on


# ── Main ───────────────────────────────────────────────────────────────────

def mix_data_v3(appliance_name: str, real_rows: int, synthetic_rows: int,
                real_path: Optional[str] = None, suffix: str = 'v3'):
    print(f"\n{'='*60}")
    print(f"NILM Mixed Dataset v3 — {appliance_name}")
    print(f"  real={real_rows:,}  synthetic_target={synthetic_rows:,}")
    print(f"{'='*60}")

    # 1. Load real data
    if real_path is None:
        real_path = f'created_data/UK_DALE/{appliance_name}_training_.csv'
    real_df   = pd.read_csv(real_path).iloc[:real_rows].copy()
    col_names = list(real_df.columns)
    print(f"  Real data: {len(real_df):,} rows")

    # 2. Find OFF segments
    is_off   = get_off_periods_mask(appliance_name, real_df)
    off_segs, in_seg = [], False
    for i, v in enumerate(is_off):
        if v and not in_seg:
            seg_s = i; in_seg = True
        elif not v and in_seg:
            off_segs.append((seg_s, i)); in_seg = False
    if in_seg:
        off_segs.append((seg_s, len(is_off)))
    off_segs      = [(s, e) for s, e in off_segs if e - s > 10]
    total_off_len = sum(e - s for s, e in off_segs)
    print(f"  OFF segments: {len(off_segs)}, total OFF rows: {total_off_len:,}")

    # 3. Baseline
    if synthetic_rows <= 0:
        out_dir = Path(f'created_data/UK_DALE/{appliance_name}')
        out_dir.mkdir(parents=True, exist_ok=True)
        real_df.to_csv(out_dir / f'{appliance_name}_training_{suffix}.csv', index=False)
        print("  Baseline mode done.")
        return

    # 4. Load synthetic events (density detection)
    all_events = load_and_build_events(appliance_name)
    if not all_events:
        raise RuntimeError(f"No synthetic events detected for {appliance_name}!")

    # Cycle until we reach target rows
    selected, total_syn = [], 0
    while total_syn < synthetic_rows:
        for ev in all_events:
            selected.append(ev)
            total_syn += ev['length']
            if total_syn >= synthetic_rows:
                break
    print(f"  Selected {len(selected)} events = {total_syn:,} rows (target={synthetic_rows:,})")

    # 5. Distribute proportionally + evenly across OFF segments
    n_ev       = len(selected)
    seg_counts = [max(0, int(np.round(n_ev * (e - s) / total_off_len)))
                  for s, e in off_segs]
    diff = n_ev - sum(seg_counts)
    if diff != 0:
        sign = 1 if diff > 0 else -1
        for i in np.argsort([e - s for s, e in off_segs])[::-1]:
            seg_counts[i] += sign; diff -= sign
            if diff == 0: break

    # 6. Build final dataset
    spec        = APPLIANCE_SPECS[appliance_name]
    final_parts = []
    ev_ptr      = 0
    curr_idx    = 0

    for seg_idx, (seg_s, seg_e) in enumerate(off_segs):
        if seg_s > curr_idx:
            final_parts.append(real_df.iloc[curr_idx:seg_s])

        seg_df  = real_df.iloc[seg_s:seg_e].copy()
        seg_len = seg_e - seg_s
        n_here  = seg_counts[seg_idx]

        if n_here == 0 or ev_ptr >= n_ev:
            final_parts.append(seg_df)
            curr_idx = seg_e
            continue

        evs_here    = selected[ev_ptr: ev_ptr + n_here]
        syn_len_sum = sum(ev['length'] for ev in evs_here)
        gap         = max(0, (seg_len - syn_len_sum) // (n_here + 1))

        seg_pos = 0
        for ev in evs_here:
            # Gap before event
            if gap > 0 and seg_pos + gap <= seg_len:
                final_parts.append(seg_df.iloc[seg_pos: seg_pos + gap])
                seg_pos += gap

            room       = seg_len - seg_pos
            inject_len = min(ev['length'], room)
            if inject_len <= 0:
                break

            # Physical reconstruction: bg + synthetic_appliance
            bg_slice   = seg_df.iloc[seg_pos: seg_pos + inject_len]
            agg_w      = bg_slice.iloc[:, 0].values * AGG_STD + AGG_MEAN
            app_w_real = bg_slice.iloc[:, 1].values * spec['std'] + spec['mean']
            bg_w       = np.maximum(agg_w - app_w_real, 0)

            y_w    = ev['power'][:inject_len]
            t_feat = ev['time'] [:inject_len]

            x_syn  = bg_w + y_w
            agg_z  = (x_syn - AGG_MEAN) / AGG_STD
            app_z  = (y_w   - spec['mean']) / spec['std']

            evt_df = pd.DataFrame(t_feat, columns=col_names[2:])
            evt_df.insert(0, col_names[0], agg_z)
            evt_df.insert(1, col_names[1], app_z)
            final_parts.append(evt_df)
            seg_pos += inject_len
            ev_ptr  += 1

        if seg_pos < seg_len:
            final_parts.append(seg_df.iloc[seg_pos:])
        curr_idx = seg_e

    if curr_idx < len(real_df):
        final_parts.append(real_df.iloc[curr_idx:])

    # 7. Save
    final_df = pd.concat(final_parts, ignore_index=True)
    out_dir  = Path(f'created_data/UK_DALE/{appliance_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{appliance_name}_training_{suffix}.csv'
    final_df.to_csv(out_file, index=False)
    print(f"\n  ✅ DONE  →  {out_file}")
    print(f"     Total rows: {len(final_df):,}")


# ── Entry ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appliance',      type=str, required=True)
    parser.add_argument('--real_rows',      type=int, default=200000)
    parser.add_argument('--synthetic_rows', type=int, default=200000)
    parser.add_argument('--suffix',         type=str, default='200k+200k_event_even_v3')
    args = parser.parse_args()
    mix_data_v3(args.appliance, args.real_rows, args.synthetic_rows, suffix=args.suffix)
