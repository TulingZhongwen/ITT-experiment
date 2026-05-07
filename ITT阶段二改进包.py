"""
ITT Stage 2: Public EEG data (ds002718) – anesthesia vs baseline
Requires data downloaded from OpenNeuro.
"""

import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import stats
from itt_core import (mutual_info_first_min, false_nearest_neighbor,
                      reconstruct, compute_lambda, lambda_significance,
                      compute_theta)

def load_ds002718(data_path, subject='sub-01', task='rest'):
    """Load EEG data for a single subject."""
    fname = os.path.join(data_path, subject, 'eeg', f'{subject}_task_{task}_eeg.edf')
    raw = mne.io.read_raw_edf(fname, preload=True)
    raw.filter(0.5, 45, fir_design='firwin')
    raw.notch_filter(50)
    raw.set_eeg_reference('average')
    # pick EEG channels only
    raw.pick_types(eeg=True)
    # events
    events, event_id = mne.events_from_annotations(raw)
    return raw, events, event_id

def extract_epochs(raw, events, event_id, condition, tmin=0, tmax=300, baseline=None):
    """Extract epochs for a given condition."""
    if condition not in event_id:
        raise ValueError(f"Condition {condition} not found in events")
    epochs = mne.Epochs(raw, events, {condition: event_id[condition]},
                        tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    return epochs

def main():
    data_path = './data/ds002718'   # modify to your local path
    if not os.path.exists(data_path):
        print("Please download ds002718 from OpenNeuro first.")
        return

    raw, events, event_id = load_ds002718(data_path)
    # Example events: 'baseline' and 'loss' (adjust according to actual event names)
    # We use first 5 minutes of baseline and first 5 minutes of loss.
    epochs_baseline = extract_epochs(raw, events, event_id, 'baseline', tmin=0, tmax=300)
    epochs_loss = extract_epochs(raw, events, event_id, 'loss', tmin=0, tmax=300)

    # Compute Λ and Θ for the first channel (or average over channels)
    data_b = epochs_baseline.get_data()[0]   # (n_channels, n_times)
    data_l = epochs_loss.get_data()[0]

    # Choose a representative channel (e.g., Fz)
    ch_idx = 0
    sig_b = data_b[ch_idx]
    sig_l = data_l[ch_idx]

    # Estimate parameters from baseline
    tau = mutual_info_first_min(sig_b, max_lag=50)
    d = false_nearest_neighbor(sig_b, tau=tau, max_dim=20)
    dt = 1/500  # 500 Hz
    delta_b = np.percentile(np.abs(np.diff(sig_b)), 5)
    delta_l = np.percentile(np.abs(np.diff(sig_l)), 5)

    # Compute Λ and significance
    lam_b, _, p_b = lambda_significance(sig_b, tau, d, dt)
    lam_l, _, p_l = lambda_significance(sig_l, tau, d, dt)

    # Compute Θ
    traj_b = reconstruct(sig_b, tau, d)
    traj_l = reconstruct(sig_l, tau, d)
    theta_b = compute_theta(traj_b, delta_b)
    theta_l = compute_theta(traj_l, delta_l)

    print("=== Stage 2 Results (single subject) ===")
    print(f"Baseline: Λ={lam_b:.4f} (p={p_b:.4f}), Θ={theta_b:.4f}")
    print(f"Loss:     Λ={lam_l:.4f} (p={p_l:.4f}), Θ={theta_l:.4f}")

    # Simple boxplot across channels (optional)
    n_ch = data_b.shape[0]
    lambdas_b, lambdas_l = [], []
    thetas_b, thetas_l = [], []
    for ch in range(n_ch):
        sigb = data_b[ch]
        sigl = data_l[ch]
        tau = mutual_info_first_min(sigb)
        d = false_nearest_neighbor(sigb, tau=tau)
        deltab = np.percentile(np.abs(np.diff(sigb)), 5)
        deltal = np.percentile(np.abs(np.diff(sigl)), 5)
        trajb = reconstruct(sigb, tau, d)
        trajl = reconstruct(sigl, tau, d)
        lambdas_b.append(compute_lambda(trajb, dt))
        lambdas_l.append(compute_lambda(trajl, dt))
        thetas_b.append(compute_theta(trajb, deltab))
        thetas_l.append(compute_theta(trajl, deltal))

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.boxplot([lambdas_b, lambdas_l], labels=['Baseline','Loss'])
    ax1.set_ylabel('Λ')
    ax1.set_title('Prediction 1: Causal loop strength')
    ax2.boxplot([thetas_b, thetas_l], labels=['Baseline','Loss'])
    ax2.set_ylabel('Θ(s)')
    ax2.set_title('Prediction 2: Self-distance')
    plt.savefig('stage2_results.png')
    plt.show()

if __name__ == '__main__':
    main()
