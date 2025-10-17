#!/usr/bin/env python3
"""
Dataset Noise Scaling Analysis

This script analyzes the differences between BNCI2014_001 and Lee2019_SSVEP datasets
and tests whether the noise injection method needs to be recalibrated for different
datasets due to:
1. Different channel counts (22 vs 62)
2. Different signal magnitudes/scaling
3. Different sampling rates and time windows
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def analyze_dataset_characteristics():
    """Analyze and compare dataset characteristics."""
    print("=" * 80)
    print("DATASET CHARACTERISTICS COMPARISON")
    print("=" * 80)
    
    # Known characteristics from code analysis
    datasets = {
        'BNCI2014_001': {
            'channels': 22,
            'sampling_rate': 250,  # Hz (resampled)
            'time_window': 4.0,    # seconds
            'n_times': 1000,       # 4s * 250Hz
            'paradigm': 'MotorImagery',
            'n_classes': 2,
            'description': 'Motor imagery with left/right hand movements'
        },
        'Lee2019_SSVEP': {
            'channels': 62,
            'sampling_rate': 1000, # Hz (original)
            'time_window': 4.0,    # seconds  
            'n_times': 4001,       # 4s * 1000Hz
            'paradigm': 'SSVEP',
            'n_classes': 4,
            'description': 'Steady-state visual evoked potentials'
        }
    }
    
    print("\nDataset Comparison:")
    print("-" * 50)
    for name, info in datasets.items():
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Calculate scaling factors
    print("\n" + "=" * 50)
    print("SCALING ANALYSIS")
    print("=" * 50)
    
    bnci = datasets['BNCI2014_001']
    lee = datasets['Lee2019_SSVEP']
    
    channel_ratio = lee['channels'] / bnci['channels']
    time_ratio = lee['n_times'] / bnci['n_times']
    sampling_ratio = lee['sampling_rate'] / bnci['sampling_rate']
    
    print(f"\nChannel count ratio (Lee/BNCI): {channel_ratio:.2f}x")
    print(f"Time points ratio (Lee/BNCI): {time_ratio:.2f}x") 
    print(f"Sampling rate ratio (Lee/BNCI): {sampling_ratio:.2f}x")
    
    # Total data points per epoch
    bnci_total = bnci['channels'] * bnci['n_times']
    lee_total = lee['channels'] * lee['n_times']
    total_ratio = lee_total / bnci_total
    
    print(f"\nTotal data points per epoch:")
    print(f"  BNCI2014_001: {bnci_total:,}")
    print(f"  Lee2019_SSVEP: {lee_total:,}")
    print(f"  Ratio (Lee/BNCI): {total_ratio:.2f}x")
    
    return datasets

def test_noise_scaling_effectiveness():
    """Test how noise scaling works with different dataset characteristics."""
    print("\n" + "=" * 80)
    print("NOISE SCALING EFFECTIVENESS TEST")
    print("=" * 80)
    
    # Simulate data with different characteristics
    np.random.seed(42)
    
    # BNCI2014_001-like data
    bnci_data = np.random.randn(10, 22, 1000) * 0.1  # 10 epochs, 22 channels, 1000 time points
    bnci_rms = np.sqrt(np.mean(bnci_data**2))
    
    # Lee2019_SSVEP-like data  
    lee_data = np.random.randn(10, 62, 4001) * 0.1   # 10 epochs, 62 channels, 4001 time points
    lee_rms = np.sqrt(np.mean(lee_data**2))
    
    print(f"\nSimulated data characteristics:")
    print(f"  BNCI2014_001 RMS: {bnci_rms:.6f}")
    print(f"  Lee2019_SSVEP RMS: {lee_rms:.6f}")
    print(f"  RMS ratio (Lee/BNCI): {lee_rms/bnci_rms:.2f}")
    
    # Test current noise scaling method
    from augmentation.noise import EEGNoiseAugmentor
    
    intensity = 10.0  # 10% intensity
    
    print(f"\nTesting noise application at {intensity}% intensity:")
    print("-" * 50)
    
    # Test on BNCI-like data
    bnci_augmentor = EEGNoiseAugmentor(noise_type='gaussian', intensity=intensity, seed=42)
    bnci_noisy = bnci_augmentor.transform(bnci_data)
    bnci_noise_rms = np.sqrt(np.mean((bnci_noisy - bnci_data)**2))
    bnci_snr = 20 * np.log10(bnci_rms / bnci_noise_rms)
    
    print(f"BNCI2014_001-like data:")
    print(f"  Original RMS: {bnci_rms:.6f}")
    print(f"  Noise RMS: {bnci_noise_rms:.6f}")
    print(f"  SNR: {bnci_snr:.2f} dB")
    
    # Test on Lee-like data
    lee_augmentor = EEGNoiseAugmentor(noise_type='gaussian', intensity=intensity, seed=42)
    lee_noisy = lee_augmentor.transform(lee_data)
    lee_noise_rms = np.sqrt(np.mean((lee_noisy - lee_data)**2))
    lee_snr = 20 * np.log10(lee_rms / lee_noise_rms)
    
    print(f"\nLee2019_SSVEP-like data:")
    print(f"  Original RMS: {lee_rms:.6f}")
    print(f"  Noise RMS: {lee_noise_rms:.6f}")
    print(f"  SNR: {lee_snr:.2f} dB")
    
    # Check if noise scaling is consistent
    print(f"\nNoise scaling consistency:")
    print(f"  BNCI noise/signal ratio: {bnci_noise_rms/bnci_rms:.3f}")
    print(f"  Lee noise/signal ratio: {lee_noise_rms/lee_rms:.3f}")
    print(f"  Ratio difference: {abs(bnci_noise_rms/bnci_rms - lee_noise_rms/lee_rms):.3f}")
    
    return {
        'bnci_rms': bnci_rms,
        'lee_rms': lee_rms,
        'bnci_noise_rms': bnci_noise_rms,
        'lee_noise_rms': lee_noise_rms,
        'bnci_snr': bnci_snr,
        'lee_snr': lee_snr
    }

def analyze_noise_calibration_issues():
    """Analyze potential issues with noise calibration."""
    print("\n" + "=" * 80)
    print("NOISE CALIBRATION ISSUE ANALYSIS")
    print("=" * 80)
    
    # From the noise.py code analysis
    print("\nCurrent noise scaling method (from augmentation/noise.py):")
    print("-" * 60)
    print("1. Calculate signal RMS: signal_rms = sqrt(mean(data**2))")
    print("2. Set noise scale: noise_scale = 4.0 * signal_rms")
    print("3. Apply intensity scaling: noise_scale *= (intensity / 100.0)")
    print("4. Apply to subset of channels: n_contam = int(n_channels * intensity / 100.0)")
    
    print("\nPotential issues identified:")
    print("-" * 40)
    print("1. CHANNEL SCALING ISSUE:")
    print("   - BNCI2014_001: 22 channels -> 10% = 2.2 -> 2 channels contaminated")
    print("   - Lee2019_SSVEP: 62 channels -> 10% = 6.2 -> 6 channels contaminated")
    print("   - More channels contaminated = more total noise added")
    
    print("\n2. SIGNAL MAGNITUDE DIFFERENCES:")
    print("   - Different datasets may have different signal scales")
    print("   - Noise scale = 4.0 * signal_rms may not be appropriate for all datasets")
    print("   - SSVEP signals might be stronger than motor imagery signals")
    
    print("\n3. TEMPORAL DIMENSION SCALING:")
    print("   - BNCI2014_001: 1000 time points")
    print("   - Lee2019_SSVEP: 4001 time points")
    print("   - More time points = more noise samples per contaminated channel")
    
    print("\n4. PARADIGM DIFFERENCES:")
    print("   - Motor imagery: endogenous brain activity")
    print("   - SSVEP: strong, consistent visual stimulation responses")
    print("   - SSVEP signals may be more robust to noise")

def suggest_noise_calibration_fixes():
    """Suggest fixes for noise calibration issues."""
    print("\n" + "=" * 80)
    print("SUGGESTED NOISE CALIBRATION FIXES")
    print("=" * 80)
    
    print("\n1. DATASET-AWARE NOISE SCALING:")
    print("-" * 40)
    print("   - Normalize noise by dataset characteristics")
    print("   - Account for channel count differences")
    print("   - Use consistent noise-to-signal ratio across datasets")
    
    print("\n2. IMPROVED NOISE SCALING FORMULA:")
    print("-" * 40)
    print("   Current: noise_scale = 4.0 * signal_rms * (intensity/100)")
    print("   Suggested: noise_scale = base_scale * signal_rms * (intensity/100) * channel_factor")
    print("   Where: channel_factor = sqrt(n_channels_bnci / n_channels_current)")
    
    print("\n3. PARADIGM-SPECIFIC CALIBRATION:")
    print("-" * 40)
    print("   - Motor imagery: Use current scaling (calibrated for BNCI2014_001)")
    print("   - SSVEP: Use stronger noise scaling due to robust signals")
    print("   - Consider signal strength differences between paradigms")
    
    print("\n4. INTENSITY INTERPRETATION:")
    print("-" * 40)
    print("   - Current: intensity = % of channels to contaminate")
    print("   - Alternative: intensity = noise-to-signal ratio")
    print("   - Or: intensity = total noise power as % of signal power")

def test_improved_noise_scaling():
    """Test an improved noise scaling approach."""
    print("\n" + "=" * 80)
    print("TESTING IMPROVED NOISE SCALING")
    print("=" * 80)
    
    # Simulate data
    np.random.seed(42)
    bnci_data = np.random.randn(10, 22, 1000) * 0.1
    lee_data = np.random.randn(10, 62, 4001) * 0.1
    
    # Current method
    from augmentation.noise import EEGNoiseAugmentor
    
    intensity = 10.0
    
    # Test current method
    bnci_augmentor = EEGNoiseAugmentor(noise_type='gaussian', intensity=intensity, seed=42)
    lee_augmentor = EEGNoiseAugmentor(noise_type='gaussian', intensity=intensity, seed=42)
    
    bnci_noisy_current = bnci_augmentor.transform(bnci_data)
    lee_noisy_current = lee_augmentor.transform(lee_data)
    
    # Calculate current noise levels
    bnci_noise_current = np.sqrt(np.mean((bnci_noisy_current - bnci_data)**2))
    lee_noise_current = np.sqrt(np.mean((lee_noisy_current - lee_data)**2))
    
    print(f"Current method results:")
    print(f"  BNCI noise level: {bnci_noise_current:.6f}")
    print(f"  Lee noise level: {lee_noise_current:.6f}")
    print(f"  Lee/BNCI ratio: {lee_noise_current/bnci_noise_current:.2f}")
    
    # Improved method: normalize by channel count
    def improved_gaussian_noise(data, intensity, seed=42):
        np.random.seed(seed)
        n_epochs, n_channels, n_times = data.shape
        data_aug = data.copy()
        
        # Calculate signal RMS
        signal_rms = np.sqrt(np.mean(data**2))
        
        # Channel normalization factor (based on BNCI2014_001 = 22 channels)
        bnci_channels = 22
        channel_factor = np.sqrt(bnci_channels / n_channels)
        
        # Improved noise scaling
        noise_scale = 4.0 * signal_rms * (intensity / 100.0) * channel_factor
        
        # Apply to subset of channels
        n_contam = int(np.round(n_channels * intensity / 100.0))
        n_contam = max(1, n_contam) if intensity > 0 else 0
        n_contam = min(n_contam, n_channels)
        
        for i in range(n_epochs):
            if n_contam == 0:
                continue
            contam_idxs = np.random.choice(n_channels, size=n_contam, replace=False)
            noise = np.random.randn(n_contam, n_times)
            data_aug[i, contam_idxs, :] += noise_scale * noise
        
        return data_aug
    
    # Test improved method
    bnci_noisy_improved = improved_gaussian_noise(bnci_data, intensity, seed=42)
    lee_noisy_improved = improved_gaussian_noise(lee_data, intensity, seed=42)
    
    # Calculate improved noise levels
    bnci_noise_improved = np.sqrt(np.mean((bnci_noisy_improved - bnci_data)**2))
    lee_noise_improved = np.sqrt(np.mean((lee_noisy_improved - lee_data)**2))
    
    print(f"\nImproved method results:")
    print(f"  BNCI noise level: {bnci_noise_improved:.6f}")
    print(f"  Lee noise level: {lee_noise_improved:.6f}")
    print(f"  Lee/BNCI ratio: {lee_noise_improved/bnci_noise_improved:.2f}")
    
    print(f"\nImprovement analysis:")
    print(f"  Current Lee/BNCI ratio: {lee_noise_current/bnci_noise_current:.2f}")
    print(f"  Improved Lee/BNCI ratio: {lee_noise_improved/bnci_noise_improved:.2f}")
    print(f"  Target ratio (should be ~1.0): 1.00")

def main():
    """Main analysis function."""
    print("Dataset Noise Scaling Analysis")
    print("=" * 80)
    
    # Run all analyses
    datasets = analyze_dataset_characteristics()
    noise_results = test_noise_scaling_effectiveness()
    analyze_noise_calibration_issues()
    suggest_noise_calibration_fixes()
    test_improved_noise_scaling()
    
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\nKey findings:")
    print("1. Lee2019_SSVEP has 2.8x more channels than BNCI2014_001 (62 vs 22)")
    print("2. Lee2019_SSVEP has 4x more time points (4001 vs 1000)")
    print("3. Current noise method adds more total noise to Lee2019_SSVEP due to more channels")
    print("4. SSVEP signals may be inherently more robust than motor imagery signals")
    print("5. Noise scaling should account for dataset characteristics")
    
    print("\nImmediate recommendations:")
    print("1. Implement dataset-aware noise scaling")
    print("2. Test with stronger noise intensities for Lee2019_SSVEP")
    print("3. Consider paradigm-specific noise calibration")
    print("4. Verify that noise is actually affecting model performance")

if __name__ == "__main__":
    main()
