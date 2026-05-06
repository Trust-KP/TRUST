#!/usr/bin/env python3
"""
KP Four-Constraint Stochastic Zero Search v2.0
==============================================

This is a numerical convergence experiment for the Riemann zeta function,
not a proof of the Riemann Hypothesis.

Changes from v1:
  1. Adaptive t-stepping: when Qimen CS is high at sigma=1/2, narrow the
     t-step so the search spends more time near a possible zero.
  2. 2D refinement: refine both sigma and t jointly, not sigma alone.
  3. BaZi boundary check runs before Bagua folding.
  4. Richer output: recall against known low zeros, layer pruning %, CS peak.

Note: this search intentionally probes sigma=1/2 on every t step. That makes
it an RH-line convergence test, not an unbiased discovery mechanism.
"""

import argparse
import math
import random

import mpmath as mp

mp.mp.dps = 25

# Known non-trivial zero heights in the default neighborhood.
KNOWN_ZEROS = [
    14.134725,
    21.022040,
    25.010858,
    30.424876,
    32.935062,
    37.586178,
    40.918720,
    43.327073,
    48.005151,
    49.773832,
    52.970322,
]


def zeta_mag(sigma, t):
    """Return |zeta(sigma + i*t)|."""
    return float(abs(mp.zeta(mp.mpc(sigma, t))))


# ----------------------------------------------------------------------
# CONSTRAINT LAYERS
# ----------------------------------------------------------------------


def qimen_cs(sigma, t, baseline):
    """行 Qimen: continuation-strength score."""
    return max(0.0, 1.0 - zeta_mag(sigma, t) / max(baseline, 0.01))


def ziwei_ok(t, found_zeros, factor=0.3):
    """度 Ziwei: minimum spacing gate based on expected zero spacing."""
    if not found_zeros:
        return True
    min_spacing = (2 * math.pi) / math.log(max(t, 10) / (2 * math.pi))
    return all(abs(t - zero[1]) >= min_spacing * factor for zero in found_zeros)


def bagua_reduce(sigma):
    """構 Bagua: V4 symmetry fold, reflecting sigma > 1/2 into [0, 1/2]."""
    return 1.0 - sigma if sigma > 0.5 else sigma


def bazi_ok_raw(sigma):
    """釋 BaZi: boundary check on raw sigma before Bagua."""
    return 0.05 <= sigma <= 0.95


def estimated_candidate_count(t_min, t_max, step=0.20):
    """Approximate candidate count before pruning in coarse mode."""
    if t_max <= t_min:
        return 0
    return int(((t_max - t_min) / step) * 9)


# ----------------------------------------------------------------------
# REFINEMENT
# ----------------------------------------------------------------------


def refine_2d(sigma0, t0, sigma_step=0.005, t_step=0.01):
    """Jointly refine (sigma, t) around a candidate near-zero."""
    best = (sigma0, t0, zeta_mag(sigma0, t0))
    for ds in [i * sigma_step for i in range(-10, 11)]:
        for dt in [i * t_step for i in range(-5, 6)]:
            sigma2 = sigma0 + ds
            t2 = t0 + dt
            if not 0 < sigma2 < 1:
                continue
            mag2 = zeta_mag(sigma2, t2)
            if mag2 < best[2]:
                best = (sigma2, t2, mag2)
    return best


# ----------------------------------------------------------------------
# SEARCH
# ----------------------------------------------------------------------


def stochastic_search_v2(
    t_min=10.5,
    t_max=52.0,
    threshold=0.06,
    cs_min=0.05,
    cs_zoom=0.85,
    seed=42,
):
    """Run the v2 bounded stochastic near-zero search."""
    random.seed(seed)
    total = 0
    found = 0
    pruned_by = {"Q": 0, "Z": 0, "A": 0}
    found_zeros = []
    cs_peaks = []

    t = t_min
    fine_mode = False

    while t < t_max:
        baseline_half = zeta_mag(0.5, t - 1.0)
        cs_half = qimen_cs(0.5, t, baseline_half)
        cs_peaks.append(cs_half)

        if cs_half > cs_zoom:
            fine_mode = True
        elif cs_half < cs_zoom * 0.7:
            fine_mode = False

        sigmas = [random.random() for _ in range(8)]
        sigmas.append(0.5)

        for raw_sigma in sigmas:
            total += 1

            if not bazi_ok_raw(raw_sigma):
                pruned_by["A"] += 1
                continue

            baseline = zeta_mag(raw_sigma, t - 1.0)
            if qimen_cs(raw_sigma, t, baseline) < cs_min:
                pruned_by["Q"] += 1
                continue

            if not ziwei_ok(t, found_zeros):
                pruned_by["Z"] += 1
                continue

            sigma = bagua_reduce(raw_sigma)
            mag = zeta_mag(sigma, t)
            if mag < threshold:
                best_sigma, best_t, best_mag = refine_2d(sigma, t)
                if best_mag < threshold:
                    if any(abs(best_t - zero[1]) < 0.3 for zero in found_zeros):
                        continue
                    found += 1
                    found_zeros.append((best_sigma, best_t, best_mag))

        if fine_mode:
            t += 0.02
        else:
            t += 0.15 + random.random() * 0.1

    return found_zeros, total, found, pruned_by, cs_peaks


# ----------------------------------------------------------------------
# REPORTING
# ----------------------------------------------------------------------


def recall_score(found, known, t_min, t_max, tol=0.3):
    """Count how many known zero heights were detected within tolerance."""
    in_range = [zero for zero in known if t_min <= zero <= t_max]
    hits = 0
    for known_t in in_range:
        if any(abs(found_zero[1] - known_t) < tol for found_zero in found):
            hits += 1
    return hits, len(in_range)


def run_and_report(t_min=10.5, t_max=52.0, **kwargs):
    zeros, total, found, pruned_by, cs_peaks = stochastic_search_v2(
        t_min=t_min,
        t_max=t_max,
        **kwargs,
    )
    hits, total_known = recall_score(zeros, KNOWN_ZEROS, t_min, t_max)
    pruned_total = sum(pruned_by.values())

    print("=" * 56)
    print("  KP STOCHASTIC ZERO SEARCH v2.0")
    print("  Order: 釋 -> 行 -> 度 -> 構  (BaZi first)")
    print("=" * 56)
    print(f"  t range:            {t_min} to {t_max}")
    print(f"  Total candidates:   {total}")
    print(f"  Pruned:             {pruned_total}  ({100 * pruned_total / max(total, 1):.1f}%)")
    print(f"    釋 BaZi (raw σ):  {pruned_by['A']}")
    print(f"    行 Qimen (CS):    {pruned_by['Q']}")
    print(f"    度 Ziwei (space): {pruned_by['Z']}")
    print(f"  Near-zeros found:   {found}")
    print(f"  Recall:             {hits}/{total_known}  ({100 * hits / max(total_known, 1):.0f}%)")
    print()
    print(f"  {'t_found':>10}  {'σ':>8}  {'|σ-1/2|':>10}  {'|ζ|':>10}  match")
    print(f"  {'-' * 8:>10}  {'-' * 6:>8}  {'-' * 8:>10}  {'-' * 8:>10}")

    deviations = []
    for sigma, t, mag in zeros:
        dev = abs(sigma - 0.5)
        deviations.append(dev)
        match = ""
        for known_t in KNOWN_ZEROS:
            if abs(t - known_t) < 0.3:
                match = f"<- known {known_t:.4f}"
                break
        print(f"  {t:10.4f}  {sigma:8.5f}  {dev:10.5f}  {mag:10.7f}  {match}")

    if deviations:
        print()
        print(f"  Mean |σ-1/2|: {sum(deviations) / len(deviations):.6f}")
        print(f"  Max  |σ-1/2|: {max(deviations):.6f}")
        print(f"  Max CS seen:  {max(cs_peaks):.4f}")
    print()
    print("  Interpretation: bounded numerical evidence only; not a proof of RH.")
    return zeros


def parse_args():
    parser = argparse.ArgumentParser(
        description="KP v2 stochastic search for small |zeta(sigma + i*t)| values."
    )
    parser.add_argument("--t-min", type=float, default=10.5, help="Starting t value.")
    parser.add_argument("--t-max", type=float, default=52.0, help="Ending t value.")
    parser.add_argument("--threshold", type=float, default=0.06, help="Magnitude threshold for near-zero detection.")
    parser.add_argument("--cs-min", type=float, default=0.05, help="Minimum Qimen continuation-strength score.")
    parser.add_argument("--cs-zoom", type=float, default=0.85, help="CS threshold for entering fine t-step mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Estimate coarse-mode candidate count without running the search.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    estimate = estimated_candidate_count(args.t_min, args.t_max)
    if args.estimate_only:
        print(f"Estimated coarse-mode candidates: {estimate}")
        print("Actual v2 count can be higher because adaptive fine mode uses smaller t steps.")
        raise SystemExit(0)

    run_and_report(
        t_min=args.t_min,
        t_max=args.t_max,
        threshold=args.threshold,
        cs_min=args.cs_min,
        cs_zoom=args.cs_zoom,
        seed=args.seed,
    )
