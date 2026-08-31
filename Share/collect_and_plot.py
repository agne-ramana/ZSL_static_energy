#!/usr/bin/env python3
"""
Collect CP2K static energies across the lattice-match scan and compare them
per-atom (eV/atom), since each match has a different atom count and raw
total energies aren't comparable across them.

Usage
-----
    cd STO_HfO2_m111_pbesol_matches      # the folder with manifest.json
    python3 collect_and_plot.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

HARTREE_EV = 27.211386245988
E_PAT = re.compile(r"ENERGY\|\s*Total FORCE_EVAL.*?:\s*(-?\d+\.\d+)")


def parse_energy(path):
    if not os.path.exists(path):
        return None
    e = None
    with open(path, errors="ignore") as f:
        for line in f:
            m = E_PAT.search(line)
            if m:
                e = float(m.group(1))
    return e


def scf_ok(path):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as f:
        return "SCF run NOT converged" not in f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.outdir or os.path.dirname(os.path.abspath(__file__)))
    mpath = os.path.join(root, "manifest.json")
    if not os.path.exists(mpath):
        sys.exit(f"no manifest.json in {root}")
    man = json.load(open(mpath))

    print(f"run folder   : {root}")
    print(f"termination  : {man['termination']}")
    print(f"matches      : {len(man['configs'])}")
    natoms_range = [c["natoms"] for c in man["configs"]]
    print(f"natoms range : {min(natoms_range)} - {max(natoms_range)}")

    rows = []
    for cfg in man["configs"]:
        out = os.path.join(root, "configs", cfg["tag"], "cp2k.out")
        e = parse_energy(out)
        rows.append(dict(cfg, E_Ha=e, converged=scf_ok(out) if e is not None else False))

    done = [r for r in rows if r["E_Ha"] is not None]
    print(f"\n{len(done)}/{len(rows)} jobs have an energy in cp2k.out")
    if not done:
        sys.exit("no energies found yet (looked for configs/<tag>/cp2k.out)")

    for r in rows:
        if r["E_Ha"] is None:
            r["E_eV"] = r["eV_per_atom"] = None
            continue
        r["E_eV"] = r["E_Ha"] * HARTREE_EV
        r["eV_per_atom"] = r["E_eV"] / r["natoms"]

    fieldnames = ["tag", "match_idx", "natoms", "area_A2", "eps_a", "eps_b", "max_eps",
                 "a", "b", "c", "gamma", "n_Hf", "n_O", "n_Sr", "n_Ti",
                 "E_Ha", "E_eV", "eV_per_atom", "converged"]
    with open(os.path.join(root, "energies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote energies.csv")

    bad = [r["tag"] for r in done if not r["converged"]]
    if bad:
        print(f"\nWARNING: SCF not converged in {len(bad)} job(s): "
              f"{', '.join(bad[:8])}{' ...' if len(bad) > 8 else ''}")

    done.sort(key=lambda r: r["max_eps"])
    print("\n--- per-atom energy vs. lattice match (sorted by strain, low to high) ---")
    print(f"  {'tag':28s} {'natoms':>6s} {'area':>8s} {'max|eps|':>9s} {'eV/atom':>12s}")
    for r in done:
        print(f"  {r['tag']:28s} {r['natoms']:6d} {r['area_A2']:8.1f} {r['max_eps']:8.2f}% "
              f"{r['eV_per_atom']:12.6f}")
    best = min(done, key=lambda r: r["eV_per_atom"])
    print(f"\n  --> lowest eV/atom: {best['tag']} (max|eps|={best['max_eps']:.2f}%, "
          f"area={best['area_A2']:.1f} A^2)")

    if not args.no_plot:
        plot_energies(done, root, man["termination"], len(rows))


def plot_energies(done, root, termination, n_total):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib unavailable - CSV + tables written, plots skipped)")
        return

    eps = [r["max_eps"] for r in done]
    area = [r["area_A2"] for r in done]
    eva = [r["eV_per_atom"] for r in done]
    natoms = [r["natoms"] for r in done]
    colors = [plt.cm.tab10(i % 10) for i in range(len(done))]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].scatter(eps, eva, c=colors)
    axes[0].set_xlabel("max|eps| (%)")
    axes[0].set_ylabel("eV / atom")
    axes[0].set_title("energy vs. strain")
    axes[0].grid(True)

    axes[1].scatter(area, eva, c=colors)
    axes[1].set_xlabel("area (A^2)")
    axes[1].set_ylabel("eV / atom")
    axes[1].set_title("energy vs. cell size")
    axes[1].grid(True)

    for ax, x in zip(axes, (eps, area)):
        for xi, yi, n in zip(x, eva, natoms):
            ax.annotate(str(n), (xi, yi), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

    fig.suptitle(f"m-HfO2(-111)/SrTiO3(001) -- {termination} -- "
                 f"lattice-match scan ({len(done)}/{n_total} jobs)")
    fig.tight_layout()
    fig.savefig(os.path.join(root, "energies.png"), dpi=150)
    print("\nwrote energies.png")


if __name__ == "__main__":
    main()
