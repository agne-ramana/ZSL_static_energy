#!/usr/bin/env python3
"""
m-HfO2(-111) / SrTiO3(001) interfaces, one termination, across the Pareto
front of ZSL lattice-match choices (different supercell areas/strains).
Builds configs/<tag>/{interface.cif, POSCAR, cp2k.inp, submit.sh,
preview.png}, manifest.csv/json, run_all.sh, and copies in
collect_and_plot.py + plot.sbatch.

Usage
-----
    python3 build_interfaces.py --list
    python3 build_interfaces.py --termination 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.analysis.interfaces.zsl import ZSLGenerator as ZSLG
from pymatgen.analysis.interfaces.coherent_interfaces import (
    CoherentInterfaceBuilder as CIB,
)
from pymatgen.io.ase import AseAtomsAdaptor as AAA

from ase.visualize.plot import plot_atoms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(SCRIPT_DIR, "..", "Reference_cell_reopt")
HFO2_CIF = os.path.join(REFDIR, "m-HfO2", "m-HfO2_final_standardized.cif")
STO_CIF = os.path.join(REFDIR, "SrTiO3", "SrTiO3_final_standardized.cif")

with open(os.path.join(SCRIPT_DIR, "cp2k_template.inp")) as f:
    CP2K = f.read()


def bulk_structures():
    hfo2 = Structure.from_file(HFO2_CIF)
    sto = Structure.from_file(STO_CIF)
    return hfo2, sto


def match_stats(m):
    """area (A^2), eps_a, eps_b (%), max|eps| (%) for a raw ZSLMatch."""
    sub = np.array(m.substrate_sl_vectors)
    film = np.array(m.film_sl_vectors)
    ls, lf = np.linalg.norm(sub, axis=1), np.linalg.norm(film, axis=1)
    area = float(np.linalg.norm(np.cross(sub[0], sub[1])))
    eps_a = 100 * (ls[0] - lf[0]) / lf[0]
    eps_b = 100 * (ls[1] - lf[1]) / lf[1]
    return area, eps_a, eps_b, max(abs(eps_a), abs(eps_b))


def pareto_front(zsl_matches):
    """Smallest-area match achieving each successive improvement in max|eps|,
    scanning matches in ascending area."""
    stats = [match_stats(m) for m in zsl_matches]
    order = np.argsort([s[0] for s in stats])
    idx, best = [], np.inf
    for i in order:
        area, ea, eb, me = stats[i]
        if me < best - 1e-9:
            idx.append(i)
            best = me
    return idx, stats


def find_zsl_matches(hfo2, sto, args):
    """Set up the lattice-match search. Returns the interface builder (with
    all raw zsl_matches on it) and the TiO2-terminated film terminations."""
    zsl = ZSLG(max_area=args.max_area, max_length_tol=args.length_tol,
              max_angle_tol=args.angle_tol)
    builder = CIB(film_structure=hfo2, substrate_structure=sto,
                  film_miller=(-1, 1, 1), substrate_miller=(1, 0, 0),
                  zslgen=zsl, filter_out_sym_slabs=False, label_index=True)
    tio2 = [t for t in builder.terminations if "TiO2" in str(t[1])]
    return builder, tio2


def select_chosen_matches(idx, builder, args):
    """Pareto-front indices, capped at --max-matches, resolved to ZSLMatch objects."""
    keep_idx = idx[: args.max_matches]
    return [builder.zsl_matches[i] for i in keep_idx]


def build_slabs(builder, term, chosen_matches, args):
    """Build one slab supercell per chosen match. Restricting builder.zsl_matches
    first keeps this fast -- building a slab per raw match (there can be
    thousands with wide tolerances) is the expensive step."""
    common = dict(vacuum_over_film=args.vacuum, film_thickness=args.film_layers,
                  substrate_thickness=args.sub_layers, in_layers=True)
    builder.zsl_matches = chosen_matches
    ifaces = list(builder.get_interfaces(termination=term, gap=args.gap, **common))
    assert len(ifaces) == len(chosen_matches), \
        f"expected 1 interface per match, got {len(ifaces)} for {len(chosen_matches)}"
    return ifaces


VIEWS = [("-90x", "side, along b"), ("-90x,-90y", "side, along a"), ("0x", "top")]


def render(structure, path, title=""):
    atoms = AAA.get_atoms(structure)
    atoms.wrap()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    for ax, (rot, label) in zip(axes, VIEWS):
        plot_atoms(atoms, ax, radii=0.42, rotation=rot, show_unit_cell=2)
        ax.set_title(label, fontsize=9, color="#6b7280")
        ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


SUBMIT = """#!/bin/bash
#SBATCH --job-name=v4_{tag}
#SBATCH --output=cp2k_run_%j.out
#SBATCH --error=cp2k_run_%j.err
#SBATCH --time={hours:02d}:00:00
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --account=e89-camm
#SBATCH --partition=standard
#SBATCH --qos=standard

module load epcc-job-env
module load cp2k/cp2k-2024.3
export OMP_NUM_THREADS=8
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

srun --distribution=block:block --hint=nomultithread cp2k.psmp -i cp2k.inp -o cp2k.out
"""

# Resumable, self-throttling submission: local ledger + squeue/sacct check,
# stops on first sbatch failure (e.g. QOS limit) so re-running picks up where
# it left off.
RUNALL = """#!/bin/bash
set -o pipefail
cd "$(dirname "$0")"

max_new=${1:-999999}
LEDGER=.submitted.tags
touch "$LEDGER"

LOCAL=()
while IFS= read -r line; do
    [ -n "$line" ] && LOCAL+=("$line")
done < "$LEDGER"

ACTIVE=()
while IFS= read -r line; do
    [ -n "$line" ] && ACTIVE+=("$line")
done < <(squeue -u "$USER" -h -o %j 2>/dev/null)

since=$(date -d '-14 days' +%Y-%m-%d 2>/dev/null || date -v-14d +%Y-%m-%d 2>/dev/null)
HIST=()
while IFS= read -r line; do
    [ -n "$line" ] && HIST+=("$line")
done < <(sacct -u "$USER" -n -X --format=JobName%64 --starttime="$since" 2>/dev/null)

is_known() {
    local tag="$1" x
    for x in "${LOCAL[@]}"; do [ "$x" = "$tag" ] && return 0; done
    for x in "${ACTIVE[@]}"; do [ "$x" = "v4_$tag" ] || [ "$x" = "$tag" ] && return 0; done
    for x in "${HIST[@]}"; do [ "$x" = "v4_$tag" ] || [ "$x" = "$tag" ] && return 0; done
    return 1
}

submitted=0
skipped=0
for d in configs/*/ ; do
    tag=$(basename "$d")
    if is_known "$tag"; then
        skipped=$((skipped + 1))
        continue
    fi
    if (( submitted >= max_new )); then
        echo "reached --max of $max_new new submissions this run, stopping (re-run to continue)"
        break
    fi
    printf 'submitting %-40s ... ' "$tag"
    if out=$(cd "$d" && sbatch submit.sh 2>&1); then
        echo "$out"
        echo "$tag" >> "$LEDGER"
        submitted=$((submitted + 1))
    else
        echo "FAILED: $out"
        echo "stopping here (this is what a QOS/submit-limit error looks like) -- re-run ./run_all.sh later to submit the rest"
        break
    fi
done
echo
echo "this run: submitted $submitted, skipped $skipped (ledger + scheduler)"
"""


def slurm_resources(natoms):
    nodes = int(np.clip(round(natoms / 220), 2, 8))
    hours = int(np.clip(round(natoms / 350), 1, 4))
    return nodes, hours


def print_terminations(tio2):
    print("=" * 78)
    print("FILM TERMINATIONS (TiO2-terminated substrate side; --termination index)")
    print("=" * 78)
    for i, t in enumerate(tio2):
        print(f"  [{i}]  {t}")


def print_pareto_front(idx, stats, n_raw, args):
    print(f"\n{'=' * 78}")
    print(f"LATTICE-MATCH PARETO FRONT  ({n_raw} raw matches searched, "
          f"max_area={args.max_area:.0f} A^2, length_tol={args.length_tol*100:.0f}%, "
          f"angle_tol={args.angle_tol:.1f} deg)")
    print("=" * 78)
    print(f"  {'#':>3s} {'area (A^2)':>10s} {'eps_a':>8s} {'eps_b':>8s} {'max|eps|':>9s}")
    for k, i in enumerate(idx):
        area, ea, eb, me = stats[i]
        print(f"  {k:3d} {area:10.1f} {ea:+7.2f}% {eb:+7.2f}% {me:8.2f}%")


def write_configs(root, ifaces, chosen_matches, term, args):
    """Write configs/<tag>/{interface.cif, POSCAR, cp2k.inp, submit.sh, preview.png}
    for each built interface, and return the manifest entries."""
    manifest = []
    for k, (iface, m) in enumerate(zip(ifaces, chosen_matches)):
        area, ea, eb, me = match_stats(m)
        tag = f"m{k:02d}_A{round(area):04d}_eps{me:.1f}".replace(".", "p")
        d = os.path.join(root, "configs", tag)
        os.makedirs(d, exist_ok=True)
        iface.to(filename=os.path.join(d, "interface.cif"), fmt="cif")
        iface.to(filename=os.path.join(d, "POSCAR"), fmt="poscar")
        with open(os.path.join(d, "cp2k.inp"), "w") as f:
            f.write(CP2K.format(tag=tag, restart_onoff="ON" if args.save_wfn else "OFF"))
        nodes, hours = slurm_resources(len(iface))
        p = os.path.join(d, "submit.sh")
        with open(p, "w") as f:
            f.write(SUBMIT.format(tag=tag, nodes=nodes, hours=hours))
        os.chmod(p, 0o755)
        if not args.no_preview:
            render(iface, os.path.join(d, "preview.png"), title=tag)

        comp = iface.composition.get_el_amt_dict()
        a, b, c = iface.lattice.abc
        al, be, ga = iface.lattice.angles
        manifest.append(dict(
            tag=tag, match_idx=k, termination=str(term),
            gap=args.gap, natoms=len(iface),
            a=round(a, 4), b=round(b, 4), c=round(c, 4),
            alpha=round(al, 3), beta=round(be, 3), gamma=round(ga, 3),
            area_A2=round(a * b * np.sin(np.radians(ga)), 3),
            eps_a=round(ea, 3), eps_b=round(eb, 3), max_eps=round(me, 3),
            n_Hf=int(comp.get("Hf", 0)), n_O=int(comp.get("O", 0)),
            n_Sr=int(comp.get("Sr", 0)), n_Ti=int(comp.get("Ti", 0)),
            nodes=nodes, hours=hours,
        ))
        print(f"  {tag:26s}  natoms={len(iface):4d}  area={area:7.1f} A^2  "
              f"max|eps|={me:5.2f}%  ({nodes} nodes, {hours}h)")
    return manifest


def write_run_outputs(root, manifest, term, args):
    """manifest.json/csv, run_all.sh, and copies of collect_and_plot.py + plot.sbatch."""
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(dict(termination=str(term), termination_idx=args.termination,
                       gap=args.gap, vacuum=args.vacuum, film_layers=args.film_layers,
                       sub_layers=args.sub_layers, max_area=args.max_area,
                       length_tol=args.length_tol, angle_tol=args.angle_tol,
                       configs=manifest), f, indent=2)
    with open(os.path.join(root, "manifest.csv"), "w") as f:
        keys = list(manifest[0].keys())
        f.write(",".join(keys) + "\n")
        for m in manifest:
            f.write(",".join(str(m[k]) for k in keys) + "\n")
    p = os.path.join(root, "run_all.sh")
    with open(p, "w") as f:
        f.write(RUNALL)
    os.chmod(p, 0o755)
    for fname in ("collect_and_plot.py", "plot.sbatch"):
        src = os.path.join(SCRIPT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(root, fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="STO_HfO2_m111_pbesol_matches")
    ap.add_argument("--termination", type=int, default=None,
                    help="index into the TiO2-terminated film list printed by --list")
    ap.add_argument("--gap", type=float, default=2.25)
    ap.add_argument("--vacuum", type=float, default=20.0)
    ap.add_argument("--film-layers", type=int, default=5)
    ap.add_argument("--sub-layers", type=int, default=3)
    ap.add_argument("--max-area", type=float, default=600.0,
                    help="wide on purpose -- this script wants the whole "
                         "strain/area tradeoff, not just the best match")
    ap.add_argument("--length-tol", type=float, default=0.10)
    ap.add_argument("--angle-tol", type=float, default=3.0)
    ap.add_argument("--max-matches", type=int, default=8,
                    help="cap on how many Pareto-front matches to build")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--save-wfn", action="store_true",
                    help="keep the CP2K wavefunction restart file per job (off by default)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    hfo2, sto = bulk_structures()
    builder, tio2 = find_zsl_matches(hfo2, sto, args)
    print_terminations(tio2)

    idx, stats = pareto_front(builder.zsl_matches)
    print_pareto_front(idx, stats, len(builder.zsl_matches), args)

    if args.termination is None or args.list:
        print("\nNo --termination given (or --list passed) -- nothing built. "
              "Re-run with --termination N once you've picked one.")
        return
    if not (0 <= args.termination < len(tio2)):
        sys.exit(f"--termination {args.termination} out of range 0-{len(tio2) - 1}")
    term = tio2[args.termination]
    print(f"\nbuilding termination [{args.termination}] = {term}")

    chosen_matches = select_chosen_matches(idx, builder, args)
    print(f"building {len(chosen_matches)} of {len(idx)} Pareto matches "
          f"(--max-matches {args.max_matches})")
    ifaces = build_slabs(builder, term, chosen_matches, args)

    root = os.path.abspath(args.outdir)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "configs"))

    manifest = write_configs(root, ifaces, chosen_matches, term, args)
    write_run_outputs(root, manifest, term, args)

    print(f"\n{len(manifest)} configurations in {root}/configs/")
    print(f"natoms range: {min(m['natoms'] for m in manifest)} - "
          f"{max(m['natoms'] for m in manifest)}")
    print(f"WFN restart files: {'SAVED (--save-wfn)' if args.save_wfn else 'NOT saved (default -- pass --save-wfn to keep them)'}")


if __name__ == "__main__":
    main()
