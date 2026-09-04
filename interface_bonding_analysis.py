#!/usr/bin/env python3
"""
interface_bonding_analysis.py

Analyzes cation-oxygen bonding and coordination numbers for the relaxed
STO/HfO2(-111) structure (last frame of the Tight_geo_opt/Run_2/Extended_run
trajectory), and identifies which atoms sit at the interface from bond
topology rather than a hand-picked z-window:

  - An O atom is "interfacial" (bridging) if it bonds to at least one Ti
    AND at least one Hf -- i.e. it directly connects the two materials.
  - Any Ti/Hf cation bonded to a bridging O is flagged interfacial too.
  - Sr is deliberately excluded from the interfacial definition: this is
    the TiO2-terminated STO(-111) surface (see folder name), so the bonding
    interface layer is Ti-O-Hf. Sr sits below that layer and, because of
    its large 12-fold coordination sphere, can still fall within normal
    Sr-O bonding distance of a bridging O without being structurally part
    of the interface -- counting it would be misleading. Sr-O bonds are
    still computed and reported (general CN stats, and listed per bridging
    O) but never drive the interfacial classification.

Cation-O bond cutoffs are determined automatically per pair (Sr-O, Ti-O,
Hf-O) from the first minimum of that pair's radial distribution function,
so they reflect the actual relaxed geometry rather than literature guesses.

Outputs (written next to this script):
  interface_bonding_summary.txt   - cutoffs, bond/CN statistics, bridging-O list
  coordination_numbers.csv        - per-atom element, position, CN, interfacial flag
  interface_bonding_plot.png      - visual check: full slab, zoomed interface, top view
  bond_cutoff_diagnostics.png     - RDFs used to pick each cutoff
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry import cellpar_to_cell
from ase.neighborlist import neighbor_list

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XYZ_PATH = os.path.join(SCRIPT_DIR, '..', 'STO_HfO2_Interface-pos-1.xyz')
FRAME_INDEX = -1  # last frame = most relaxed geometry in this trajectory

# Cell parameters (a, b, c, alpha, beta, gamma) from interface.cif, which was
# generated for this same structure (xyz trajectory itself carries no cell info).
CELLPAR = [19.52500000, 19.91167120, 47.07119550, 90.0, 90.0, 78.690068]

INTERFACE_STO_CATION = {'Ti'}   # only Ti defines the STO side of the bonding interface (TiO2-terminated)
FILM_CATIONS = {'Hf'}
CATION_O_PAIRS = [('Sr', 'O'), ('Ti', 'O'), ('Hf', 'O')]

SEARCH_RADIUS = 4.0   # Angstrom, radius used to build each pair's RDF
BIN_WIDTH = 0.03       # Angstrom, RDF histogram bin width
FALLBACK_CUTOFFS = {'Sr-O': 3.20, 'Ti-O': 2.60, 'Hf-O': 2.60}  # used only if auto-detection fails

ELEMENT_COLORS = {'Sr': '#3fa34d', 'Ti': '#6b7280', 'Hf': '#8a5fd6', 'O': '#d64550'}
ELEMENT_MARKER_SIZE = {'Sr': 55, 'Ti': 40, 'Hf': 55, 'O': 22}

# Bond valence parameters s = exp((R0 - R) / B), Brese & O'Keeffe (1991) Acta Cryst. B47, 192.
# Used only as an independent chemical sanity check on the auto-detected cutoffs/bonds: the
# valence sum at each O should land near its formal charge (2.0) if the bonding makes sense.
BOND_VALENCE_R0 = {'Ti': 1.815, 'Hf': 1.923, 'Sr': 2.118}  # cation(4+/4+/2+) - O2-, all B = 0.37
BOND_VALENCE_B = 0.37
FORMAL_VALENCE = {'Sr': 2, 'Ti': 4, 'Hf': 4, 'O': -2}


def bond_valence_sum(o_idx, neighbor_map, symbols, distances=None):
    """Sum of bond valences s_i = exp((R0-R_i)/B) over an O atom's cation neighbors."""
    total = 0.0
    contributions = []
    for n, d in neighbor_map[o_idx]:
        sym = symbols[n]
        r0 = BOND_VALENCE_R0[sym]
        s = np.exp((r0 - d) / BOND_VALENCE_B)
        total += s
        contributions.append((sym, n, d, s))
    return total, contributions


def find_first_shell_cutoff(distances, fallback, bin_width=BIN_WIDTH, max_r=SEARCH_RADIUS,
                             search_upper=3.5):
    """Return the first RDF minimum after the first peak, i.e. a bond cutoff."""
    if len(distances) < 10:
        centers = np.arange(bin_width / 2, max_r, bin_width)
        return fallback, centers, np.zeros_like(centers)

    bins = np.arange(0.0, max_r + bin_width, bin_width)
    hist, edges = np.histogram(distances, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    kernel = np.ones(5) / 5.0
    smooth = np.convolve(hist.astype(float), kernel, mode='same')

    search_n = max(1, int(search_upper / bin_width))
    peak_idx = int(np.argmax(smooth[:search_n]))

    idx = peak_idx
    while idx < len(smooth) - 1 and smooth[idx + 1] <= smooth[idx]:
        idx += 1

    if idx == peak_idx or idx >= len(centers) - 1:
        return fallback, centers, smooth

    return float(centers[idx]), centers, smooth


def build_bonds(atoms, cutoffs):
    """Cation-O bonds only, using per-pair cutoffs. Returns neighbor_map: idx -> [(idx, dist)]."""
    symbols = np.array(atoms.get_chemical_symbols())
    max_cut = max(cutoffs.values())
    i_nb, j_nb, d_nb = neighbor_list('ijd', atoms, cutoff=max_cut)

    neighbor_map = {idx: [] for idx in range(len(atoms))}
    n_atoms = len(atoms)
    seen = set()
    for ii, jj, dd in zip(i_nb, j_nb, d_nb):
        sym_i, sym_j = symbols[ii], symbols[jj]
        if sym_i in ('Sr', 'Ti', 'Hf') and sym_j == 'O':
            key = f'{sym_i}-O'
            if dd <= cutoffs[key]:
                pair_key = (min(ii, jj), max(ii, jj), round(float(dd), 5))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                neighbor_map[ii].append((int(jj), float(dd)))
                neighbor_map[jj].append((int(ii), float(dd)))
    return neighbor_map


def mic_vector(atoms, i, j):
    """Displacement vector from atom i to its nearest periodic image of atom j."""
    d = atoms.get_distance(i, j, mic=True, vector=True)
    return d


def main():
    atoms = read(XYZ_PATH, index=FRAME_INDEX)
    atoms.set_cell(cellpar_to_cell(CELLPAR))
    atoms.set_pbc(True)
    atoms.wrap()

    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    # --- Auto-determine cation-O bond cutoffs from each pair's RDF ---
    i_all, j_all, d_all = neighbor_list('ijd', atoms, cutoff=SEARCH_RADIUS)
    cutoffs = {}
    rdf_data = {}
    for a, b in CATION_O_PAIRS:
        mask = (symbols[i_all] == a) & (symbols[j_all] == b)
        dists = d_all[mask]
        key = f'{a}-O'
        cutoff, centers, hist = find_first_shell_cutoff(dists, FALLBACK_CUTOFFS[key])
        cutoffs[key] = cutoff
        rdf_data[key] = (centers, hist, dists)

    # --- Bonds and coordination numbers ---
    neighbor_map = build_bonds(atoms, cutoffs)
    cn = np.array([len(neighbor_map[idx]) for idx in range(n_atoms)])

    # --- Interfacial atoms from bond topology (Ti-O-Hf bridges only; Sr excluded, see docstring) ---
    bridging_O = []
    for idx in range(n_atoms):
        if symbols[idx] != 'O':
            continue
        neigh_syms = {symbols[n] for n, _ in neighbor_map[idx]}
        if (neigh_syms & INTERFACE_STO_CATION) and (neigh_syms & FILM_CATIONS):
            bridging_O.append(idx)
    bridging_O = sorted(bridging_O)

    interfacial = set(bridging_O)
    for o_idx in bridging_O:
        for n, _ in neighbor_map[o_idx]:
            if symbols[n] in ('Ti', 'Hf'):  # Sr never counted as interfacial, even if bonded to a bridging O
                interfacial.add(n)
    interfacial = sorted(interfacial)
    interfacial_mask = np.zeros(n_atoms, dtype=bool)
    interfacial_mask[interfacial] = True

    # --- Bond valence sums: independent chemical-sensibility check on the bonds found above ---
    bvs = np.full(n_atoms, np.nan)
    for idx in range(n_atoms):
        if symbols[idx] == 'O':
            bvs[idx], _ = bond_valence_sum(idx, neighbor_map, symbols)
        elif symbols[idx] in BOND_VALENCE_R0:
            # cation BVS: sum over the same bonds, counted from the cation's own neighbor list
            total = 0.0
            for n, d in neighbor_map[idx]:
                r0 = BOND_VALENCE_R0[symbols[idx]]
                total += np.exp((r0 - d) / BOND_VALENCE_B)
            bvs[idx] = total

    # ---------------------------------------------------------------
    # Summary text
    # ---------------------------------------------------------------
    summary_path = os.path.join(SCRIPT_DIR, 'interface_bonding_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== INTERFACE BONDING & COORDINATION ANALYSIS ===\n")
        f.write(f"Structure: {os.path.abspath(XYZ_PATH)} (frame index {FRAME_INDEX})\n")
        f.write(f"Total atoms: {n_atoms}\n\n")

        f.write("Auto-detected cation-O bond cutoffs (first RDF minimum):\n")
        for key, val in cutoffs.items():
            f.write(f"  {key}: {val:.3f} A\n")
        f.write("\n")

        f.write("Coordination number statistics (cation CN = # O neighbors; O CN = # cation neighbors):\n")
        for el in ['Sr', 'Ti', 'Hf', 'O']:
            el_mask = symbols == el
            if not el_mask.any():
                continue
            cn_el = cn[el_mask]
            f.write(f"  {el:<3s} (n={el_mask.sum():4d})  "
                    f"mean CN = {cn_el.mean():.2f}  min = {cn_el.min()}  max = {cn_el.max()}\n")
            vals, counts = np.unique(cn_el, return_counts=True)
            hist_str = ', '.join(f"CN={v}: {c}" for v, c in zip(vals, counts))
            f.write(f"       distribution -> {hist_str}\n")
        f.write("\n")

        f.write("Bond valence sums (Brese & O'Keeffe; target = formal oxidation state, "
                 "Sr2+/Ti4+/Hf4+/O2-):\n")
        f.write("  Values near the target confirm the auto-detected cutoffs/bonds are chemically sound;\n")
        f.write("  large deviations would flag a bad cutoff or a genuinely under/over-bonded site.\n")
        for el in ['Sr', 'Ti', 'Hf', 'O']:
            el_mask = symbols == el
            bvs_el = bvs[el_mask]
            target = FORMAL_VALENCE[el] if el != 'O' else 2.0
            f.write(f"  {el:<3s}  target=|{target}|  mean={np.abs(bvs_el).mean():.2f}  "
                    f"std={bvs_el.std():.2f}  min={bvs_el.min():.2f}  max={bvs_el.max():.2f}\n")
        bvs_bridge = bvs[bridging_O]
        f.write(f"  O (bridging only, n={len(bridging_O)})  target=2.0  mean={bvs_bridge.mean():.2f}  "
                f"std={bvs_bridge.std():.2f}  min={bvs_bridge.min():.2f}  max={bvs_bridge.max():.2f}\n")
        f.write("\n")

        f.write(f"Bridging O atoms (bonded to >=1 Ti AND >=1 Hf; Sr excluded from this criterion): {len(bridging_O)}\n")
        f.write(f"Total interfacial atoms (bridging O + the Ti/Hf cations bonded to them): {len(interfacial)}\n")
        for el in ['Ti', 'Hf', 'O']:
            n_el = sum(1 for idx in interfacial if symbols[idx] == el)
            if n_el:
                f.write(f"  interfacial {el}: {n_el}\n")
        f.write("  interfacial Sr: 0 (by design -- see docstring)\n\n")

        # Bond-count breakdown per bridging O: how many Ti bonds / Hf bonds / (incidental) Sr bonds
        from collections import Counter, defaultdict
        pattern_counts = Counter()
        pattern_bvs = defaultdict(list)
        for o_idx in bridging_O:
            n_ti = sum(1 for n, _ in neighbor_map[o_idx] if symbols[n] == 'Ti')
            n_hf = sum(1 for n, _ in neighbor_map[o_idx] if symbols[n] == 'Hf')
            n_sr = sum(1 for n, _ in neighbor_map[o_idx] if symbols[n] == 'Sr')
            pattern_counts[(n_ti, n_hf, n_sr)] += 1
            pattern_bvs[(n_ti, n_hf, n_sr)].append(bvs[o_idx])

        f.write("Bridging-O bond-pattern breakdown (Ti bonds / Hf bonds / incidental Sr bonds -> # of O atoms,\n")
        f.write("mean BVS = bond valence sum for that pattern, target 2.0 -- see note above):\n")
        f.write("  For reference, known bulk motifs: STO bridging O = 2 Ti (+ up to 4 Sr); monoclinic\n")
        f.write("  HfO2 has O sites that are natively 3-fold or 4-fold coordinated by Hf ALONE (no Ti).\n")
        f.write("  So e.g. '1 Ti + 3 Hf' is simply an HfO2-like 4-fold O reaching one Ti across the\n")
        f.write("  boundary -- not an artifact -- and a mean BVS near 2.0 confirms it's not overbonded.\n")
        for (n_ti, n_hf, n_sr), c in sorted(pattern_counts.items(), key=lambda x: (-x[1], x[0])):
            sr_note = f", {n_sr} Sr (not counted)" if n_sr else ""
            mean_bvs = np.mean(pattern_bvs[(n_ti, n_hf, n_sr)])
            f.write(f"  {n_ti} Ti + {n_hf} Hf{sr_note}  ->  {c} oxygen(s), mean BVS = {mean_bvs:.2f}\n")
        f.write("\n")

        f.write("Bridging O detail (index is 0-based, matches coordination_numbers.csv):\n")
        f.write("  format: O<idx>  z=<z>  total_CN=<n>  BVS=<bond valence sum, target 2.0>\n")
        f.write("          |  Ti bonds  |  Hf bonds  |  Sr bonds (excluded, shown for reference)\n")
        for o_idx in bridging_O:
            neighbors = sorted(neighbor_map[o_idx], key=lambda x: x[1])
            ti_bonds = [(n, d) for n, d in neighbors if symbols[n] == 'Ti']
            hf_bonds = [(n, d) for n, d in neighbors if symbols[n] == 'Hf']
            sr_bonds = [(n, d) for n, d in neighbors if symbols[n] == 'Sr']

            ti_str = ', '.join(f"Ti{n}({d:.3f} A)" for n, d in ti_bonds) or 'none'
            hf_str = ', '.join(f"Hf{n}({d:.3f} A)" for n, d in hf_bonds) or 'none'
            sr_str = ', '.join(f"Sr{n}({d:.3f} A)" for n, d in sr_bonds) or 'none'

            f.write(f"  O{o_idx:4d}  z={positions[o_idx, 2]:7.3f}  total_CN={len(neighbors)}  BVS={bvs[o_idx]:.2f}"
                    f"  ({len(ti_bonds)} Ti, {len(hf_bonds)} Hf, {len(sr_bonds)} Sr)\n")
            f.write(f"           Ti -> {ti_str}\n")
            f.write(f"           Hf -> {hf_str}\n")
            f.write(f"           Sr -> {sr_str}  [excluded from interfacial definition]\n")

    print(f"Saved summary to: {summary_path}")

    # ---------------------------------------------------------------
    # Per-atom CSV
    # ---------------------------------------------------------------
    csv_path = os.path.join(SCRIPT_DIR, 'coordination_numbers.csv')
    with open(csv_path, 'w') as f:
        f.write("index,element,x,y,z,coordination_number,is_interfacial\n")
        for idx in range(n_atoms):
            x, y, z = positions[idx]
            f.write(f"{idx},{symbols[idx]},{x:.6f},{y:.6f},{z:.6f},{cn[idx]},{int(interfacial_mask[idx])}\n")
    print(f"Saved per-atom coordination numbers to: {csv_path}")

    # ---------------------------------------------------------------
    # RDF / cutoff diagnostic plot
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for ax, (a, b) in zip(axes, CATION_O_PAIRS):
        key = f'{a}-O'
        centers, hist, dists = rdf_data[key]
        ax.hist(dists, bins=np.arange(0.0, SEARCH_RADIUS + BIN_WIDTH, BIN_WIDTH),
                color=ELEMENT_COLORS[a], alpha=0.7)
        ax.axvline(cutoffs[key], color='black', linestyle='--', lw=1.2,
                   label=f'cutoff = {cutoffs[key]:.2f} A')
        ax.set_title(f'{key} distances')
        ax.set_xlabel('distance (A)')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.set_xlim(0, SEARCH_RADIUS)
    plt.tight_layout()
    diag_path = os.path.join(SCRIPT_DIR, 'bond_cutoff_diagnostics.png')
    plt.savefig(diag_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cutoff diagnostics plot to: {diag_path}")

    # ---------------------------------------------------------------
    # Main visual-check plot: full side view, zoomed interface, top view
    # ---------------------------------------------------------------
    z_bridge = positions[bridging_O, 2] if bridging_O else np.array([positions[:, 2].mean()])
    z_lo, z_hi = z_bridge.min() - 3.0, z_bridge.max() + 3.0

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1])
    ax_full = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_top = fig.add_subplot(gs[2])

    def scatter_atoms(ax, mask, highlight_mask):
        for el in ['Sr', 'Ti', 'Hf', 'O']:
            sel = mask & (symbols == el) & ~highlight_mask
            if sel.any():
                ax.scatter(positions[sel, 0], positions[sel, 2], s=ELEMENT_MARKER_SIZE[el] * 0.35,
                           c=ELEMENT_COLORS[el], alpha=0.35, linewidths=0, label=None)
        for el in ['Sr', 'Ti', 'Hf', 'O']:
            sel = mask & (symbols == el) & highlight_mask
            if sel.any():
                ax.scatter(positions[sel, 0], positions[sel, 2], s=ELEMENT_MARKER_SIZE[el] * 1.4,
                           c=ELEMENT_COLORS[el], edgecolors='black', linewidths=1.0, zorder=5,
                           label=f'{el} (interfacial)')

    # Panel 1: full slab, x vs z, interfacial atoms highlighted
    all_mask = np.ones(n_atoms, dtype=bool)
    scatter_atoms(ax_full, all_mask, interfacial_mask)
    ax_full.axhspan(z_lo, z_hi, color='yellow', alpha=0.12, zorder=0)
    ax_full.set_xlabel('x (A)')
    ax_full.set_ylabel('z (A)')
    ax_full.set_title('Full slab (side view)\nyellow band = zoomed region')

    # Panel 2: zoomed interface region with bonds drawn
    zoom_mask = (positions[:, 2] >= z_lo) & (positions[:, 2] <= z_hi)
    scatter_atoms(ax_zoom, zoom_mask, interfacial_mask & zoom_mask)
    drawn = set()
    for o_idx in bridging_O:
        for n, d in neighbor_map[o_idx]:
            if symbols[n] not in ('Ti', 'Hf'):  # skip incidental Sr-O contacts, not part of the interface bond
                continue
            pair = (min(o_idx, n), max(o_idx, n))
            if pair in drawn:
                continue
            drawn.add(pair)
            vec = mic_vector(atoms, o_idx, n)
            p0 = positions[o_idx]
            p1 = p0 + vec
            color = 'tab:blue' if symbols[n] in INTERFACE_STO_CATION else 'tab:purple'
            ax_zoom.plot([p0[0], p1[0]], [p0[2], p1[2]], color=color, lw=1.0, alpha=0.8, zorder=4)
    ax_zoom.set_xlabel('x (A)')
    ax_zoom.set_ylabel('z (A)')
    ax_zoom.set_title(f'Interface region zoom\n(blue bond = O-Ti, purple bond = O-Hf; Sr-O contacts not drawn)')
    handles, labels = ax_zoom.get_legend_handles_labels()
    if handles:
        ax_zoom.legend(fontsize=8, loc='upper right')

    # Panel 3: top-down view of interfacial atoms only
    scatter_atoms(ax_top, interfacial_mask, interfacial_mask)
    ax_top.set_xlabel('x (A)')
    ax_top.set_ylabel('y (A)')
    ax_top.set_title('Interfacial atoms only (top view)')
    ax_top.set_aspect('equal', adjustable='box')

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor=ELEMENT_COLORS[el],
                            markeredgecolor='black', markersize=8, label=el) for el in ['Sr', 'Ti', 'Hf', 'O']]
    fig.legend(handles=legend_elems, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plot_path = os.path.join(SCRIPT_DIR, 'interface_bonding_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visual-check plot to: {plot_path}")


if __name__ == '__main__':
    main()
