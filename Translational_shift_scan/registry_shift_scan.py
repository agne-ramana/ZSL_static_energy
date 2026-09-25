"""Scan the in-plane registry of the HfO2 film on the STO substrate and write
two candidate interfaces next to the input (reference):

  align_00      - most first-layer Hf directly on top of the O in the STO
                  termination plane (highest `align`)
  nooverlap_01  - first-layer Hf as far as possible from the cation M in that plane
                  (largest `rMmin`, then fewest clashes), taken among registries
                  at least MIN_SEP away from the reference and from align_00

The film is shifted rigidly in-plane only; z, the gap, the vacuum, the atom
order and the substrate constraints are unchanged, so each candidate drops
straight into the existing CP2K input.

Every distinct registry is reached by shifting the film within ONE STO surface
cell (shift = u*v1 + w*v2, u, w in [0, 1)), because the substrate repeats with
its own surface lattice. Shifts that differ by a translation of the film itself
give the same structure and are treated as one registry.

Scores per shift (lateral = in-plane distance, periodic):
  align   mean over plane O of exp(-r^2 / 2 SIGMA_O^2), r = lateral distance to the
          nearest first-layer Hf (1 = an Hf directly above every O).
          onTop = # plane O with r < ON_TOP_R.
  rMmin   smallest lateral distance between a first-layer Hf and a plane cation M.
          clash = # first-layer Hf with r < CLASH_R.
  OOstk   # film-bottom O within OO_STACK_R (lateral) of a plane O  (reported only)
  d_min   shortest film-substrate distance at the input gap        (reported only)
M = the cation of the STO termination plane: Sr for SrO, Ti for TiO2 (found
automatically). First-layer Hf = film Hf within HF_LAYER_DEPTH of the lowest film Hf; film-bottom
O = every film O below the lowest first-layer Hf.

Cell borders: the film is periodic in (a, b), so a rigid shift + in-plane wrap only
changes which image of each atom sits inside the cell. Lateral distances use the
true minimum image (the cell is oblique, so neighbouring images are searched
explicitly), and every written structure is checked: each film atom must keep the
exact film neighbour shell (< ENV_CUTOFF, full PBC) it has in the input.

Output in <run_dir>/registry_scan/: one folder per candidate with
STO_HfO2_interface_fixed.vasp, STO_HfO2_interface.cif/.xyz, HfO2_film.vasp,
STO_substrate.vasp and cp2k_fixed_atoms.inc; plus registry_scan_summary.txt,
registry_scan_grid.csv, registry_scan_maps.png and registry_scan_candidates.png.

Usage:
    python registry_shift_scan.py [STO_HfO2_interface_fixed.vasp] [--grid 55]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.neighborlist import neighbor_list

HERE = Path(__file__).resolve().parent

SIGMA_O = 0.5          # A, width of the Hf-on-O alignment Gaussian
ON_TOP_R = 0.6         # A, plane O counts as "Hf on top" below this lateral distance
CLASH_R = 1.5          # A, Hf counts as "over a plane cation" below this lateral distance
OO_STACK_R = 1.0       # A, film-bottom O counted as stacked on a plane O below this
HF_LAYER_DEPTH = 1.0   # A, first-layer Hf = film Hf within this of the lowest film Hf
PLANE_DEPTH = 0.6      # A, termination plane = substrate atoms within this of the top cation
MIN_SEP = 0.5          # A, nooverlap_01 must be at least this far (in shift) from the others
ENV_CUTOFF = 3.5       # A, neighbour shell compared in the cell-border check

COLORS = {"Sr": "#3fa34d", "Ti": "#6b7280", "Hf": "#8a5fd6", "O": "#d64550"}
IMAGES = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)


def lateral_mic(d, cell2):
    """Shortest periodic image of in-plane vectors d (..., 2); cell2 rows = a, b."""
    f = d @ np.linalg.inv(cell2)
    base = (f - np.round(f)) @ cell2
    cand = base[..., None, :] + IMAGES @ cell2                 # (..., 9, 2)
    k = np.argmin((cand ** 2).sum(axis=-1), axis=-1)
    return np.take_along_axis(cand, k[..., None, None], axis=-2)[..., 0, :]


def nearest_lateral(src_xy, dst_xy, cell2):
    """For each src point, lateral distance to its nearest dst point."""
    d = lateral_mic(dst_xy[None, :, :] - src_xy[:, None, :], cell2)
    return np.linalg.norm(d, axis=2).min(axis=1)


def surface_vectors(xy, cell2):
    """Two shortest non-collinear lattice vectors (right-handed) of a periodic 2D point set."""
    d = lateral_mic(xy[None, :, :] - xy[:, None, :], cell2).reshape(-1, 2)
    d = d[np.linalg.norm(d, axis=1) > 0.3]
    d = d[np.argsort(np.linalg.norm(d, axis=1))]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    v1 = d[0]
    v2 = next(v for v in d[1:] if abs(cross(v1, v)) > 0.3 * np.linalg.norm(v1) * np.linalg.norm(v))
    return (v1, v2) if cross(v1, v2) > 0 else (v1, -v2)


def film_translations(xy, z, sym, cell2, tol=0.1):
    """In-plane translations that map the film onto itself (per element and height)."""
    layers = [(sym == e) & (np.abs(z - zl) < 0.05)
              for e in np.unique(sym) for zl in np.unique(np.round(z[sym == e], 2))]
    out = []
    for j in np.where(layers[0])[0]:
        T = lateral_mic(xy[j] - xy[layers[0]][0], cell2)
        if all(nearest_lateral(xy[m] + T, xy[m], cell2).max() < tol for m in layers):
            out.append(T)
    return np.array(out)


def split_interface(atoms):
    """Boolean mask of film atoms: every Hf, and O above the top Sr/Ti plane."""
    sym = np.array(atoms.get_chemical_symbols())
    z = atoms.positions[:, 2]
    z_sub_cation = z[np.isin(sym, ["Sr", "Ti"])].max()
    return (sym == "Hf") | ((sym == "O") & (z > z_sub_cation + 0.5))


def fixed_ranges(indices):
    """1-based CP2K LIST string with a..b ranges, e.g. [0, 1, 2, 5] -> '1..3 6'."""
    idx = sorted(i + 1 for i in indices)
    out, start = [], None
    for k, i in enumerate(idx):
        if start is None:
            start = i
        if k == len(idx) - 1 or idx[k + 1] != i + 1:
            out.append(f"{start}..{i}" if i > start else f"{i}")
            start = None
    return " ".join(out)


def film_environment(atoms, is_film):
    """Per film atom: sorted distances (full PBC) to film neighbours within ENV_CUTOFF."""
    i, j, d = neighbor_list("ijd", atoms, ENV_CUTOFF)
    keep = is_film[i] & is_film[j]
    i, d = i[keep], d[keep]
    return {k: np.sort(d[i == k]) for k in np.where(is_film)[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("interface", nargs="?", default=str(HERE / "STO_HfO2_interface_fixed.vasp"),
                    help="interface POSCAR with selective dynamics (Lattice_match_monoclinic.py output)")
    ap.add_argument("--grid", type=int, default=55, help="grid points per STO surface vector")
    args = ap.parse_args()

    # --- read input ---
    src = Path(args.interface).resolve()
    iface = read(src)
    cell2 = iface.cell.array[:2, :2]                        # in-plane a, b
    fixed = np.concatenate([c.index for c in iface.constraints])   # FixAtoms indices

    sym = np.array(iface.get_chemical_symbols())
    pos = iface.positions
    z = pos[:, 2]
    is_film = split_interface(iface)

    # --- atom groups ---
    z_top = z[~is_film & np.isin(sym, ["Sr", "Ti"])].max()
    plane = ~is_film & (z > z_top - PLANE_DEPTH)
    plane_O = plane & (sym == "O")
    M = "Sr" if (plane & (sym == "Sr")).sum() >= (plane & (sym == "Ti")).sum() else "Ti"
    term = {"Sr": "SrO", "Ti": "TiO2"}[M]                  # termination plane label
    plane_M = plane & (sym == M)

    hf_film = is_film & (sym == "Hf")
    hf1 = hf_film & (z < z[hf_film].min() + HF_LAYER_DEPTH)          # first Hf layer
    o_bot = is_film & (sym == "O") & (z < z[hf1].min())               # film O below it
    near = hf1 | o_bot

    v1, v2 = surface_vectors(pos[plane_M, :2], cell2)
    film_T = film_translations(pos[is_film, :2], z[is_film], sym[is_film], cell2)

    def shift_dist(a, b):
        """Distance between two (u, w) shifts, modulo the STO surface cell and film translations."""
        d = (a[0] - b[0]) * v1 + (a[1] - b[1]) * v2
        return np.linalg.norm(lateral_mic(d - film_T, np.array([v1, v2])), axis=1).min()

    print(f"input              : {src}")
    print(f"{term + ' plane':<19s}: z = {z_top:.3f} A, {plane_O.sum()} O, {plane_M.sum()} {M}")
    print(f"first-layer Hf     : {hf1.sum()} atoms, {z[hf1].min() - z_top:.2f}-{z[hf1].max() - z_top:.2f} A above")
    print(f"film-bottom O      : {o_bot.sum()} atoms, {z[o_bot].min() - z_top:.2f}-{z[o_bot].max() - z_top:.2f} A above")
    print(f"STO surface vectors: v1 = ({v1[0]:+.3f}, {v1[1]:+.3f}), v2 = ({v2[0]:+.3f}, {v2[1]:+.3f}) A")
    print(f"film translations  : {len(film_T)}")

    # --- scan every shift in one STO surface cell ---
    g = args.grid
    uu, ww = np.meshgrid(np.arange(g) / g, np.arange(g) / g, indexing="ij")
    uw = np.column_stack([uu.ravel(), ww.ravel()])
    shifts = uw @ np.array([v1, v2])
    n = len(uw)
    align, r_m_min, d_min = np.empty(n), np.empty(n), np.empty(n)
    n_on_top, n_clash, oo_stack = np.empty(n, int), np.empty(n, int), np.empty(n, int)

    dz_near = z[near][:, None] - z[plane][None, :]
    for k, t in enumerate(shifts):
        xy_hf = pos[hf1, :2] + t
        r_o = nearest_lateral(pos[plane_O, :2], xy_hf, cell2)        # plane O -> nearest Hf
        r_m = nearest_lateral(xy_hf, pos[plane_M, :2], cell2)       # Hf -> nearest plane cation
        align[k] = np.mean(np.exp(-r_o ** 2 / (2 * SIGMA_O ** 2)))
        n_on_top[k] = np.sum(r_o < ON_TOP_R)
        r_m_min[k] = r_m.min()
        n_clash[k] = np.sum(r_m < CLASH_R)
        oo_stack[k] = np.sum(nearest_lateral(pos[o_bot, :2] + t, pos[plane_O, :2], cell2) < OO_STACK_R)
        lat = lateral_mic(pos[plane, :2][None, :, :] - (pos[near, :2] + t)[:, None, :], cell2)
        d_min[k] = np.sqrt((lat ** 2).sum(axis=2) + dz_near ** 2).min()

    # --- pick candidates ---
    ref = 0                                                   # zero shift = the input
    best_align = np.lexsort((-r_m_min, -align))[0]
    best_noov = next(i for i in np.lexsort((-align, n_clash, -r_m_min))
                     if shift_dist(uw[i], uw[ref]) >= MIN_SEP and shift_dist(uw[i], uw[best_align]) >= MIN_SEP)
    cands = [("reference", ref), ("align_00", best_align), ("nooverlap_01", best_noov)]

    # --- write structures ---
    out_root = src.parent / "registry_scan"
    out_root.mkdir(exist_ok=True)
    env_ref = film_environment(iface, is_film)
    for name, i in cands:
        model = iface.copy()                              # keeps order, cell and FixAtoms
        model.positions[is_film, :2] += shifts[i]
        model.wrap(pbc=[True, True, False])               # in-plane only; z is never touched

        env = film_environment(model, is_film)
        if any(len(env[k]) != len(d0) or np.abs(env[k] - d0).max() > 1e-4 for k, d0 in env_ref.items()):
            sys.exit(f"{name}: film neighbour shells changed after shift + wrap; "
                     "the film is not periodic across the cell borders")

        d = out_root / name
        d.mkdir(exist_ok=True)
        write(d / "STO_HfO2_interface_fixed.vasp", model, format="vasp", direct=True)
        write(d / "STO_HfO2_interface.cif", model, format="cif")
        write(d / "STO_HfO2_interface.xyz", model, format="extxyz")
        for part, mask in (("HfO2_film.vasp", is_film), ("STO_substrate.vasp", ~is_film)):
            sub = model[mask]
            sub.set_constraint()
            write(d / part, sub, format="vasp", direct=True, sort=True)
        (d / "cp2k_fixed_atoms.inc").write_text(
            "&FIXED_ATOMS\n"
            "  COMPONENTS_TO_FIX XYZ\n"
            f"  LIST {fixed_ranges(fixed)}\n"
            "&END FIXED_ATOMS\n")

    # --- tables ---
    cols = ["u", "w", "shift_x", "shift_y", "align", "n_on_top", "n_clash", f"min_r_Hf_{M}", "O_O_stack", "d_min"]
    grid = np.column_stack([uw, shifts, align, n_on_top, n_clash, r_m_min, oo_stack, d_min])
    np.savetxt(out_root / "registry_scan_grid.csv", grid, delimiter=",", header=",".join(cols),
               comments="", fmt=["%.4f"] * 5 + ["%d"] * 2 + ["%.4f", "%d", "%.4f"])

    lines = [f"Registry scan of {src}",
             f"{term} plane: {plane_O.sum()} O, {plane_M.sum()} {M}; {hf1.sum()} first-layer Hf; "
             f"{o_bot.sum()} film-bottom O; grid {g} x {g}",
             f"align: sigma {SIGMA_O} A, onTop < {ON_TOP_R} A;  rMmin = min lateral Hf-{M}, clash < {CLASH_R} A;  "
             f"OOstk < {OO_STACK_R} A;  d_min at the input gap",
             f"Cell-border check passed (film neighbour shells < {ENV_CUTOFF} A unchanged).",
             "Rigid, unrelaxed geometry: use this to choose starting registries, not to rank energies.",
             "",
             f"{'name':<14s} {'dx':>7s} {'dy':>7s} {'align':>6s} {'onTop':>5s} {'rMmin':>6s} "
             f"{'clash':>5s} {'OOstk':>5s} {'d_min':>6s}"]
    for name, i in cands:
        lines.append(f"{name:<14s} {shifts[i, 0]:+7.3f} {shifts[i, 1]:+7.3f} {align[i]:6.3f} {n_on_top[i]:5d} "
                     f"{r_m_min[i]:6.2f} {n_clash[i]:5d} {oo_stack[i]:5d} {d_min[i]:6.2f}")
    text = "\n".join(lines)
    print("\n" + text)
    (out_root / "registry_scan_summary.txt").write_text(text + "\n")

    # --- figure 1: score maps over 2 x 2 STO surface cells + scatter plots of the scores ---
    ue = np.arange(2 * g + 1) / g - 0.5 / g
    UE, WE = np.meshgrid(ue, ue, indexing="ij")
    XE, YE = UE * v1[0] + WE * v2[0], UE * v1[1] + WE * v2[1]
    sto_cell = np.array([[0, 0], v1, v1 + v2, v2, [0, 0]])
    markers = {"reference": "*", "align_00": "o", "nooverlap_01": "s"}

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.8))
    for ax, vals, title, cmap in [(axes[0, 0], align, "align (Hf on plane O), maximise", "viridis"),
                                  (axes[0, 1], r_m_min, f"min lateral Hf-{M} distance (A), maximise", "magma")]:
        pc = ax.pcolormesh(XE, YE, np.tile(vals.reshape(g, g), (2, 2)), cmap=cmap, shading="flat")
        fig.colorbar(pc, ax=ax, shrink=0.85)
        ax.plot(sto_cell[:, 0], sto_cell[:, 1], color="white", lw=1.0, ls="--")
        for name, i in cands:
            ax.scatter(*shifts[i], marker=markers[name], s=90, c="white", edgecolors="k", zorder=5)
            ax.annotate(name, shifts[i], xytext=(5, 4), textcoords="offset points", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8), clip_on=False)
        ax.set_aspect("equal")
        ax.set_xlabel("film shift x (A)")
        ax.set_ylabel("film shift y (A)")
        ax.set_title(f"{title}\n(2 x 2 STO surface cells; dashed = one cell)", fontsize=10)

    align_label = "align  (higher = more Hf on top of plane O)"
    rm_label = f"min lateral Hf-{M} distance (A)"
    clash_label = f"Hf clash count (Hf within {CLASH_R} A of a plane {M})"
    for ax, x, y, xlabel, ylabel, title in [
            (axes[0, 2], align, r_m_min, align_label, rm_label, "trade-off between the two objectives"),
            (axes[1, 0], align, n_clash, align_label, clash_label, "align vs Hf clashes"),
            (axes[1, 1], r_m_min, n_clash, rm_label, clash_label, f"min Hf-{M} distance vs Hf clashes")]:
        ax.scatter(x, y, s=8, c="#4c78a8", lw=0, alpha=0.5, label=f"{n} shifts")
        for name, i in cands:
            ax.scatter(x[i], y[i], marker=markers[name], s=90, c="white", edgecolors="k", zorder=5)
            ax.annotate(name, (x[i], y[i]), xytext=(5, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(out_root / "registry_scan_maps.png", dpi=200)
    plt.close(fig)

    # --- figure 2: top view of the interface for each candidate ---
    def wrap_xy(xy):
        f = xy @ np.linalg.inv(cell2)
        return (f - np.floor(f)) @ cell2

    corners = np.array([[0, 0], cell2[0], cell2[0] + cell2[1], cell2[1], [0, 0]])
    fig, axes = plt.subplots(1, len(cands), figsize=(4.6 * len(cands), 5.3))
    for ax, (name, i) in zip(axes, cands):
        t = shifts[i]
        ax.plot(corners[:, 0], corners[:, 1], color="k", lw=0.7, alpha=0.5)
        ax.scatter(*wrap_xy(pos[plane_M, :2]).T, s=70, c=COLORS[M], lw=0, label=f"{M} (plane)")
        ax.scatter(*wrap_xy(pos[plane_O, :2]).T, s=30, c=COLORS["O"], lw=0, label="O (plane)")
        ax.scatter(*wrap_xy(pos[o_bot, :2] + t).T, s=26, facecolors="none", edgecolors=COLORS["O"],
                   lw=0.9, label="O (film bottom)")
        ax.scatter(*wrap_xy(pos[hf1, :2] + t).T, s=42, facecolors="none", edgecolors=COLORS["Hf"],
                   lw=1.4, label="Hf (first layer)")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}: shift ({t[0]:+.2f}, {t[1]:+.2f}) A\n"
                     f"align {align[i]:.2f} (on top {n_on_top[i]}/{plane_O.sum()}), "
                     f"min Hf-{M} {r_m_min[i]:.2f} A (clash {n_clash[i]})\n"
                     f"O-O stacked {oo_stack[i]}, d_min {d_min[i]:.2f} A", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Candidate registries on {term}, top view (filled = STO plane, hollow = film)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_root / "registry_scan_candidates.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote {len(cands)} structures, maps and summary to {out_root}")


if __name__ == "__main__":
    main()
