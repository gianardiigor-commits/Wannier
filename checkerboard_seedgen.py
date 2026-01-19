#!/usr/bin/env python3
"""
checkerboard_seedgen.py

Generate Wannier90 seed files (seed.win, seed.mmn, seed.amn, seed.eig)
for the 2D checkerboard optical lattice potential:

V(x,y) = -V0 [cos^2(kL x) + cos^2(kL y) + 2 cos(theta) cos(kL x) cos(kL y)]

We:
  1) write seed.win (k-mesh etc.)
  2) run wannier90.x -pp seed  -> seed.nnkp
  3) parse seed.nnkp to get neighbor list (nnkpts)
  4) solve Bloch eigenproblem in plane-wave basis at each k
  5) compute mmn overlaps and amn projections (2 Gaussian projectors)
  6) write seed.mmn, seed.amn, seed.eig

Dependencies: numpy, scipy, matplotlib (optional but recommended).
Requires: wannier90.x in PATH.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import subprocess
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import eigh
except ImportError as e:
    raise SystemExit("You need scipy: pip install scipy") from e


# -----------------------------
# Parameters
# -----------------------------

@dataclasses.dataclass
class Params:
    seed: str = "chk"

    # Lattice / units:
    # We use the rotated primitive cell with
    # a1=(pi/kL, pi/kL), a2=(pi/kL, -pi/kL).
    kL: float = 1.0
    Lz: float = 30.0  # "vacuum" direction (only for .win/.nnkp 3rd vector)

    # Potential
    V0: float = 6.0                      # in recoil units if you interpret E_R = ħ² kL² / (2m)
    theta: float = 0.38 * math.pi        # your value

    # Wannier90 targets
    num_wann: int = 2
    num_bands: int = 3                   # compute a small isolated subspace by default
    mp_grid: Tuple[int, int, int] = (8, 8, 1)

    # Plane-wave basis cutoff (integer G components in reciprocal lattice units)
    # Use a square cutoff to make G-shifts from nnkp safer.
    Gmax: int = 8

    # Real-space grid for building projectors (and optional checks)
    ngr: Tuple[int, int] = (64, 64)

    # Gaussian projector width (in real-space length units)
    # A reasonable starting value is ~0.2 * lattice spacing.
    sigma: float = 0.35

    # If None: auto-detect two deepest minima in the cell for projector centers.
    # Otherwise specify two centers in Cartesian coordinates (x,y) within [0,Lx)×[0,Ly).
    proj_centers_xy: List[Tuple[float, float]] | None = None

    # Wannier90 plotting
    wannier_plot: bool = False
    wannier_plot_format: str = "xcrysden"
    wannier_plot_supercell: Tuple[int, int, int] | None = None
    wannier_plot_list: str | None = None
    wvfn_formatted: bool = False


# -----------------------------
# Potential and Fourier components
# -----------------------------

def V_xy(x: NDArray[np.float64], y: NDArray[np.float64], p: Params) -> NDArray[np.float64]:
    """Checkerboard potential on arrays x,y (broadcastable)."""
    k = p.kL
    c = math.cos(p.theta)
    return -p.V0 * (np.cos(k * x) ** 2 + np.cos(k * y) ** 2 + 2.0 * c * np.cos(k * x) * np.cos(k * y))


def fourier_coeffs(p: Params) -> Dict[Tuple[int, int, int], float]:
    """
    Fourier coefficients V_G for the potential expressed on the reciprocal lattice of
    the rotated primitive cell with a1=(pi/kL, pi/kL), a2=(pi/kL, -pi/kL).
    The reciprocal basis vectors are b1=(kL,kL), b2=(kL,-kL).

    V(x,y) = -V0 [cos^2 x + cos^2 y + 2c cos x cos y]
           = -V0 [1 + 0.5 cos 2x + 0.5 cos 2y + c cos(x+y) + c cos(x-y)]
    """
    c = math.cos(p.theta)
    V0 = p.V0
    d: Dict[Tuple[int, int, int], float] = {}

    # Constant
    d[(0, 0, 0)] = -V0

    # cos 2x term: wavevector (2,0) = b1 + b2 -> (1,1) in (b1,b2) indices
    d[(1, 1, 0)] = d.get((1, 1, 0), 0.0) + (-V0 / 4.0)
    d[(-1, -1, 0)] = d.get((-1, -1, 0), 0.0) + (-V0 / 4.0)

    # cos 2y term: wavevector (0,2) = b1 - b2 -> (1,-1)
    d[(1, -1, 0)] = d.get((1, -1, 0), 0.0) + (-V0 / 4.0)
    d[(-1, 1, 0)] = d.get((-1, 1, 0), 0.0) + (-V0 / 4.0)

    # c cos(x+y): wavevector (1,1) = b1
    d[(1, 0, 0)] = d.get((1, 0, 0), 0.0) + (-V0 * c / 2.0)
    d[(-1, 0, 0)] = d.get((-1, 0, 0), 0.0) + (-V0 * c / 2.0)

    # c cos(x-y): wavevector (1,-1) = b2
    d[(0, 1, 0)] = d.get((0, 1, 0), 0.0) + (-V0 * c / 2.0)
    d[(0, -1, 0)] = d.get((0, -1, 0), 0.0) + (-V0 * c / 2.0)

    return d


# -----------------------------
# Lattice, k-mesh, .win, -pp
# -----------------------------

def lattice_vectors(p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotated primitive real lattice:
      a1=(pi/kL,  pi/kL, 0), a2=(pi/kL, -pi/kL, 0), a3=(0,0,Lz)
    Reciprocal lattice:
      b1=(kL,kL,0), b2=(kL,-kL,0), b3=(0,0,2*pi/Lz)
    """
    L = math.pi / p.kL
    A = np.array([[L, L, 0.0],
                  [L, -L, 0.0],
                  [0.0, 0.0, p.Lz]], dtype=float)
    # b = 2π (A^-T)
    B = 2.0 * math.pi * np.linalg.inv(A).T
    return A, B


def monkhorst_pack_kpoints(mp: Tuple[int, int, int]) -> np.ndarray:
    """
    Gamma-centered grid used by many Wannier90 workflows:
      k(i,j,k) = (i-1)/N1 b1 + (j-1)/N2 b2 + (k-1)/N3 b3
    i.e. fractional coords in [0,1).
    (This is also shown schematically in Wannier90 docs/tutorial literature.) :contentReference[oaicite:5]{index=5}
    """
    N1, N2, N3 = mp
    kpts = []
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                kpts.append([i / N1, j / N2, k / N3])
    return np.array(kpts, dtype=float)


def write_win(p: Params) -> None:
    A, _B = lattice_vectors(p)
    kpts_frac = monkhorst_pack_kpoints(p.mp_grid)

    with open(p.seed + ".win", "w", encoding="utf-8") as f:
        f.write(f"# {p.seed}.win generated by checkerboard_seedgen.py\n")
        f.write("num_wann = %d\n" % p.num_wann)
        f.write("num_bands = %d\n" % p.num_bands)
        f.write("mp_grid = %d %d %d\n" % (p.mp_grid[0], p.mp_grid[1], p.mp_grid[2]))
        f.write("guiding_centres = true\n")
        f.write("write_hr = true\n")
        f.write("write_xyz = true\n")
        f.write("begin unit_cell_cart\n")
        f.write("  ang\n")
        for row in A:
            f.write("  %16.10f %16.10f %16.10f\n" % (row[0], row[1], row[2]))
        f.write("end unit_cell_cart\n\n")

        # We don't need to specify projections here because we provide .amn,
        # but keeping projections=random ensures num_proj is defined.
        f.write("begin projections\n")
        f.write("  random\n")
        f.write("end projections\n\n")

        if p.wannier_plot:
            f.write("wannier_plot = .true.\n")
            if p.wannier_plot_list is None:
                f.write(f"wannier_plot_list = 1-{p.num_wann}\n")
            else:
                f.write(f"wannier_plot_list = {p.wannier_plot_list}\n")
            f.write(f"wannier_plot_format = {p.wannier_plot_format}\n")
            if p.wannier_plot_supercell is not None:
                f.write("wannier_plot_supercell = %d %d %d\n" % p.wannier_plot_supercell)
            f.write("\n")
        if p.wvfn_formatted:
            f.write("wvfn_formatted = .true.\n\n")

        f.write("begin kpoints\n")
        for k in kpts_frac:
            f.write("  %16.10f %16.10f %16.10f\n" % (k[0], k[1], k[2]))
        f.write("end kpoints\n")


def run_w90_pp(seed: str) -> None:
    cmd = ["wannier90.x", "-pp", seed]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise SystemExit("wannier90.x not found in PATH. Install Wannier90 or adjust PATH.") from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"wannier90.x -pp failed with code {e.returncode}") from e


def parse_nnkp(seed: str) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Parse seed.nnkp to get:
      kpts_frac: (num_kpts,3)
      nntot: int
      nnlist: (num_kpts,nntot)  [1-based indices in file, we'll convert to 0-based]
      nncell: (num_kpts,nntot,3) integer G-shifts
    nnkp format described in wannier90 kmesh_write source docs. :contentReference[oaicite:6]{index=6}
    """
    path = seed + ".nnkp"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Did wannier90.x -pp run successfully?")

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    def find_block(name: str) -> int:
        for i, ln in enumerate(lines):
            if ln.lower() == f"begin {name}".lower():
                return i
        raise ValueError(f"Block begin {name} not found in nnkp")

    # kpoints
    i0 = find_block("kpoints")
    nk = int(lines[i0 + 1].split()[0])
    kpts = []
    for i in range(nk):
        vals = lines[i0 + 2 + i].split()
        kpts.append([float(vals[0]), float(vals[1]), float(vals[2])])
    kpts_frac = np.array(kpts, dtype=float)

    # nnkpts
    j0 = find_block("nnkpts")
    nntot = int(lines[j0 + 1].split()[0])
    nnlist = np.zeros((nk, nntot), dtype=int)
    nncell = np.zeros((nk, nntot, 3), dtype=int)

    # There are nk*nntot lines of: nkp, nnlist(nkp,nn), (nncellx,nncelly,nncellz)
    idx = j0 + 2
    for _ in range(nk * nntot):
        parts = lines[idx].split()
        idx += 1
        ik = int(parts[0]) - 1
        ik2 = int(parts[1]) - 1
        gx, gy, gz = int(parts[2]), int(parts[3]), int(parts[4])
        # figure out which neighbor slot this is: first free column
        col = np.where(nnlist[ik] == 0)[0]
        if len(col) == 0:
            # if ik2 can be 0, this fails; so use a robust counter:
            pass
        # use a running counter per ik:
        # We'll re-parse with a deterministic method:
    # Robust parse: redo neighbor reading with per-ik counters
    nnlist.fill(-1)
    nncell.fill(0)
    counters = np.zeros(nk, dtype=int)
    idx = j0 + 2
    for _ in range(nk * nntot):
        parts = lines[idx].split()
        idx += 1
        ik = int(parts[0]) - 1
        slot = counters[ik]
        counters[ik] += 1
        nnlist[ik, slot] = int(parts[1]) - 1
        nncell[ik, slot, :] = [int(parts[2]), int(parts[3]), int(parts[4])]

    return kpts_frac, nntot, nnlist, nncell


def filter_nnkp_z(
    nnlist: np.ndarray,
    nncell: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Drop neighbors with non-zero gz and compact the neighbor list."""
    nk, nntot, _ = nncell.shape
    keep = nncell[:, :, 2] == 0
    new_nntot = int(np.max(np.sum(keep, axis=1)))
    nnlist_new = np.full((nk, new_nntot), -1, dtype=int)
    nncell_new = np.zeros((nk, new_nntot, 3), dtype=int)
    for ik in range(nk):
        cols = np.where(keep[ik])[0]
        for j, col in enumerate(cols):
            if j >= new_nntot:
                break
            nnlist_new[ik, j] = nnlist[ik, col]
            nncell_new[ik, j] = nncell[ik, col]
    return new_nntot, nnlist_new, nncell_new


def rewrite_nnkp_nnlist(
    seed: str,
    nntot: int,
    nnlist: np.ndarray,
    nncell: np.ndarray,
) -> None:
    """Rewrite only the nnkpts block in seed.nnkp to match provided neighbors."""
    path = seed + ".nnkp"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    begin = None
    end = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "begin nnkpts":
            begin = i
        if ln.strip().lower() == "end nnkpts":
            end = i
            break
    if begin is None or end is None:
        raise ValueError("nnkpts block not found in nnkp file.")

    nk = nnlist.shape[0]
    new_block = []
    new_block.append("begin nnkpts\n")
    new_block.append(f"{nntot:4d}\n")
    for ik in range(nk):
        for inn in range(nntot):
            ik2 = nnlist[ik, inn]
            gx, gy, gz = nncell[ik, inn]
            new_block.append(f"{ik + 1:6d}{ik2 + 1:6d}{gx:7d}{gy:4d}{gz:4d}\n")
    new_block.append("end nnkpts\n")

    lines = lines[:begin] + new_block + lines[end + 1 :]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# -----------------------------
# Plane-wave Bloch solver
# -----------------------------

def pw_basis(Gmax: int) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int]]:
    """
    Square cutoff basis: G = (gx,gy,0) with gx,gy in [-Gmax..Gmax].
    Returns:
      Gvecs_int: (M,3) integer components in reciprocal-lattice units
      index: dict mapping (gx,gy,0) -> basis index
    """
    gs = []
    for gx in range(-Gmax, Gmax + 1):
        for gy in range(-Gmax, Gmax + 1):
            gs.append((gx, gy, 0))
    G = np.array(gs, dtype=int)
    idx = {(int(g[0]), int(g[1]), int(g[2])): i for i, g in enumerate(G)}
    return G, idx


def build_Hk(k_cart: np.ndarray, G_int: np.ndarray, Vdiff: Dict[Tuple[int, int, int], float], p: Params) -> np.ndarray:
    """
    Build Hamiltonian matrix in plane-wave basis |k+G>.
    - k_cart is in Cartesian reciprocal units matching b1,b2,b3 from lattice_vectors.
      For the rotated cell, b1=(kL,kL,0), b2=(kL,-kL,0).
    - G_int are integers multiplying reciprocal basis vectors.
    Kinetic energy in recoil units: |k+G|^2 / kL^2 (since E_R = ħ² kL² /2m).
    """
    # Convert integer G into Cartesian vector:
    _A, B = lattice_vectors(p)
    G_cart = G_int @ B  # (M,3)

    # Kinetic diag
    kk = k_cart[None, :] + G_cart
    # in E_R units:
    T = np.sum(kk[:, :2] ** 2, axis=1) / (p.kL ** 2)  # ignore z
    M = G_int.shape[0]
    H = np.zeros((M, M), dtype=complex)
    np.fill_diagonal(H, T)

    # Potential coupling: H_{GG'} += V_{G-G'}
    # We loop over all pairs in a moderately sized basis; for bigger bases you would sparsify.
    for i in range(M):
        gi = tuple(int(x) for x in G_int[i])
        for j in range(M):
            gj = tuple(int(x) for x in G_int[j])
            diff = (gi[0] - gj[0], gi[1] - gj[1], 0)
            v = Vdiff.get(diff, 0.0)
            if v != 0.0:
                H[i, j] += v

    # Hermiticity check (numerical)
    if not np.allclose(H, H.conj().T, atol=1e-12, rtol=0):
        raise RuntimeError("Hamiltonian is not Hermitian (check Fourier coefficients / basis).")
    return H


def solve_all_kpoints(kpts_frac: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int,int,int],int]]:
    """
    Returns:
      eigvals: (nk,num_bands)
      coeffs:  (nk,num_bands,M) complex plane-wave coeffs for periodic part u_{nk}(r)
      G_int:   (M,3)
      G_index: dict
    """
    Vdiff = fourier_coeffs(p)
    G_int, G_index = pw_basis(p.Gmax)
    _A, B = lattice_vectors(p)

    nk = kpts_frac.shape[0]
    M = G_int.shape[0]
    eigvals = np.zeros((nk, p.num_bands), dtype=float)
    coeffs = np.zeros((nk, p.num_bands, M), dtype=complex)

    for ik in range(nk):
        k_cart = kpts_frac[ik] @ B  # (3,)
        Hk = build_Hk(k_cart, G_int, Vdiff, p)
        w, v = eigh(Hk)  # ascending
        eigvals[ik, :] = w[:p.num_bands]
        # v columns are eigenvectors in basis; normalize is handled by eigh
        coeffs[ik, :, :] = v[:, :p.num_bands].T  # (bands,M)

        # Orthonormality check on eigenvectors
        ov = coeffs[ik] @ coeffs[ik].conj().T
        if not np.allclose(ov, np.eye(p.num_bands), atol=1e-10):
            raise RuntimeError(f"Eigenvectors not orthonormal at ik={ik}")

    return eigvals, coeffs, G_int, G_index


# -----------------------------
# Projectors and overlaps
# -----------------------------

def periodic_displacement_vec(
    r: np.ndarray,
    r0: np.ndarray,
    A2: np.ndarray,
    A2_inv: np.ndarray,
) -> np.ndarray:
    """Minimal-image displacement in a 2D Bravais cell defined by A2 (rows a1,a2)."""
    s = r @ A2_inv
    s0 = r0 @ A2_inv
    ds = s - s0
    ds -= np.round(ds)
    return ds @ A2


def _origin_shift_from_centers(p: Params, centers: List[Tuple[float, float]]) -> np.ndarray:
    """Choose origin shift so the midpoint of the first two centers sits at the cell center."""
    if len(centers) < 2:
        return np.zeros(2)
    A, _B = lattice_vectors(p)
    a1 = A[0, :2]
    a2 = A[1, :2]
    cell_center = 0.5 * (a1 + a2)
    mid = 0.5 * (np.array(centers[0]) + np.array(centers[1]))
    return mid - cell_center


def _find_projector_centers(
    p: Params,
    origin_shift: np.ndarray,
) -> List[Tuple[float, float]]:
    """Find minima within a cell whose origin is shifted by origin_shift."""
    A, _B = lattice_vectors(p)
    A2 = A[:2, :2]
    A2_inv = np.linalg.inv(A2)
    a1 = A2[0]
    a2 = A2[1]
    nx, ny = 301, 301
    us = np.linspace(0.0, 1.0, nx, endpoint=False)
    vs = np.linspace(0.0, 1.0, ny, endpoint=False)
    U, V = np.meshgrid(us, vs, indexing="ij")
    X = origin_shift[0] + U * a1[0] + V * a2[0]
    Y = origin_shift[1] + U * a1[1] + V * a2[1]
    Vpot = V_xy(X, Y, p)

    # Identify local minima on the periodic grid.
    mask = np.ones_like(Vpot, dtype=bool)
    shifts = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]
    for dx, dy in shifts:
        nbr = np.roll(Vpot, shift=(dx, dy), axis=(0, 1))
        mask &= Vpot <= nbr + 1e-12

    if not np.any(mask):
        mask = np.ones_like(Vpot, dtype=bool)

    flat = np.column_stack([X[mask], Y[mask], Vpot[mask]])
    flat = flat[np.argsort(flat[:, 2])]  # ascending (most negative first)

    centers: List[Tuple[float, float]] = []
    min_sep = 0.25 * min(np.linalg.norm(a1), np.linalg.norm(a2))  # crude
    for x, y, _v in flat:
        ok = True
        for (cx, cy) in centers:
            dxy = periodic_displacement_vec(
                np.array([[x, y]]),
                np.array([cx, cy]),
                A2,
                A2_inv,
            )[0]
            if math.hypot(dxy[0], dxy[1]) < min_sep:
                ok = False
                break
        if ok:
            centers.append((float(x), float(y)))
        if len(centers) == p.num_wann:
            break

    if len(centers) < p.num_wann:
        raise RuntimeError("Failed to find enough distinct minima for projector centers.")
    return centers


def auto_projector_centers(p: Params) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Find two deepest minima within the rotated primitive cell.
    Very simple grid search + uniqueness filter.
    """
    centers0 = _find_projector_centers(p, origin_shift=np.zeros(2))
    A, _B = lattice_vectors(p)
    A2 = A[:2, :2]
    A2_inv = np.linalg.inv(A2)
    a1 = A2[0]
    a2 = A2[1]
    if p.num_wann == 2:
        expected = [
            np.zeros(2),
            0.5 * (a1 + a2),
        ]
        tol = 2.0 * np.linalg.norm(a1) / 301.0
        for exp in expected:
            dmin = min(
                np.linalg.norm(
                    periodic_displacement_vec(
                        np.array([c]),
                        exp,
                        A2,
                        A2_inv,
                    )[0]
                )
                for c in centers0
            )
            if dmin > tol:
                raise RuntimeError(
                    "Autodetected minima deviate from expected sites; "
                    f"min distance {dmin:.3e} > tol {tol:.3e}."
                )
    origin_shift = _origin_shift_from_centers(p, centers0)
    centers = _find_projector_centers(p, origin_shift=origin_shift)
    return centers, origin_shift


def build_projectors_G(
    p: Params,
    G_int: np.ndarray,
    G_index: Dict[Tuple[int,int,int],int],
) -> np.ndarray:
    """
    Build Gaussian projectors g_n(r) on a real-space grid and compute their plane-wave
    coefficients g_n(G) in the same orthonormal basis exp(iG·r)/sqrt(Ω).

    Returns:
      gG: (num_proj, M) complex
    """
    A, B = lattice_vectors(p)
    A2 = A[:2, :2]
    A2_inv = np.linalg.inv(A2)
    a1 = A2[0]
    a2 = A2[1]
    nx, ny = p.ngr
    us = np.linspace(0.0, 1.0, nx, endpoint=False)
    vs = np.linspace(0.0, 1.0, ny, endpoint=False)
    U, V = np.meshgrid(us, vs, indexing="ij")
    Omega = abs(np.linalg.det(A2))
    dA = Omega / (nx * ny)

    if p.proj_centers_xy is None:
        centers, origin_shift = auto_projector_centers(p)
    else:
        centers = p.proj_centers_xy
        origin_shift = _origin_shift_from_centers(p, centers)

    X = origin_shift[0] + U * a1[0] + V * a2[0]
    Y = origin_shift[1] + U * a1[1] + V * a2[1]

    # Basis functions phi_G(r) = exp(i G·r)/sqrt(Omega)
    G_cart = G_int @ B
    gx = G_cart[:, 0].astype(float)
    gy = G_cart[:, 1].astype(float)

    # Flatten grid
    xr = X.reshape(-1)
    yr = Y.reshape(-1)

    # exp(iG·r) matrix: (Ngrid, M)
    phase = np.exp(1j * (xr[:, None] * gx[None, :] + yr[:, None] * gy[None, :])) / math.sqrt(Omega)

    gG = np.zeros((p.num_wann, G_int.shape[0]), dtype=complex)

    for n, (x0, y0) in enumerate(centers):
        r = np.stack([X, Y], axis=-1)
        dr = periodic_displacement_vec(r, np.array([x0, y0]), A2, A2_inv)
        g = np.exp(-(dr[..., 0] * dr[..., 0] + dr[..., 1] * dr[..., 1]) /
                   (2.0 * p.sigma * p.sigma))

        # normalize projector in cell: ∫|g|^2 dA = 1
        norm = math.sqrt(np.sum(np.abs(g) ** 2) * dA)
        g = g / norm

        gr = g.reshape(-1).astype(complex)

        # gG(G) = ∫ phi_G*(r) g(r) dA
        gG[n, :] = (phase.conj().T @ gr) * dA

    return gG


def compute_amn(coeffs: np.ndarray, gG: np.ndarray) -> np.ndarray:
    """
    coeffs: (nk, num_bands, M) for periodic u_{nk}
    gG:     (num_proj, M)
    Returns:
      A: (nk, num_bands, num_proj) where A[ik,m,n] = <u_{m,k} | g_n>
    """
    # <u|g> = sum_G conj(c_G) * gG_G
    return np.einsum("kbm,nm->kbn", coeffs.conj(), gG)


def compute_mmn(
    coeffs: np.ndarray,
    nnlist: np.ndarray,
    nncell: np.ndarray,
    G_index: Dict[Tuple[int,int,int],int],
    z_neighbors_identity: bool = False,
) -> np.ndarray:
    """
    Compute overlaps M_{mn}(k, b) with the nncell G-vector shift:
      M(k,b) = <u_{m,k} | e^{-i G·r} u_{n,k2}>

    In plane-wave coefficients, e^{-iG·r} shifts neighbor coefficients:
      c_n(k+b, G') = c_n(k2, G'+G)
    so
      M_{mn} = sum_{G'} conj(c_m(k,G')) * c_n(k2, G'+G)

    Returns:
      Mmn: (nk, nntot, num_bands, num_bands)
    """
    nk, nb, M = coeffs.shape
    nntot = nnlist.shape[1]
    out = np.zeros((nk, nntot, nb, nb), dtype=complex)

    # Precompute shift maps for each unique nncell vector
    unique_shifts = {tuple(int(x) for x in nncell[ik, inn]) for ik in range(nk) for inn in range(nntot)}
    shift_map: Dict[Tuple[int,int,int], np.ndarray] = {}

    # Build list of basis G tuples in same order as coeffs axis
    # (We can recover from G_index keys, but easier: invert dict.)
    inv = [None] * len(G_index)
    for g, i in G_index.items():
        inv[i] = g
    basis_G = inv  # list of tuples length M

    for sh in unique_shifts:
        # map i -> j where basis_G[j] = basis_G[i] + sh
        mp = np.full(M, -1, dtype=int)
        sx, sy, sz = sh
        for i, g in enumerate(basis_G):
            gp = (g[0] + sx, g[1] + sy, g[2] + sz)
            j = G_index.get(gp, -1)
            mp[i] = j
        shift_map[sh] = mp

    for ik in range(nk):
        Ck = coeffs[ik]  # (nb,M)
        for inn in range(nntot):
            ik2 = int(nnlist[ik, inn])
            sh = tuple(int(x) for x in nncell[ik, inn])
            if z_neighbors_identity and sh[2] != 0:
                out[ik, inn] = np.eye(nb, dtype=complex)
                continue
            mp = shift_map[sh]
            C2 = coeffs[ik2]  # (nb,M)

            # Build shifted neighbor coeffs: (nb,M) where missing components are zero
            C2s = np.zeros_like(C2)
            good = mp >= 0
            C2s[:, good] = C2[:, mp[good]]

            # Overlap matrix: (nb,nb) = Ck * C2s^†? careful with indices:
            # M[m,n] = sum_G conj(Ck[m,G]) * C2s[n,G]
            # => M = Ck.conj() @ C2s.T
            out[ik, inn] = Ck.conj() @ C2s.T

    return out


# -----------------------------
# Optional UNK output for Wannier90 plotting
# -----------------------------

def write_unk_files(
    p: Params,
    kpts_frac: np.ndarray,
    coeffs: np.ndarray,
    G_int: np.ndarray,
) -> None:
    """
    Write formatted UNK files containing the periodic part u_{n,k}(r)
    on a real-space grid over the unit cell.
    """
    A, B = lattice_vectors(p)
    A2 = A[:2, :2]
    a1 = A2[0]
    a2 = A2[1]
    nx, ny = p.ngr
    nz = 1
    us = np.linspace(0.0, 1.0, nx, endpoint=False)
    vs = np.linspace(0.0, 1.0, ny, endpoint=False)
    U, V = np.meshgrid(us, vs, indexing="ij")
    X = U * a1[0] + V * a2[0]
    Y = U * a1[1] + V * a2[1]
    Omega = abs(np.linalg.det(A2))

    G_cart = G_int @ B
    gx = G_cart[:, 0].astype(float)
    gy = G_cart[:, 1].astype(float)
    phase = np.exp(1j * (X[:, :, None] * gx[None, None, :] +
                         Y[:, :, None] * gy[None, None, :])) / math.sqrt(Omega)

    nk, nb, _M = coeffs.shape
    for ik in range(nk):
        fname = f"UNK{ik + 1:05d}.1"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"{nx:4d} {ny:4d} {nz:4d} {ik + 1:4d} {nb:4d}\n")
            for b in range(nb):
                u = np.tensordot(phase, coeffs[ik, b, :], axes=(2, 0))
                flat = u.reshape(-1, order="F")
                for val in flat:
                    f.write(f" {val.real: .10f} {val.imag: .10f}\n")


# -----------------------------
# Write Wannier90 seed files
# -----------------------------

def write_eig(seed: str, eigvals: np.ndarray) -> None:
    nk, nb = eigvals.shape
    with open(seed + ".eig", "w", encoding="utf-8") as f:
        for ik in range(nk):
            for m in range(nb):
                f.write("%5d %5d % .14e\n" % (m + 1, ik + 1, eigvals[ik, m]))


def write_amn(seed: str, A: np.ndarray) -> None:
    nk, nb, npj = A.shape
    with open(seed + ".amn", "w", encoding="utf-8") as f:
        f.write("# projections A_mn(k) generated by checkerboard_seedgen.py\n")
        f.write("%d %d %d\n" % (nb, nk, npj))
        # conventional ordering: ik outer, m, n inner :contentReference[oaicite:7]{index=7}
        for ik in range(nk):
            for m in range(nb):
                for n in range(npj):
                    z = A[ik, m, n]
                    f.write("%5d %5d %5d % .14e % .14e\n" % (m + 1, n + 1, ik + 1, z.real, z.imag))


def write_mmn(seed: str, Mmn: np.ndarray, nnlist: np.ndarray, nncell: np.ndarray) -> None:
    nk, nntot, nb, nb2 = Mmn.shape
    assert nb == nb2
    with open(seed + ".mmn", "w", encoding="utf-8") as f:
        f.write("# overlaps M_mn(k,b) generated by checkerboard_seedgen.py\n")
        f.write("%d %d %d\n" % (nb, nk, nntot))
        # ordering consistent with Wannier90 readers :contentReference[oaicite:8]{index=8}
        for ik in range(nk):
            for inn in range(nntot):
                ik2 = int(nnlist[ik, inn]) + 1
                gx, gy, gz = (int(nncell[ik, inn, 0]), int(nncell[ik, inn, 1]), int(nncell[ik, inn, 2]))
                f.write("%5d %5d %4d %4d %4d\n" % (ik + 1, ik2, gx, gy, gz))
                M = Mmn[ik, inn]  # (nb,nb) with indices [m,n]
                for n in range(nb):
                    for m in range(nb):
                        z = M[m, n]
                        f.write("% .14e % .14e\n" % (z.real, z.imag))


# -----------------------------
# Diagnostics / checks
# -----------------------------

def check_mmn_unitarity(Mmn: np.ndarray, tol: float = 5e-3) -> None:
    """
    If the chosen band subspace is isolated and basis cutoff is sufficient,
    M(k,b) should be close to unitary.
    """
    nk, nntot, nb, _ = Mmn.shape
    worst = 0.0
    for ik in range(nk):
        for inn in range(nntot):
            M = Mmn[ik, inn]
            err = np.linalg.norm(M.conj().T @ M - np.eye(nb))
            worst = max(worst, float(err))
    print(f"[check] worst ||M^†M - I||_F = {worst:.3e}")
    if worst > tol:
        print("[warn] Overlaps not very unitary. Increase Gmax and/or include more bands, "
              "or check that the chosen bands form an isolated group.")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="chk")
    ap.add_argument("--V0", type=float, default=6.0)
    ap.add_argument("--theta_pi", type=float, default=0.38, help="theta / pi, so 0.38 means 0.38*pi")
    ap.add_argument("--Nk", type=int, nargs=2, default=[8, 8], help="k-mesh Nkx Nky")
    ap.add_argument("--Gmax", type=int, default=8)
    ap.add_argument("--num_bands", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--two_band_case", action="store_true",
                    help="set theta=0.5*pi and num_bands=2 for symmetric two-well case")
    ap.add_argument("--write_unk", action="store_true",
                    help="write formatted UNK files for wannier_plot")
    ap.add_argument("--wannier_plot", action="store_true",
                    help="write wannier_plot settings into the .win file")
    ap.add_argument("--wannier_plot_format", default="xcrysden")
    ap.add_argument("--wannier_plot_supercell", type=int, nargs=3)
    ap.add_argument("--wannier_plot_list", default=None,
                    help="list of WF indices to plot, e.g. '1-2'")
    ap.add_argument("--drop_z_neighbors", action="store_true",
                    help="drop nnkp neighbors with gz != 0 (2D models)")
    ap.add_argument("--z_neighbors_identity", action="store_true",
                    help="set overlaps for gz!=0 neighbors to identity")
    ap.add_argument("--step", type=int, default=4, choices=[1,2,3,4],
                    help="1: write win+plot minima; 2: run -pp+parse nnkp; 3: solve bands; 4: write mmn/amn/eig")
    args = ap.parse_args()
    if args.wannier_plot and not args.write_unk:
        args.write_unk = True

    if args.two_band_case:
        args.theta_pi = 0.5
        args.num_bands = 2

    p = Params(
        seed=args.seed,
        V0=args.V0,
        theta=args.theta_pi * math.pi,
        mp_grid=(args.Nk[0], args.Nk[1], 1),
        Gmax=args.Gmax,
        num_bands=args.num_bands,
        sigma=args.sigma,
        wannier_plot=args.wannier_plot,
        wannier_plot_format=args.wannier_plot_format,
        wannier_plot_supercell=(tuple(args.wannier_plot_supercell)
                                if args.wannier_plot_supercell is not None else None),
        wannier_plot_list=args.wannier_plot_list,
        wvfn_formatted=args.write_unk,
    )

    A, _B = lattice_vectors(p)
    print(f"[info] Using rotated cell a1=({A[0, 0]:.6f},{A[0, 1]:.6f}) "
          f"a2=({A[1, 0]:.6f},{A[1, 1]:.6f}) (kL={p.kL}); "
          f"theta={p.theta/math.pi:.4f}*pi; V0={p.V0}")
    print(f"[info] mp_grid={p.mp_grid}, num_bands={p.num_bands}, num_wann={p.num_wann}, Gmax={p.Gmax}")

    # STEP 1: write .win and (optionally) report minima locations
    nk = p.mp_grid[0] * p.mp_grid[1] * p.mp_grid[2]
    write_win(p)
    print(f"[step1] wrote {p.seed}.win with mp_grid {p.mp_grid} ({nk} k-points)")

    if p.proj_centers_xy is None:
        centers, origin_shift = auto_projector_centers(p)
    else:
        centers = p.proj_centers_xy
        origin_shift = _origin_shift_from_centers(p, centers)
    print(f"[step1] using cell origin shift: x={origin_shift[0]:.6f}, y={origin_shift[1]:.6f}")
    print("[step1] suggested projector centers (x,y) in unit cell:")
    for i, (x, y) in enumerate(centers, start=1):
        print(f"        proj {i}: x={x:.6f}, y={y:.6f}")

    if args.step == 1:
        return

    # STEP 2: run wannier90 -pp and parse nnkp
    print(f"[step2] running: wannier90.x -pp {p.seed}")
    run_w90_pp(p.seed)
    kpts2, nntot, nnlist, nncell = parse_nnkp(p.seed)
    if args.drop_z_neighbors:
        nntot, nnlist, nncell = filter_nnkp_z(nnlist, nncell)
        rewrite_nnkp_nnlist(p.seed, nntot, nnlist, nncell)
        print(f"[step2] filtered nnkp: nntot={nntot} (dropped gz != 0)")
    print(f"[step2] parsed {p.seed}.nnkp: nk={len(kpts2)}, nntot={nntot}")
    if args.step == 2:
        return

    # STEP 3: solve Bloch problem
    eigvals, coeffs, G_int, G_index = solve_all_kpoints(kpts2, p)
    print(f"[step3] solved bands: eigvals shape = {eigvals.shape}, coeffs shape = {coeffs.shape}")
    print(f"[step3] sample eigenvalues at kpoint 1: {eigvals[0, :min(6,p.num_bands)]}")
    if args.write_unk:
        write_unk_files(p, kpts2, coeffs, G_int)
        print("[step3] wrote UNK files for wannier_plot")
    if args.step == 3:
        return

    # STEP 4: projections + overlaps + write seed files
    gG = build_projectors_G(p, G_int, G_index)
    A = compute_amn(coeffs, gG)
    Mmn = compute_mmn(coeffs, nnlist, nncell, G_index, z_neighbors_identity=args.z_neighbors_identity)

    check_mmn_unitarity(Mmn)

    write_eig(p.seed, eigvals)
    write_amn(p.seed, A)
    write_mmn(p.seed, Mmn, nnlist, nncell)

    print(f"[step4] wrote: {p.seed}.eig, {p.seed}.amn, {p.seed}.mmn")
    print(f"[next] run: wannier90.x {p.seed}")


if __name__ == "__main__":
    main()
