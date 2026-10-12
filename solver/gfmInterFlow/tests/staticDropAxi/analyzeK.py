#!/usr/bin/env python3
"""
Per-cell curvature error of the axisymmetric static drop (exact K = 2/R = 8).

Reads ASCII fields of one time directory (C, alpha.phase1, K_, kappaTheta,
foundHeightField, interfaceCentre.phase1) and prints the error of
K, kappa_m = K - kappaTheta and kappaTheta (exact 4 each), binned by the
distance of the interface centre from the axis, split into height-function
(HF) and fallback (paraboloid) cells, plus the worst cells.

Usage: python3 analyzeK.py <timeDir> [R] [N] [wedgeAngleDeg]
       e.g. analyzeK.py 0.02 0.25 80 5
Axis: y (as in staticDropAxi), radial: x.

If the time directory contains the cell volumes (field V, from
postProcess -func writeCellVolumes), the reference curvature is 2/R_eq with
R_eq the radius of a sphere with the drop volume sum(alpha V) 2 pi/theta,
i.e. the equilibrium drop actually present on the mesh. Otherwise 2/R.
"""
import re
import sys
import numpy as np

import os

tdir = sys.argv[1] if len(sys.argv) > 1 else "0.0001"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
N = int(sys.argv[3]) if len(sys.argv) > 3 else 80
theta = np.radians(float(sys.argv[4]) if len(sys.argv) > 4 else 5.0)
d = 1.0/N


def read(name):
    s = open(f"{tdir}/{name}").read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\n?\(", s)
    if not m:
        raise SystemExit(f"{name}: not an ASCII nonuniform field")
    typ, n = m.group(1), int(m.group(2))
    body = s[m.end():]
    if typ == "scalar":
        return np.array(body.split(")", 1)[0].split(), float)[:n]
    vals = re.findall(r"\(([^()]*)\)", body)[:n]
    return np.array([list(map(float, v.split())) for v in vals])


C = read("C")
a = read("alpha.phase1")
K = read("K_")
kt = read("kappaTheta")
fh = read("foundHeightField")
ic = read("interfaceCentre.phase1")

if os.path.exists(f"{tdir}/V"):
    V = read("V")
    vol3D = (a*V).sum()*2*np.pi/theta
    Req = (3*vol3D/(4*np.pi))**(1.0/3.0)
    print(f"drop volume -> R_eq = {Req:.6f} (R = {R}), reference K = 2/R_eq = {2/Req:.5f}")
else:
    Req = R
    print(f"no V field in {tdir}: reference K = 2/R = {2/R:.5f}")
K_EXACT, KHALF = 2/Req, 1/Req

mask = (a > 1e-6) & (a < 1 - 1e-6)
idx = np.where(mask)[0]
rI = ic[idx, 0]                      # radial = x
km = K[idx] - kt[idx]
eK = K[idx] - K_EXACT
eM = km - KHALF
eT = kt[idx] - KHALF
hf = fh[idx] > 0.5

print(f"time {tdir}, N = {N}, R/Delta = {R/d:g}, interface cells {len(idx)}, "
      f"HF {hf.sum()}, fallback {(~hf).sum()}")
print(f"overall: mean(K-Kref) {eK.mean():+.4f}  rms {np.sqrt((eK**2).mean()):.4f}  "
      f"max|K-Kref| {np.abs(eK).max():.4f}")
print()
bins = [0, 1, 2, 3, 5, 10, 1e9]
print(f"{'r_I/Delta':>10} {'type':>8} {'n':>4} {'mean dK':>9} {'max|dK|':>9} "
      f"{'mean dkm':>10} {'mean dkt':>10}   (dK = K - Kref, dkm/dkt vs Kref/2)")
for lo, hi in zip(bins, bins[1:]):
    for name, sel in (("HF", hf), ("fallback", ~hf)):
        b = (rI/d >= lo) & (rI/d < hi) & sel
        if b.sum() == 0:
            continue
        lab = f"[{lo:g},{hi:g})" if hi < 1e8 else f">={lo:g}"
        print(f"{lab:>10} {name:>8} {b.sum():4d} {eK[b].mean():+9.4f} "
              f"{np.abs(eK[b]).max():9.4f} {eM[b].mean():+10.4f} {eT[b].mean():+10.4f}")

print()
print("worst 12 cells")
print(f"{'cell':>6} {'type':>8} {'r_I/D':>7} {'z_I':>7} {'alpha':>6} "
      f"{'K':>8} {'km':>8} {'kt':>8}")
for k in np.argsort(-np.abs(eK))[:12]:
    c = idx[k]
    print(f"{c:6d} {'HF' if hf[k] else 'fallback':>8} {rI[k]/d:7.2f} {ic[c,1]:7.4f} "
          f"{a[c]:6.3f} {K[c]:8.3f} {km[k]:8.3f} {kt[c]:8.3f}")
