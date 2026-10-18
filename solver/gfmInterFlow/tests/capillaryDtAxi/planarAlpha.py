#!/usr/bin/env python3
"""
Volume fractions of a planar (2D) drop r(theta) = R [1 + eps cos(2 theta)]
centred at (xc, yc) on a uniform 1-cell-thick mesh of spacing 1/N, by
sub-sampling each cell with ns x ns points (stability tests only; not exact).

Usage: planarAlpha.py <timeDir> N R xc yc eps [ns]
Needs <timeDir>/C in ASCII (postProcess -func writeCellCentres -time 0);
replaces internalField in <timeDir>/alpha.phase1.
"""
import re
import sys
import numpy as np

tdir = sys.argv[1]
N = int(sys.argv[2])
R, xc, yc, eps = (float(a) for a in sys.argv[3:7])
ns = int(sys.argv[7]) if len(sys.argv) > 7 else 24
d = 1.0/N

s = open(f"{tdir}/C").read()
m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n?\(", s)
if not m:
    raise SystemExit(f"{tdir}/C: need an ASCII nonuniform vector field")
n = int(m.group(1))
C = np.array([list(map(float, v.split()))
              for v in re.findall(r"\(([^()]*)\)", s[m.end():])[:n]])

off = (np.arange(ns) + 0.5)/ns - 0.5
ox, oy = np.meshgrid(off*d, off*d, indexing="ij")
alpha = np.zeros(n)
for c in range(n):
    dx = C[c, 0] - xc
    dy = C[c, 1] - yc
    rc = np.hypot(dx, dy)
    if rc < R*(1 - abs(eps)) - d:
        alpha[c] = 1.0
        continue
    if rc > R*(1 + abs(eps)) + d:
        continue
    X = dx + ox
    Y = dy + oy
    th = np.arctan2(Y, X)
    alpha[c] = np.mean(np.hypot(X, Y) < R*(1 + eps*np.cos(2*th)))

fname = f"{tdir}/alpha.phase1"
s = open(fname).read()
body = "internalField   nonuniform List<scalar>\n%d\n(\n%s\n)\n;" % (
    n, "\n".join(f"{v:.16g}" for v in alpha))
s, k = re.subn(r"internalField\s+(uniform[^;]*|nonuniform[\s\S]*?\)\s*);", body, s, count=1)
if k != 1:
    raise SystemExit("could not find internalField in " + fname)
open(fname, "w").write(s)
print(f"planar drop: cells {n}, area fraction sum*d^2 = {alpha.sum()*d*d:.6f}, "
      f"exact pi R^2 (1 + eps^2/2) = {np.pi*R*R*(1 + eps*eps/2):.6f}")
