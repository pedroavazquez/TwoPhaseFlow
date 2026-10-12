#!/usr/bin/env python3
"""
Exact initial volume fractions of a sphere on the staticDropAxi wedge mesh.

The wedge cells (blockMesh, axis = y, radial = x, half-angle theta/2) are
    x in [I dx, (I+1) dx],  y in [J dy, (J+1) dy],  |z| <= x tan(theta/2)
with dx = (0.5/nR) cos(theta/2), dy = 1/N. The sphere |p - (0, yc, 0)| < R
is integrated exactly in z and with Gauss-Legendre quadrature in (x, y):
    V_phase = int int 2 min( x t, sqrt(max(0, R^2 - x^2 - (y-yc)^2)) ) dx dy
The result replaces internalField in <timeDir>/alpha.phase1 (ASCII).

Unlike initAlphaField (linear interpolation of the implicit function along
the cell edges = faceted, inscribed interface, fraction error O(Delta/R)),
the fractions here are exact to ~1e-6 (total volume).

Usage: python3 exactAlphaAxi.py <timeDir> N [R] [yc] [wedgeAngleDeg] [nQuad]
       e.g. exactAlphaAxi.py 0 80 0.25 0.5 5
Needs <timeDir>/C in ASCII (postProcess -func writeCellCentres -time 0).
"""
import re
import sys
import numpy as np

tdir = sys.argv[1]
N = int(sys.argv[2])
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
yc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
theta = np.radians(float(sys.argv[5]) if len(sys.argv) > 5 else 5.0)
nq = int(sys.argv[6]) if len(sys.argv) > 6 else 64

t = np.tan(0.5*theta)
nR = N//2
dx = (0.5/nR)*np.cos(0.5*theta)
dy = 1.0/N


def read_vec(name):
    s = open(f"{tdir}/{name}").read()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?(\d+)\s*\n?\(", s)
    if not m:
        raise SystemExit(f"{tdir}/{name}: need an ASCII nonuniform vector field")
    n = int(m.group(1))
    vals = re.findall(r"\(([^()]*)\)", s[m.end():])[:n]
    return np.array([list(map(float, v.split())) for v in vals])


C = read_vec("C")
nCells = len(C)

# Gauss-Legendre points on [0,1], split each cell into 4x4 sub-cells so the
# kink of min(x t, sqrt(...)) and the sphere edge are resolved
gx, gw = np.polynomial.legendre.leggauss(nq)
gx = 0.5*(gx + 1)
gw = 0.5*gw
nsub = 4


def phase_volume(x0, x1, y0, y1):
    V = 0.0
    for a in range(nsub):
        for b in range(nsub):
            xa = x0 + (x1 - x0)*(a + gx)/nsub
            yb = y0 + (y1 - y0)*(b + gx)/nsub
            X, Y = np.meshgrid(xa, yb, indexing="ij")
            W = np.outer(gw, gw)*((x1 - x0)/nsub)*((y1 - y0)/nsub)
            zs = np.sqrt(np.maximum(0.0, R**2 - X**2 - (Y - yc)**2))
            V += np.sum(W*2*np.minimum(X*t, zs))
    return V


alpha = np.zeros(nCells)
for c in range(nCells):
    I = int(np.floor(C[c, 0]/dx + 1e-9))
    J = int(np.floor(C[c, 1]/dy + 1e-9))
    x0, x1 = I*dx, (I + 1)*dx
    y0, y1 = J*dy, (J + 1)*dy
    # distance range of the cell's meridional rectangle to the drop centre
    dmin = np.hypot(x0, max(y0 - yc, 0.0, yc - y1))
    dmax = np.hypot(x1, max(abs(y0 - yc), abs(y1 - yc)))
    if dmax*np.sqrt(1 + t**2) <= R:
        alpha[c] = 1.0
        continue
    if dmin >= R:
        alpha[c] = 0.0
        continue
    Vcell = t*(x1**2 - x0**2)*(y1 - y0)
    alpha[c] = min(max(phase_volume(x0, x1, y0, y1)/Vcell, 0.0), 1.0)

# total-volume check against the exact sphere sector
Vtot = 0.0
for c in range(nCells):
    I = int(np.floor(C[c, 0]/dx + 1e-9))
    x0, x1 = I*dx, (I + 1)*dx
    Vtot += alpha[c]*t*(x1**2 - x0**2)*dy
Vexact = 4/3*np.pi*R**3*theta/(2*np.pi)
print(f"cells {nCells}, interface cells {np.sum((alpha > 1e-12) & (alpha < 1 - 1e-12))}")
print(f"phase volume {Vtot:.10e}, exact sphere sector {Vexact:.10e}, "
      f"rel. diff {(Vtot - Vexact)/Vexact:+.2e}")

# write
fname = f"{tdir}/alpha.phase1"
s = open(fname).read()
body = "internalField   nonuniform List<scalar>\n%d\n(\n%s\n)\n;" % (
    nCells, "\n".join(f"{v:.16g}" for v in alpha))
s, n = re.subn(r"internalField\s+(uniform[^;]*|nonuniform[\s\S]*?\)\s*);", body, s, count=1)
if n != 1:
    raise SystemExit("could not find internalField in " + fname)
open(fname, "w").write(s)
print(f"written {fname}")
