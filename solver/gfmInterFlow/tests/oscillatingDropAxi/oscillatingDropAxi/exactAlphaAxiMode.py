#!/usr/bin/env python3
"""
Exact initial volume fractions of an axisymmetric drop with a Legendre-mode
perturbation on the staticDropAxi / oscillatingDropAxi wedge mesh.

Shape (spherical coordinates about (0, yc, 0), polar angle from +y):
    r(theta) = R0 [1 + eps P_l(cos theta)]
R0 is chosen so that the drop volume equals the sphere 4/3 pi R^3 exactly:
    (R0/R)^3 = 2 / int_{-1}^{1} (1 + eps P_l(mu))^3 dmu
(for l = 2: (R0/R)^3 = 1/(1 + 3 eps^2/5 + 2 eps^3/35)).
eps = 0 gives the sphere (same result as exactAlphaAxi.py).

Wedge cells (blockMesh, axis = y, radial = x, half-angle theta_w/2):
    x in [I dx, (I+1) dx],  y in [J dy, (J+1) dy],  |z| <= x tan(theta_w/2)
At height y the drop is the disc s = sqrt(x^2 + z^2) < S(y), where
    S(y) = r(th) sin(th)   with   r(th) cos(th) = y - yc     (bisection),
so the phase volume of a cell is
    V = int int 2 min( x t, sqrt(max(0, S(y)^2 - x^2)) ) dx dy
integrated with Gauss-Legendre (nq points) on 4x4 sub-cells.
The result replaces internalField in <timeDir>/alpha.phase1 (ASCII).

Usage:
  exactAlphaAxiMode.py <timeDir> N [R] [yc] [wedgeAngleDeg] [l] [eps] [nQuad]
  e.g. exactAlphaAxiMode.py 0 80 0.25 0.5 5 2 0.05
Needs <timeDir>/C in ASCII (postProcess -func writeCellCentres -time 0).
"""
import re
import sys
import numpy as np
from numpy.polynomial import legendre as L

tdir = sys.argv[1]
N = int(sys.argv[2])
R = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
yc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
thetaW = np.radians(float(sys.argv[5]) if len(sys.argv) > 5 else 5.0)
lmode = int(sys.argv[6]) if len(sys.argv) > 6 else 2
eps = float(sys.argv[7]) if len(sys.argv) > 7 else 0.05
nq = int(sys.argv[8]) if len(sys.argv) > 8 else 64

t = np.tan(0.5*thetaW)
nR = N//2
dx = (0.5/nR)*np.cos(0.5*thetaW)
dy = 1.0/N

cl = np.zeros(lmode + 1)
cl[lmode] = 1.0


def Pl(mu):
    return L.legval(mu, cl)


# volume-preserving base radius
mu_g, w_g = L.leggauss(200)
R0 = R*(2.0/np.sum(w_g*(1 + eps*Pl(mu_g))**3))**(1.0/3.0)


def rshape(th):
    return R0*(1 + eps*Pl(np.cos(th)))


# extreme radii (for the full/empty cell test)
thg = np.linspace(0, np.pi, 20001)
rg = rshape(thg)
rmin, rmax = rg.min(), rg.max()
yTop = yc + rshape(0.0)
yBot = yc - rshape(np.pi)

# monotonicity of r cos(th) (needed for a unique S(y))
g = rg*np.cos(thg)
if np.any(np.diff(g) >= 0):
    raise SystemExit("r(theta) cos(theta) not monotonic: eps too large for this mode")


def S_of_y(y):
    """Radial extent of the drop at heights y (array)."""
    y = np.asarray(y, dtype=float)
    S = np.zeros_like(y)
    inside = (y < yTop) & (y > yBot)
    if not np.any(inside):
        return S
    target = y[inside] - yc
    lo = np.zeros_like(target)          # g(lo) >= target
    hi = np.full_like(target, np.pi)    # g(hi) <= target
    for _ in range(60):
        mid = 0.5*(lo + hi)
        gm = rshape(mid)*np.cos(mid)
        above = gm > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    th = 0.5*(lo + hi)
    S[inside] = rshape(th)*np.sin(th)
    return S


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

gx, gw = np.polynomial.legendre.leggauss(nq)
gx = 0.5*(gx + 1)
gw = 0.5*gw
nsub = 4


def phase_volume(x0, x1, y0, y1):
    V = 0.0
    for b in range(nsub):
        yb = y0 + (y1 - y0)*(b + gx)/nsub
        S2 = S_of_y(yb)**2
        for a in range(nsub):
            xa = x0 + (x1 - x0)*(a + gx)/nsub
            X, S2m = np.meshgrid(xa, S2, indexing="ij")
            W = np.outer(gw, gw)*((x1 - x0)/nsub)*((y1 - y0)/nsub)
            zs = np.sqrt(np.maximum(0.0, S2m - X**2))
            V += np.sum(W*2*np.minimum(X*t, zs))
    return V


alpha = np.zeros(nCells)
for c in range(nCells):
    I = int(np.floor(C[c, 0]/dx + 1e-9))
    J = int(np.floor(C[c, 1]/dy + 1e-9))
    x0, x1 = I*dx, (I + 1)*dx
    y0, y1 = J*dy, (J + 1)*dy
    dmin = np.hypot(x0, max(y0 - yc, 0.0, yc - y1))
    dmax = np.hypot(x1, max(abs(y0 - yc), abs(y1 - yc)))
    if dmax*np.sqrt(1 + t**2) <= rmin:
        alpha[c] = 1.0
        continue
    if dmin >= rmax:
        alpha[c] = 0.0
        continue
    Vcell = t*(x1**2 - x0**2)*(y1 - y0)
    alpha[c] = min(max(phase_volume(x0, x1, y0, y1)/Vcell, 0.0), 1.0)

# checks: total volume and quadrupole moment against the exact shape
Vtot = 0.0
for c in range(nCells):
    I = int(np.floor(C[c, 0]/dx + 1e-9))
    x0, x1 = I*dx, (I + 1)*dx
    Vtot += alpha[c]*t*(x1**2 - x0**2)*dy
Vexact = 4/3*np.pi*R**3*thetaW/(2*np.pi)
print(f"mode l = {lmode}, eps = {eps}, R0/R = {R0/R:.10f}")
print(f"cells {nCells}, interface cells {np.sum((alpha > 1e-12) & (alpha < 1 - 1e-12))}")
print(f"phase volume {Vtot:.10e}, exact {Vexact:.10e}, rel. diff {(Vtot - Vexact)/Vexact:+.2e}")
# exact moment Q_l = int r^l P_l dV = 2 pi int r(mu)^(l+3)/(l+3) P_l dmu, per wedge
Qex = 2*np.pi*np.sum(w_g*(R0*(1 + eps*Pl(mu_g)))**(lmode + 3)/(lmode + 3)*Pl(mu_g))
print(f"exact moment Q_{lmode} (full drop) {Qex:.10e}, "
      f"linear estimate 4 pi eps R^{lmode+3}/((2l+1)) = "
      f"{4*np.pi*eps*R**(lmode+3)/(2*lmode+1):.10e}")

fname = f"{tdir}/alpha.phase1"
s = open(fname).read()
body = "internalField   nonuniform List<scalar>\n%d\n(\n%s\n)\n;" % (
    nCells, "\n".join(f"{v:.16g}" for v in alpha))
s, n = re.subn(r"internalField\s+(uniform[^;]*|nonuniform[\s\S]*?\)\s*);", body, s, count=1)
if n != 1:
    raise SystemExit("could not find internalField in " + fname)
open(fname, "w").write(s)
print(f"written {fname}")
