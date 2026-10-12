#!/usr/bin/env python3
"""
Oscillating-drop benchmark summary (cases case_<model>_GFM-<on|off>_N<res>).

Per case, from postProcessing/dropShape/*/dropShape.dat (coded FO):
    q(t) = Q(t)/Q_lin,   Q = sum alpha V ((y-yc)^2 - s^2/2)
    a(t) = P_2 amplitude from the exact nonlinear relation q = a + 4/7 a^2 + ...
Fit (Levenberg-Marquardt, numpy only):
    a(t) = c + A exp(-gamma t) cos(omega t + phi)
Cross-check: period and log decrement from the extrema (parabolic refinement).

Reference: exact linear normal mode s = -gamma + i omega of the viscous
drop in a viscous unbounded fluid (l = 2; Miller & Scriven 1968, Prosperetti
1980): u = grad(phi) + curl curl(r chi), phi ~ r^2, r^-3 and chi ~ i_2(k1 r),
k_2(k2 r) times P_2, k^2 = s rho/mu; continuity of u_r, u_theta, tau_rtheta
and the normal-stress jump sigma (l-1)(l+2) zeta/R^2 -> 4x4 determinant,
root by complex secant. Lamb's irrotational estimate (leading order in Oh,
printed for comparison only)
    omega0^2 = l(l-1)(l+1)(l+2) sigma / (R^3 [(l+1) rho1 + l rho2])
    gamma    = (2l+1) [(l+1)(l-1) mu1 + l(l+2) mu2] / (R^2 [(l+1) rho1 + l rho2])
is 10% off in gamma at Oh = 0.03 (boundary-layer correction ~ Oh^1/2).
Reference curve from rest: a0 exp(-gamma t) [cos(w t) + gamma/w sin(w t)].

Time-discretisation damping: the capillary oscillation behaves like implicit
Euler, |amplification| = 1/sqrt(1 + (w dt)^2), i.e. an extra damping
    gamma_Euler = ln(1 + (w dt)^2)/(2 dt) ~ w^2 dt/2
(confirmed by the dt/2 test at N=40). For ddtSchemes Euler the table also
gives the error of gamma - gamma_Euler (spatial part).

Output: table, Richardson orders for omega and gamma (three successive N),
oscillation.csv, oscillation_<model>_GFM-<gfm>.png (matplotlib optional).

Usage: python3 fitOscillation.py [--no-plot] [--tmin T]
"""
import cmath
import glob
import math
import os
import re
import sys

import numpy as np

SIGMA = 1.0
R = 0.25

CASE_RE = re.compile(r"^case_(?P<model>\w+?)_GFM-(?P<gfm>on|off)_N(?P<res>\d+)(?P<tag>_\w+)?$")

args = sys.argv[1:]
doPlot = "--no-plot" not in args
tmin = 0.0
if "--tmin" in args:
    tmin = float(args[args.index("--tmin") + 1])


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_phase(case, phase):
    txt = open(os.path.join(case, "constant", "transportProperties")).read()
    blk = re.search(phase + r"\s*\{([^}]*)\}", txt, re.S).group(1)
    num = r"(?:\[[^\]]*\]\s*)?([0-9eE.+-]+)"
    nu = float(re.search(r"\bnu\s+(?:nu\s+)?" + num, blk).group(1))
    rho = float(re.search(r"\brho\s+(?:rho\s+)?" + num, blk).group(1))
    return rho, nu * rho


def read_mode(case):
    try:
        s = open(os.path.join(case, "log.exactAlphaAxiMode")).read()
        m = re.search(r"mode l = (\d+), eps = ([0-9eE.+-]+)", s)
        return int(m.group(1)), float(m.group(2))
    except Exception:
        return 2, None


def read_shape(case):
    files = glob.glob(os.path.join(case, "postProcessing", "dropShape", "*", "dropShape.dat"))

    def t0(f):
        try:
            return float(os.path.basename(os.path.dirname(f)))
        except ValueError:
            return -1.0

    data = {}
    for f in sorted(files, key=t0):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            data[v[0]] = v[1:]        # later restarts overwrite overlaps
    if not data:
        return None
    t = np.array(sorted(data))
    a = np.array([data[x] for x in t])
    return {"t": t, "vol": a[:, 0], "Q": a[:, 1], "eps": a[:, 2],
            "halfL": a[:, 3], "Req": a[:, 4]}


def read_maxU(case):
    files = glob.glob(os.path.join(case, "postProcessing", "minMax", "*", "*.dat"))
    best = 0.0
    for f in files:
        idx = None
        for line in open(f):
            s = line.strip()
            if s.startswith("#"):
                toks = s.lstrip("#").split()
                if toks and toks[0] == "Time":
                    for i, h in enumerate(toks):
                        if "max(mag(U))" in h or "max(U)" in h:
                            idx = i
                            break
                continue
            if idx is None or not s:
                continue
            toks = re.findall(r"\([^()]*\)|[^\s()]+", s)
            try:
                best = max(best, float(toks[idx]))
            except (ValueError, IndexError):
                pass
    return best


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def model_f(p, t):
    c, A, g, w, ph = p
    return c + A*np.exp(-g*t)*np.cos(w*t + ph)


def jac(p, t):
    c, A, g, w, ph = p
    e = np.exp(-g*t)
    co = np.cos(w*t + ph)
    si = np.sin(w*t + ph)
    return np.column_stack([np.ones_like(t), e*co, -A*t*e*co, -A*t*e*si, -A*e*si])


def lm_fit(t, y, p0, nit=200):
    p = np.array(p0, dtype=float)
    lam = 1e-3
    r = y - model_f(p, t)
    cost = r @ r
    for _ in range(nit):
        J = jac(p, t)
        H = J.T @ J
        gvec = J.T @ r
        improved = False
        for _ in range(20):
            dp = np.linalg.solve(H + lam*np.diag(np.diag(H) + 1e-30), gvec)
            pn = p + dp
            rn = y - model_f(pn, t)
            cn = rn @ rn
            if cn < cost:
                p, r, cost = pn, rn, cn
                lam = max(lam/3, 1e-12)
                improved = True
                break
            lam *= 4
        if not improved or np.max(np.abs(dp)/(np.abs(p) + 1e-12)) < 1e-12:
            break
    return p, math.sqrt(cost/len(t))


def extrema_estimate(t, y):
    """Period and decay rate from refined extrema of y(t)."""
    ext = []
    for i in range(1, len(y) - 1):
        if (y[i] - y[i-1])*(y[i+1] - y[i]) < 0:
            # parabola through three points
            t0, t1, t2 = t[i-1], t[i], t[i+1]
            y0, y1, y2 = y[i-1], y[i], y[i+1]
            den = (t0 - t1)*(t0 - t2)*(t1 - t2)
            a = (t2*(y1 - y0) + t1*(y0 - y2) + t0*(y2 - y1))/den
            b = (t2**2*(y0 - y1) + t1**2*(y2 - y0) + t0**2*(y1 - y2))/den
            cc = y1 - a*t1**2 - b*t1
            te = -b/(2*a) if a != 0 else t1
            ext.append((te, a*te**2 + b*te + cc))
    if len(ext) < 3:
        return None, None
    te = np.array([e[0] for e in ext])
    ye = np.array([e[1] for e in ext])
    halfT = np.diff(te)
    omega = math.pi/np.mean(halfT)
    # log decrement of the peak-to-peak values (independent of the offset):
    # |y_k - y_k+1| = a_k (1 + exp(-gamma T/2)) decays like exp(-gamma t)
    pp = np.abs(np.diff(ye))
    tm = 0.5*(te[1:] + te[:-1])
    gamma = -np.polyfit(tm, np.log(pp), 1)[0]
    return omega, gamma


def amplitude_from_moment(q):
    """
    Invert q = Q/Q_lin for the P_2 amplitude a of r = R0(a)[1 + a P_2],
    R0(a) volume preserving. Exact relation (Gauss quadrature):
        q(a) = (5/2) int (R0/R)^5 (1 + a P_2)^5 P_2 dmu / 5
             = a + (4/7) a^2 + O(a^3)
    The quadratic term would otherwise appear as a 2-omega harmonic and
    an offset in the fitted signal.
    """
    mu, wq = np.polynomial.legendre.leggauss(64)
    P2 = 0.5*(3*mu**2 - 1)
    a_tab = np.linspace(-0.4, 0.4, 4001)
    q_tab = np.empty_like(a_tab)
    for i, a in enumerate(a_tab):
        r0 = (2.0/np.sum(wq*(1 + a*P2)**3))**(1.0/3.0)
        q_tab[i] = 0.5*np.sum(wq*(r0*(1 + a*P2))**5*P2)
    return np.interp(q, q_tab, a_tab)


def exact_mode2(rho1, mu1, rho2, mu2, sigma, R):
    """Exact normal mode s = -gamma + i omega of the l = 2 mode (linear,
    viscous drop in a viscous unbounded fluid). Closed-form spherical Bessel
    functions, determinant of the 4x4 interface conditions, complex secant."""
    l = 2
    L = l*(l + 1)

    def i_l(x):        # i_1, i_2 scaled by exp(-x)
        e = np.exp(-x)
        sh, ch = 0.5*(1 - e*e), 0.5*(1 + e*e)
        i1 = (x*ch - sh)/x**2
        i2 = ((3/x**2 + 1)*sh - 3*ch/x)/x
        return i1, i2

    def k_l(x):        # k_1, k_2 scaled by exp(+x)
        return (1 + 1/x)/x, (1 + 3/x + 3/x**2)/x

    def cols(s, inner):
        rho, mu = (rho1, mu1) if inner else (rho2, mu2)
        kap = cmath.sqrt(s*rho/mu)
        if kap.real < 0:
            kap = -kap
        x = kap*R
        # potential part a(r) = r^l (inside) or r^-(l+1) (outside)
        if inner:
            a, a1, a2 = R**l, l*R**(l - 1), l*(l - 1)*R**(l - 2)
            b1, b2 = i_l(x)
            g = b2
            gp = kap*(b1 - (l + 1)/x*b2)
        else:
            m = -(l + 1)
            a, a1, a2 = R**m, m*R**(m - 1), m*(m - 1)*R**(m - 2)
            b1, b2 = k_l(x)
            g = b2
            gp = kap*(-b1 - (l + 1)/x*b2)
        gpp = -2/R*gp + (L/R**2 + kap**2)*g
        out = []
        for (A0, A1, A2, G0, G1, G2, pot) in ((a, a1, a2, 0, 0, 0, True),
                                             (0, 0, 0, g, gp, gpp, False)):
            U = A1 + L*G0/R
            Up = A2 + L*(G1/R - G0/R**2)
            V = A0/R + (G0 + R*G1)/R
            # V(r) = a/r + g/r + g' ;  V' = a'/r - a/r^2 - g/r^2 + g'/r + g''
            Vp = A1/R - A0/R**2 - G0/R**2 + G1/R + G2
            p = -rho*s*A0
            trr = -p + 2*mu*Up
            trt = mu*(Vp - V/R + U/R)      # r (V/r)' = V' - V/r
            out.append((U, V, trt, trr))
        return out

    def det(s):
        ci, co = cols(s, True), cols(s, False)
        M = np.zeros((4, 4), complex)
        for j, (U, V, trt, trr) in enumerate(ci):
            M[:, j] = [U, V, trt, trr + sigma*(l - 1)*(l + 2)*U/(s*R**2)]
        for j, (U, V, trt, trr) in enumerate(co):
            M[:, 2 + j] = [-U, -V, -trt, -trr]
        return np.linalg.det(M)

    den = (l + 1)*rho1 + l*rho2
    w0 = np.sqrt(l*(l - 1)*(l + 1)*(l + 2)*sigma/(R**3*den))
    g0 = (2*l + 1)*((l + 1)*(l - 1)*mu1 + l*(l + 2)*mu2)/(R**2*den)
    z0, z1 = complex(-g0, w0), complex(-0.9*g0, 0.999*w0)
    f0, f1 = det(z0), det(z1)
    for _ in range(100):
        z2 = z1 - f1*(z1 - z0)/(f1 - f0)
        z0, f0, z1, f1 = z1, f1, z2, det(z2)
        if abs(z1 - z0) < 1e-14*abs(z1):
            break
    return -z1.real, abs(z1.imag)


def read_dt_ddt(case):
    """deltaT and the default ddt scheme of a case."""
    dt, ddt = None, "Euler"
    try:
        cd = open(os.path.join(case, "system", "controlDict")).read()
        dt = float(re.search(r"^\s*deltaT\s+([0-9eE.+-]+)\s*;", cd, re.M).group(1))
    except Exception:
        pass
    try:
        fs = open(os.path.join(case, "system", "fvSchemes")).read()
        blk = re.search(r"ddtSchemes\s*\{([^}]*)\}", fs, re.S).group(1)
        ddt = re.search(r"default\s+([^;]+);", blk).group(1).strip()
    except Exception:
        pass
    return dt, ddt


def fmt(x, f="%.4e"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f % x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

results = {}
csv_rows = []
for case in sorted(glob.glob("case_*")):
    m = CASE_RE.match(case)
    if not m:
        continue
    key = (m.group("model") + (m.group("tag") or ""), m.group("gfm"), int(m.group("res")))
    d = read_shape(case)
    if d is None or len(d["t"]) < 10:
        print(f"{case}: no (or too short) dropShape data")
        continue
    rho1, mu1 = read_phase(case, "phase1")
    rho2, mu2 = read_phase(case, "phase2")
    l, epsIn = read_mode(case)
    den = (l + 1)*rho1 + l*rho2
    w0 = math.sqrt(l*(l - 1)*(l + 1)*(l + 2)*SIGMA/(R**3*den))
    gth = (2*l + 1)*((l + 1)*(l - 1)*mu1 + l*(l + 2)*mu2)/(R**2*den)
    wLamb = math.sqrt(max(w0**2 - gth**2, 0.0))
    gLamb = gth
    gth, wth = exact_mode2(rho1, mu1, rho2, mu2, SIGMA, R) if l == 2 else (gLamb, wLamb)
    dt, ddt = read_dt_ddt(case)

    t = d["t"]
    eps = amplitude_from_moment(d["eps"]) if l == 2 else d["eps"]
    d["amp"] = eps
    sel = t >= tmin
    p, rms = lm_fit(t[sel], eps[sel], [0.0, eps[0], gth, wth, 0.0])
    if p[1] < 0:
        p[1], p[4] = -p[1], p[4] + math.pi
    wE, gE = extrema_estimate(t[sel], eps[sel])
    w = abs(p[3])
    gEul = math.log(1 + (w*dt)**2)/(2*dt) if (dt and ddt == "Euler") else None

    res = {"t": t[-1], "eps0": eps[0], "epsIn": epsIn, "l": l,
           "w": w, "g": p[2], "gEul": gEul,
           "gc": p[2] - gEul if gEul is not None else None,
           "wLamb": wLamb, "gLamb": gLamb, "dt": dt, "ddt": ddt, "c": p[0], "rms": rms, "wE": wE, "gE": gE,
           "wth": wth, "gth": gth, "w0": w0,
           "dvol": (d["vol"][-1] - d["vol"][0])/d["vol"][0],
           "ca": read_maxU(case)*mu1/SIGMA, "d": d}
    results[key] = res
    for i in range(len(t)):
        csv_rows.append((case, key[2], t[i], d["vol"][i], d["eps"][i], eps[i],
                         d["halfL"][i], d["Req"][i]))

if not results:
    print("No cases with dropShape data found.")
    sys.exit(0)

r0 = next(iter(results.values()))
print(f"Exact normal mode (l = {r0['l']}): omega = {r0['wth']:.6f}, gamma = {r0['gth']:.6f}, "
      f"period = {2*math.pi/r0['wth']:.4f} s")
print(f"Lamb estimate:            omega = {r0['wLamb']:.6f}, gamma = {r0['gLamb']:.6f}")
print()

hdr = ("model", "GFM", "N", "dt", "eps(0)", "omega_fit", "err_w",
       "gamma_fit", "err_g", "err_g-Euler", "omega_ext", "gamma_ext", "offset c", "fit rms", "dVol/Vol", "max Ca")
tab = []
for k in sorted(results):
    r = results[k]
    egc = (r["gc"] - r["gth"])/r["gth"] if r["gc"] is not None else None
    tab.append((k[0], k[1], str(k[2]), fmt(r["dt"], "%g"), fmt(r["eps0"], "%.5f"),
                fmt(r["w"], "%.6f"), fmt((r["w"] - r["wth"])/r["wth"], "%+.2e"),
                fmt(r["g"], "%.5f"), fmt((r["g"] - r["gth"])/r["gth"], "%+.2e"), fmt(egc, "%+.2e"),
                fmt(r["wE"], "%.6f"), fmt(r["gE"], "%.5f"), fmt(r["c"], "%+.1e"),
                fmt(r["rms"], "%.1e"), fmt(r["dvol"], "%+.1e"), fmt(r["ca"], "%.2e")))
w = [max(len(row[i]) for row in tab + [hdr]) for i in range(len(hdr))]
line = "  ".join("{:<%d}" % x for x in w)
print(line.format(*hdr))
print(line.format(*["-"*x for x in w]))
for row in tab:
    print(line.format(*row))

# Richardson: order from three successive resolutions, extrapolated value
print()
print("Richardson (N, 2N, 4N):  p = log2(|f_N - f_2N| / |f_2N - f_4N|),"
      "  f_inf = f_4N + (f_4N - f_2N)/(2^p - 1)")
for model, gfm in sorted({(k[0], k[1]) for k in results}):
    ress = sorted(k[2] for k in results if k[:2] == (model, gfm))
    for a, b, c in zip(ress, ress[1:], ress[2:]):
        if b != 2*a or c != 2*b:
            continue
        for q, name, th in (("w", "omega", "wth"), ("g", "gamma", "gth"),
                            ("gc", "gamma-Euler", "gth")):
            fa, fb, fc = (results[(model, gfm, n)][q] for n in (a, b, c))
            if None in (fa, fb, fc):
                continue
            if abs(fb - fc) > 0 and abs(fa - fb) > 0:
                pr = math.log(abs(fa - fb)/abs(fb - fc))/math.log(2)
                finf = fc + (fc - fb)/(2**pr - 1) if pr > 0 else float("nan")
            else:
                pr, finf = float("nan"), float("nan")
            thv = results[(model, gfm, c)][th]
            print(f"  {model} GFM-{gfm} {a}/{b}/{c} {name}: p = {pr:.2f}, "
                  f"f_inf = {finf:.6f}  (theory {thv:.6f}, "
                  f"diff {(finf - thv)/thv:+.2e})")

with open("oscillation.csv", "w") as f:
    f.write("case,N,time,vol,q,amp,halfL,Req\n")
    for row in csv_rows:
        f.write(",".join(str(x) for x in row) + "\n")
print("\nwritten oscillation.csv")

if doPlot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, no plots")
        sys.exit(0)
    for model, gfm in sorted({(k[0], k[1]) for k in results}):
        keys = sorted(k for k in results if k[:2] == (model, gfm))
        fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        tt = None
        for k in keys:
            r = results[k]
            d = r["d"]
            ax[0].plot(d["t"], d["amp"], label=f"N={k[2]}")
            if tt is None or d["t"][-1] >= tt[-1]:
                tt, e0 = d["t"], r["eps0"]
        rr = results[keys[-1]]
        wd, g = rr["wth"], rr["gth"]
        th = e0*np.exp(-g*tt)*(np.cos(wd*tt) + g/wd*np.sin(wd*tt))
        ax[0].plot(tt, th, "k--", lw=1, label="exact normal mode (from rest)")
        ax[0].set_ylabel("P2 amplitude a(t)")
        ax[0].legend()
        ax[0].set_title(f"oscillating drop, {model}, GFM {gfm}")
        for k in keys:
            r = results[k]
            d = r["d"]
            # each case against the normal mode started from its own a(0)
            th_k = r["eps0"]*np.exp(-g*d["t"])*(np.cos(wd*d["t"]) + g/wd*np.sin(wd*d["t"]))
            ax[1].plot(d["t"], d["amp"] - th_k, label=f"N={k[2]}")
        ax[1].set_ylabel("a - normal mode")
        ax[1].set_xlabel("t [s]")
        ax[1].legend()
        fig.tight_layout()
        fn = f"oscillation_{model}_GFM-{gfm}.png"
        fig.savefig(fn, dpi=120)
        print("written " + fn)
