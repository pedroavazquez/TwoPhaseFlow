#!/usr/bin/env python3
"""
Summary of the capillaryDtAxi calibration (cases case_cMu-<value>[_scaled]).

All quantities are made dimensionless per case (parameters in caseInfo), so
SI and scaled runs are directly comparable:

  dt/tauMu  median time step after the start-up ramp, in units of
            tauMu = (mu1 + mu2) Delta/sigma, Delta = 8 R/res
  dt/tauRho same in units of tauRho = sqrt(2 rho Delta^3/(2 pi sigma))
  steps     number of time steps
  tEnd      last time reached, in units of tau_th
  tau/th    relaxation time (least-squares fit of ln eps over
            0.3 < t/tau_th < 3) relative to the Stokes theory
  tau/ref   relative to the cMu = 0 run of the same unit system
  dev       max |eps(t) - eps_ref(t)| / eps0 over the common time range
  Ulate     max|U| over the last 20 % of the run relative to its peak;
            the physical flow decays like eps, so values ~1 flag growth

Debris (alpha outside 1.5 R from the drop centre, newer cases only):
  first time with any cell alpha > 1e-12 outside, max vDebris/vDrop,
  max number of such cells, and the largest outside alpha with its (r, y)/R

Theory (small deformation, Stokes, unbounded outer fluid):
  tau_th = (mu2 R/sigma) (2 lam + 3)(19 lam + 16) / (40 (lam + 1)),  lam = mu1/mu2

Usage: python3 collect.py [--plot]
"""
import bisect
import glob
import math
import os
import sys


def read_info(case):
    fn = os.path.join(case, "caseInfo")
    if not os.path.exists(fn):
        # cases made before caseInfo existed: SI defaults, RES = 128
        cmu = case.split("cMu-")[1]
        return dict(R=30e-6, mu1=1e-3, mu2=0.1, sigma=5e-3, rho1=1000.0,
                    res=128.0, cMu=cmu)
    tok = open(fn).read().split()
    d = dict(zip(tok[0::2], tok[1::2]))
    out = {k: float(v) for k, v in d.items() if k != "cMu"}
    out["cMu"] = d.get("cMu", "?")
    return out


def read_table(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            vals = []
            for x in line.split():
                try:
                    vals.append(float(x))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)
    rows.sort(key=lambda r: r[0])
    return rows


def fit_tau(t, e, tau_th):
    pts = [(ti, math.log(abs(ei))) for ti, ei in zip(t, e)
           if 0.3*tau_th < ti < 3*tau_th and ei != 0 and math.isfinite(ei)]
    if len(pts) < 5:
        return float("nan")
    n = len(pts)
    mt = sum(p[0] for p in pts)/n
    ml = sum(p[1] for p in pts)/n
    sxx = sum((p[0] - mt)**2 for p in pts)
    sxy = sum((p[0] - mt)*(p[1] - ml) for p in pts)
    slope = sxy/sxx
    return -1.0/slope if slope < 0 else float("inf")


def interp(t, e, tq):
    i = bisect.bisect_left(t, tq)
    if i <= 0:
        return e[0]
    if i >= len(t):
        return e[-1]
    w = (tq - t[i-1])/(t[i] - t[i-1])
    return (1 - w)*e[i-1] + w*e[i]


def load(case):
    p = read_info(case)
    lam = p["mu1"]/p["mu2"]
    p["tau_th"] = (p["mu2"]*p["R"]/p["sigma"]) \
        * (2*lam + 3)*(19*lam + 16)/(40*(lam + 1))
    delta = 8*p["R"]/p["res"]
    p["tauMu"] = (p["mu1"] + p["mu2"])*delta/p["sigma"]
    p["tauRho"] = math.sqrt(2*p["rho1"]*delta**3/(2*math.pi*p["sigma"]))
    ds = read_table(os.path.join(case, "postProcessing/dropShape/*/dropShape.dat"))
    mm = read_table(os.path.join(case, "postProcessing/minMax/*/fieldMinMax.dat"))
    log = os.path.join(case, "log.gfmInterFlow")
    p["ended"] = os.path.exists(log) and any(l.startswith("End") for l in open(log))
    p["t"] = [r[0] for r in ds]
    p["eps"] = [r[3] for r in ds]
    p["dt"] = [r[4] for r in ds]
    # debris columns (cases made after the dropShape update)
    p["deb"] = [r[5:10] for r in ds if len(r) >= 10]
    p["tm"] = [r[0] for r in mm]
    p["um"] = [max(r[1:]) for r in mm]
    return p


def suffix(c):
    return "_scaled" if c.endswith("_scaled") else ""


def key(c):
    v = c.split("cMu-")[1].replace("_scaled", "")
    return (suffix(c), -1.0 if v == "legacy" else float(v))


cases = sorted(glob.glob("case_cMu-*"), key=key)
data = {c: load(c) for c in cases}

print(f"{'case':22s} {'dt/tauMu':>9s} {'dt/tauRho':>9s} {'steps':>7s} {'tEnd':>6s} "
      f"{'tau/th':>7s} {'tau/ref':>7s} {'dev':>8s} {'Ulate':>8s}  status")

for c in cases:
    d = data.get(c)
    if d is None:
        continue
    if not d["t"]:
        print(f"{c:22s}  no dropShape output")
        continue
    th = d["tau_th"]
    ref = data.get("case_cMu-0" + suffix(c))
    t, eps = d["t"], d["eps"]
    dts = sorted(x for ti, x in zip(t, d["dt"]) if ti > 0.1*th)
    dt_med = dts[len(dts)//2] if dts else float("nan")
    tau = fit_tau(t, eps, th)
    tau_ref = fit_tau(ref["t"], ref["eps"], th) if ref and ref["t"] else float("nan")
    dev = float("nan")
    if ref and ref["t"]:
        tmax = min(t[-1], ref["t"][-1])
        dev = max(abs(e - interp(ref["t"], ref["eps"], ti))
                  for ti, e in zip(t, eps) if ti <= tmax)/abs(eps[0])
    ulate = float("nan")
    if d["um"]:
        upk = max(d["um"])
        t0 = 0.8*d["tm"][-1]
        ulate = max(u for ti, u in zip(d["tm"], d["um"]) if ti >= t0)/upk
    bad = (not d["ended"]) or any(not math.isfinite(e) for e in eps) \
        or ulate > 0.5 or (math.isfinite(dev) and dev > 0.05)
    status = "ok" if not bad else ("FAILED" if not d["ended"] else "UNSTABLE")
    print(f"{c:22s} {dt_med/d['tauMu']:9.3e} {dt_med/d['tauRho']:9.3e} {len(t):7d} "
          f"{t[-1]/th:6.2f} {tau/th:7.4f} {tau/tau_ref:7.4f} {dev:8.2e} "
          f"{ulate:8.2e}  {status}")

print()
print(f"{'case':22s} {'tFirst':>7s} {'vDeb/vDrop':>11s} {'nDeb':>6s} "
      f"{'alphaMaxOut':>12s} {'at r/R':>7s} {'y/R':>7s}   (debris outside 1.5 R)")
for c in cases:
    d = data.get(c)
    if d is None or not d.get("deb"):
        continue
    th, R = d["tau_th"], d["R"]
    tdeb = [ti for ti, r in zip(d["t"], d["deb"]) if r[1] > 0]
    imax = max(range(len(d["deb"])), key=lambda i: d["deb"][i][2])
    rm = d["deb"][imax]
    print(f"{c:22s} {(tdeb[0]/th if tdeb else float('nan')):7.3f} "
          f"{max(r[0] for r in d['deb']):11.3e} {int(max(r[1] for r in d['deb'])):6d} "
          f"{rm[2]:12.3e} {rm[3]/R:7.3f} {rm[4]/R:7.3f}")

if "--plot" in sys.argv:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in cases:
        d = data.get(c)
        if d is None or not d["t"]:
            continue
        th = d["tau_th"]
        U0 = d["sigma"]/d["mu2"]
        lab = c.replace("case_", "")
        ax[0].semilogy([x/th for x in d["t"]], [abs(e) for e in d["eps"]], label=lab)
        ax[1].semilogy([x/th for x in d["tm"]], [u/U0 for u in d["um"]], label=lab)
    t = [i*0.05 for i in range(120)]
    ax[0].semilogy(t, [0.05*math.exp(-x) for x in t], "k--", label="Stokes theory")
    ax[0].set_xlabel("t / tau_th")
    ax[0].set_ylabel("|eps|")
    ax[1].set_xlabel("t / tau_th")
    ax[1].set_ylabel("max|U| mu2/sigma")
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("capillaryDtAxi.png", dpi=130)
    print("written capillaryDtAxi.png")
