#!/usr/bin/env python3
"""
Summary of the capillaryDtAxi calibration (cases case_cMu-<value>).

Per case:
  dt      median time step after the start-up ramp (t > 0.1 tau_th)
  steps   number of time steps
  tau     relaxation time from a least-squares fit of ln(eps) over
          0.3 tau_th < t < 3 tau_th
  tau/ref tau relative to the cMu = 0 run (Brackbill limit, unbiased Delta)
  dev     max |eps(t) - eps_ref(t)| / eps0 over the common time range
  Ulate   max|U| over the last 20 % of the run, relative to the peak max|U|.
          The physical flow decays like eps, i.e. Ulate ~ exp(-4 tau_th/tau);
          growth or a plateau far above that flags a capillary instability.

Theory (small deformation, Stokes, unbounded outer fluid):
  tau_th = (mu2 R/sigma) (2 lam + 3)(19 lam + 16) / (40 (lam + 1)),  lam = mu1/mu2

Usage: python3 collect.py [--plot]
"""
import glob
import math
import os
import sys

R, MU1, MU2, SIGMA = 30e-6, 1e-3, 0.1, 5e-3
LAM = MU1/MU2
TAU_TH = (MU2*R/SIGMA)*(2*LAM + 3)*(19*LAM + 16)/(40*(LAM + 1))


def read_table(pattern, ncols=None):
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            tok = line.split()
            try:
                vals = [float(x) for x in tok if not x.isalpha()]
            except ValueError:
                continue
            rows.append(vals)
    rows.sort(key=lambda r: r[0])
    return rows


def fit_tau(t, e):
    pts = [(ti, math.log(abs(ei))) for ti, ei in zip(t, e)
           if 0.3*TAU_TH < ti < 3*TAU_TH and ei != 0]
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
    import bisect
    i = bisect.bisect_left(t, tq)
    if i <= 0:
        return e[0]
    if i >= len(t):
        return e[-1]
    w = (tq - t[i-1])/(t[i] - t[i-1])
    return (1 - w)*e[i-1] + w*e[i]


def key(c):
    v = c.split("cMu-")[1]
    return -1.0 if v == "legacy" else float(v)


cases = sorted(glob.glob("case_cMu-*"), key=key)
data = {}
for c in cases:
    ds = read_table(os.path.join(c, "postProcessing/dropShape/*/dropShape.dat"))
    mm = read_table(os.path.join(c, "postProcessing/minMax/*/fieldMinMax.dat"))
    log = os.path.join(c, "log.gfmInterFlow")
    ended = os.path.exists(log) and any(l.startswith("End") for l in open(log))
    if not ds:
        data[c] = None
        continue
    t = [r[0] for r in ds]
    eps = [r[3] for r in ds]
    dt = [r[4] for r in ds]
    tm = [r[0] for r in mm]
    um = [max(r[1:]) for r in mm]
    data[c] = dict(t=t, eps=eps, dt=dt, tm=tm, um=um, ended=ended)

print(f"tau_th (Stokes, unbounded) = {TAU_TH:.4e} s,  "
      f"tau_mu = (mu1+mu2) Delta/sigma = {(MU1+MU2)*R/16/SIGMA:.3e} s at R/Delta = 16")
print()
print(f"{'case':16s} {'dt [s]':>10s} {'steps':>7s} {'tau [s]':>11s} "
      f"{'tau/tau_th':>10s} {'tau/ref':>8s} {'dev':>8s} {'Ulate':>8s}  status")

ref = data.get("case_cMu-0")
for c in cases:
    d = data[c]
    if d is None:
        print(f"{c:16s}  no dropShape output")
        continue
    t, eps = d["t"], d["eps"]
    late = [x for x in d["dt"][len(d["dt"])//10:]]
    dtm = sorted(x for ti, x in zip(t, d["dt"]) if ti > 0.1*TAU_TH) or late
    dt_med = dtm[len(dtm)//2] if dtm else float("nan")
    tau = fit_tau(t, eps)
    tau_ref = fit_tau(ref["t"], ref["eps"]) if ref else float("nan")
    dev = float("nan")
    if ref:
        tmax = min(t[-1], ref["t"][-1])
        dev = max(abs(e - interp(ref["t"], ref["eps"], ti))
                  for ti, e in zip(t, eps) if ti <= tmax)/abs(eps[0])
    ulate = float("nan")
    if d["um"]:
        upk = max(d["um"])
        t0 = 0.8*d["tm"][-1]
        ulate = max(u for ti, u in zip(d["tm"], d["um"]) if ti >= t0)/upk
    bad = (not d["ended"]) or any(math.isnan(e) for e in eps) \
        or (ulate > 0.5) or (not math.isnan(dev) and dev > 0.05)
    status = "UNSTABLE/FAILED" if bad else "ok"
    print(f"{c:16s} {dt_med:10.3e} {len(t):7d} {tau:11.4e} {tau/TAU_TH:10.4f} "
          f"{tau/tau_ref:8.4f} {dev:8.2e} {ulate:8.2e}  {status}")

if "--plot" in sys.argv:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in cases:
        d = data[c]
        if d is None:
            continue
        lab = c.replace("case_", "")
        ax[0].semilogy([x/TAU_TH for x in d["t"]],
                       [abs(e) for e in d["eps"]], label=lab)
        ax[1].semilogy([x/TAU_TH for x in d["tm"]], d["um"], label=lab)
    t = [i*0.05 for i in range(120)]
    e0 = abs(data[cases[0]]["eps"][0]) if data[cases[0]] else 0.05
    ax[0].semilogy(t, [e0*math.exp(-x) for x in t], "k--", label="Stokes theory")
    ax[0].set_xlabel("t / tau_th"); ax[0].set_ylabel("|eps|")
    ax[1].set_xlabel("t / tau_th"); ax[1].set_ylabel("max|U| [m/s]")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("capillaryDtAxi.png", dpi=130)
    print("written capillaryDtAxi.png")
