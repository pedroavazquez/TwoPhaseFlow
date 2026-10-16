#!/usr/bin/env python3
"""
Taylor-drop summary (cases case_<name>_N<res>, written by Allrun).

Per case, from postProcessing/dropShape/*/dropShape.dat and taylor.info:
  a(t)    P2 amplitude from the moment, exact inversion q = a + 4/7 a^2 + ...
  D_Q     = 3a/(4 + a)          (L = R0(1+a), B = R0(1-a/2))
  D_axis  = (halfL - Req)/(halfL + Req) - same at t = 0
            (axis-column half length and equatorial radius; the t = 0 value
             of a sphere is the bias of these two estimators and is removed)
Steady value = mean over the last 10% of the run; drift = D(end) - D(90%).

Theory (small deformation, Ca = eps_e E^2 a / sigma_s with E = E0 (DC) or
E_rms (AC), lambda = mu_i/mu_e):
  R^ = (sigma_i + i w eps_i)/(sigma_e + i w eps_e)   (w = 0 for DC)
  S  = eps_i/eps_e
  Phi = 1 + |R^|^2 - 2S + 3/5 (2 + 3 lambda)/(1 + lambda) (Re R^ - S)
  D  = 9 Ca Phi / (16 |R^ + 2|^2)
DC: Taylor (1966); AC time-averaged: Torza, Cox & Mason (1971).

Usage: python3 taylorDeformation.py [--no-plot]
"""
import glob
import math
import os
import re
import sys

import numpy as np

CASE_RE = re.compile(r"^case_(?P<name>[A-Za-z]+?)_N(?P<res>\d+)(?P<tag>_\w+)?$")


def read_info(case):
    info = {}
    for line in open(os.path.join(case, "taylor.info")):
        k, v = line.split()
        info[k] = v if k == "mode" else float(v)
    return info


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
            data[v[0]] = v[1:]
    if not data:
        return None
    t = np.array(sorted(data))
    a = np.array([data[x] for x in t])
    return {"t": t, "vol": a[:, 0], "q": a[:, 2], "halfL": a[:, 3], "Req": a[:, 4]}


def amplitude_from_moment(q):
    mu, wq = np.polynomial.legendre.leggauss(64)
    P2 = 0.5*(3*mu**2 - 1)
    a_tab = np.linspace(-0.4, 0.4, 4001)
    q_tab = np.empty_like(a_tab)
    for i, a in enumerate(a_tab):
        r0 = (2.0/np.sum(wq*(1 + a*P2)**3))**(1.0/3.0)
        q_tab[i] = 0.5*np.sum(wq*(r0*(1 + a*P2))**5*P2)
    return np.interp(q, q_tab, a_tab)


def theory(info):
    w = info["omega"] if info["mode"] == "AC" else 0.0
    Rh = complex(info["sigmaI"], w*info["epsI"])/complex(info["sigmaE"], w*info["epsE"])
    S = info["epsI"]/info["epsE"]
    lam = info["lambda"]
    Phi = 1 + abs(Rh)**2 - 2*S + 0.6*(2 + 3*lam)/(1 + lam)*(Rh.real - S)
    return 9*info["Ca"]*Phi/(16*abs(Rh + 2)**2), Rh, S


def fmt(x, f="%.4e"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f % x


rows = []
series = {}
for case in sorted(glob.glob("case_*")):
    m = CASE_RE.match(case)
    if not m or not os.path.isfile(os.path.join(case, "taylor.info")):
        continue
    d = read_shape(case)
    if d is None or len(d["t"]) < 10:
        print(f"{case}: no (or too short) dropShape data")
        continue
    info = read_info(case)
    Dth, Rh, S = theory(info)

    t = d["t"]
    amp = amplitude_from_moment(d["q"])
    DQ = 3*amp/(4 + amp)
    Dax = (d["halfL"] - d["Req"])/(d["halfL"] + d["Req"])
    Dax = Dax - Dax[0]

    tail = t >= 0.9*t[-1]
    DQs = float(np.mean(DQ[tail]))
    Daxs = float(np.mean(Dax[tail]))
    i90 = int(np.searchsorted(t, 0.9*t[-1]))
    drift = float(DQ[-1] - DQ[i90])

    rows.append((m.group("name") + (m.group("tag") or ""), int(m.group("res")), info["mode"], info["Ca"],
                 Rh, S, info["sigmaE"], Dth, DQs, Daxs, drift,
                 (d["vol"][-1] - d["vol"][0])/d["vol"][0], t[-1],
                 info.get("aDelta", int(m.group("res"))/8)))
    series[case] = (t, DQ, Dax, Dth)

if not rows:
    print("No Taylor-drop cases with data found.")
    sys.exit(0)

hdr = ("case", "N", "a/Delta", "mode", "Ca", "R", "S", "sigma_e", "D_theory", "D (P2)",
       "err", "D (axis)", "drift", "dVol/Vol", "t_end")
tab = []
for (name, res, mode, Ca, Rh, S, se, Dth, DQs, Daxs, drift, dvol, tend, ad) in rows:
    Rs = f"{Rh.real:.3g}" if abs(Rh.imag) < 1e-12 else f"{Rh.real:.3g}{Rh.imag:+.3g}i"
    tab.append((name, str(res), f"{ad:g}", mode, f"{Ca:g}", Rs, f"{S:g}", f"{se:g}",
                fmt(Dth, "%+.5f"), fmt(DQs, "%+.5f"),
                fmt((DQs - Dth)/abs(Dth) if Dth else None, "%+.2e"),
                fmt(Daxs, "%+.5f"), fmt(drift, "%+.1e"), fmt(dvol, "%+.1e"),
                fmt(tend, "%.2f")))
w = [max(len(r[i]) for r in tab + [hdr]) for i in range(len(hdr))]
line = "  ".join("{:<%d}" % x for x in w)
print(line.format(*hdr))
print(line.format(*["-"*x for x in w]))
for r in tab:
    print(line.format(*r))
print("\nerr = (D(P2) - D_theory)/|D_theory|; the theory is O(Ca), so expect an "
      "O(Ca) relative difference from the shape nonlinearity at finite Ca.")

if "--no-plot" not in sys.argv:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, no plots")
        sys.exit(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    for case, (t, DQ, Dax, Dth) in series.items():
        line_, = ax.plot(t, DQ, label=f"{case} (P2)")
        ax.plot(t, Dax, ":", color=line_.get_color(), lw=1)
        ax.axhline(Dth, color=line_.get_color(), ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("D = (L - B)/(L + B)")
    ax.set_title("Taylor drop: solid P2 moment, dotted axis/equator, dashed theory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("taylorDeformation.png", dpi=120)
    print("written taylorDeformation.png")
