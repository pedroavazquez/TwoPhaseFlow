#!/usr/bin/env python3
"""
Summarise the static-drop benchmark (cases case_<model>_GFM-<on|off>_N<res>).

Per case, from the function objects in system/dropFunctions:
  max|U|(t)        fieldMinMax                       (postProcessing/minMax)
  <p>_core(t)      volume average in zone 'core'     (postProcessing/pCore)
  <p>_far(t)       volume average in zone 'far'      (postProcessing/pFar)

Derived quantities:
  Ca(t)  = max|U| mu_1 / sigma                  (mu_1 = nu*rho of phase1)
  Dp     = <p>_core - <p>_far,  error vs exact 2 sigma/R = 8

Output:
  - table at the final time (Ca_end, max Ca over the second half, Dp error)
  - convergence orders between successive resolutions
  - timeseries.csv  (all cases, all time steps)
  - staticDrop_Ca_N<res>.png  if matplotlib is available

Usage: python3 collect.py [--no-plot]
"""
import csv
import glob
import math
import os
import re
import sys

SIGMA = 1.0
DP_EXACT = 8.0

CASE_RE = re.compile(r"^case_(?P<model>\w+?)_GFM-(?P<gfm>on|off)_N(?P<res>\d+)$")
TOKEN = re.compile(r"\([^()]*\)|[^\s()]+")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_mu(case):
    """Dynamic viscosity nu*rho of phase1 from transportProperties."""
    try:
        txt = open(os.path.join(case, "constant", "transportProperties")).read()
        blk = re.search(r"phase1\s*\{([^}]*)\}", txt, re.S).group(1)
        num = r"(?:\[[^\]]*\]\s*)?([0-9eE.+-]+)"
        nu = float(re.search(r"\bnu\s+(?:nu\s+)?" + num, blk).group(1))
        rho = float(re.search(r"\brho\s+(?:rho\s+)?" + num, blk).group(1))
        return nu * rho
    except Exception:
        return None


def to_float(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def read_table(case, fo):
    """
    All .dat files of function object 'fo', concatenated in start-time order.
    Returns (header tokens, list of row token lists) or (None, []).
    """
    files = glob.glob(os.path.join(case, "postProcessing", fo, "*", "*.dat"))

    def start_time(f):
        try:
            return float(os.path.basename(os.path.dirname(f)))
        except ValueError:
            return -1.0

    header = None
    rows = []
    for f in sorted(files, key=lambda f: (start_time(f), f)):
        for line in open(f):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                toks = s.lstrip("#").split()
                if toks and toks[0] == "Time":
                    header = toks
                continue
            rows.append(TOKEN.findall(s))
    return header, rows


def column(header, *patterns, exclude=None):
    """Index of the first header token containing any of the patterns."""
    if not header:
        return None
    for i, h in enumerate(header[1:], start=1):
        if exclude and exclude in h:
            continue
        for p in patterns:
            if p in h:
                return i
    return None


def series(header, rows, idx):
    """{time: value} for column idx."""
    out = {}
    if idx is None:
        return out
    for r in rows:
        if len(r) > idx:
            t, v = to_float(r[0]), to_float(r[idx])
            if t is not None and v is not None:
                out[t] = v
    return out


def read_case(case):
    """Time series of max|U|, <p>_core, <p>_far for one case."""
    note = []

    hU, rU = read_table(case, "minMax")
    iU = column(hU, "max(mag(U))", "max(U)")
    maxU = series(hU, rU, iU)
    if not maxU:
        note.append("no max|U| (minMax header: %s)" % (" ".join(hU) if hU else "none"))

    avg = {}
    for fo in ("pCore", "pFar"):
        h, r = read_table(case, fo)
        avg[fo] = series(h, r, column(h, "(p)") or (1 if h and len(h) > 1 else None))
        if not avg[fo]:
            note.append("no <p> in %s (header: %s)" % (fo, " ".join(h) if h else "none"))

    return maxU, avg["pCore"], avg["pFar"], "; ".join(note)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def fmt(x, f="%.3e"):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f % x


def order(e_coarse, e_fine, n_coarse, n_fine):
    if None in (e_coarse, e_fine) or e_coarse <= 0 or e_fine <= 0:
        return None
    return math.log(e_coarse / e_fine) / math.log(n_fine / n_coarse)


results = {}
timeseries = []
notes = []
skipped = []

for case in sorted(glob.glob("case_*")):
    m = CASE_RE.match(case)
    if not m:
        skipped.append(case)
        continue
    model, gfm, res = m.group("model"), m.group("gfm"), int(m.group("res"))

    mu = read_mu(case)
    maxU, pCore, pFar, note = read_case(case)
    if note:
        notes.append("%s: %s" % (case, note))

    times = sorted(maxU)
    if not times:
        results[(model, gfm, res)] = None
        continue

    ca = {t: maxU[t] * mu / SIGMA if mu else float("nan") for t in times}

    dp = {t: pCore[t] - pFar[t] for t in times if t in pCore and t in pFar}

    for t in times:
        timeseries.append((case, model, gfm, res, t, maxU[t], ca[t], dp.get(t, "")))

    tEnd = times[-1]
    second_half = [ca[t] for t in times if t >= 0.5 * tEnd]
    dpEnd = dp.get(tEnd)
    results[(model, gfm, res)] = {
        "t": tEnd,
        "caEnd": ca[tEnd],
        "caMax2": max(second_half) if second_half else None,
        "dp": dpEnd,
        "dpErr": abs(dpEnd - DP_EXACT) / DP_EXACT if dpEnd is not None else None,
    }

if not results:
    print("No cases case_<model>_GFM-<on|off>_N<res> found.")
    if skipped:
        print("Ignored (old naming): " + " ".join(skipped))
    sys.exit(0)

# Table at the final time
hdr = ("model", "GFM", "N", "time", "Ca_end", "max Ca (2nd half)", "Dp", "|Dp-8|/8")
tab = []
for (model, gfm, res) in sorted(results, key=lambda k: (k[0], k[1], k[2])):
    r = results[(model, gfm, res)]
    if r is None:
        tab.append((model, gfm, str(res), "n/a", "n/a", "n/a", "n/a", "n/a"))
    else:
        tab.append((model, gfm, str(res), fmt(r["t"]), fmt(r["caEnd"]),
                    fmt(r["caMax2"]), fmt(r["dp"], "%.5f"), fmt(r["dpErr"])))

w = [max(len(row[i]) for row in tab + [hdr]) for i in range(len(hdr))]
line = "  ".join("{:<%d}" % x for x in w)
print(line.format(*hdr))
print(line.format(*["-" * x for x in w]))
for row in tab:
    print(line.format(*row))

# Convergence orders between successive resolutions
conv = []
for model, gfm in sorted({(k[0], k[1]) for k in results}):
    ress = sorted(k[2] for k in results if k[0] == model and k[1] == gfm
                  and results[k] is not None)
    for a, b in zip(ress, ress[1:]):
        ra, rb = results[(model, gfm, a)], results[(model, gfm, b)]
        conv.append((model, gfm, "%d->%d" % (a, b),
                     fmt(order(ra["caMax2"], rb["caMax2"], a, b), "%.2f"),
                     fmt(order(ra["dpErr"], rb["dpErr"], a, b), "%.2f")))

if conv:
    print()
    print("Convergence orders  log(e_coarse/e_fine)/log(N_fine/N_coarse)")
    h2 = ("model", "GFM", "N", "order Ca", "order Dp")
    w2 = [max(len(row[i]) for row in conv + [h2]) for i in range(len(h2))]
    l2 = "  ".join("{:<%d}" % x for x in w2)
    print(l2.format(*h2))
    print(l2.format(*["-" * x for x in w2]))
    for row in conv:
        print(l2.format(*row))

# Time series
with open("timeseries.csv", "w", newline="") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["case", "model", "GFM", "N", "time", "maxU", "Ca", "Dp"])
    wcsv.writerows(timeseries)
print()
print("Time series of all cases written to timeseries.csv")

# Plots
if "--no-plot" not in sys.argv and timeseries:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = sorted({k[0] for k in results})
        colours = {m: "C%d" % i for i, m in enumerate(models)}
        for res in sorted({k[2] for k in results}):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for model in models:
                for gfm, ls in (("off", "--"), ("on", "-")):
                    pts = [(r[4], r[6]) for r in timeseries
                           if r[1] == model and r[2] == gfm and r[3] == res
                           and r[6] > 0]
                    if pts:
                        t, c = zip(*pts)
                        ax.semilogy(t, c, ls, color=colours[model],
                                    label="%s, %s" % (model,
                                                      "GFM" if gfm == "on" else "CSF"))
            ax.set_xlabel("t [s]")
            ax.set_ylabel(r"Ca = max|U| $\mu_1/\sigma$")
            ax.set_title("Static drop, N = %d (R/$\\Delta$ = %g)" % (res, 0.25 * res))
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            name = "staticDrop_Ca_N%d.png" % res
            fig.savefig(name, dpi=150)
            plt.close(fig)
            print("Plot written to %s" % name)
    except ImportError:
        print("(matplotlib not available: no plots; use timeseries.csv)")

if notes:
    print()
    print("Notes:")
    for n in notes:
        print("  " + n)

if skipped:
    print()
    print("Ignored (old naming, remove with rm -rf): " + " ".join(skipped))
