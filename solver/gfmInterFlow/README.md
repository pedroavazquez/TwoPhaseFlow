# gfmInterFlow

Ghost-fluid (sharp) surface-tension extension of the `interFlow` solver from
the [TwoPhaseFlow](https://github.com/DLR-RY/TwoPhaseFlow) library
(Scheufler & Roenby, OpenFOAM Journal 3, 2023).

Based on TwoPhaseFlow master `ba10702` (2026-09-08), which compiles with
OpenFOAM ESI v2406 to v2606.

## What changes with respect to interFlow

Only the face force entering the momentum predictor (`UEqn.H`) and the
pressure equation (`pEqn.H`) changes. It is computed in `interfaceForce.H`
and selected at run time:

```
PIMPLE
{
    ghostFluid  off;   // default: CSF, identical to interFlow
//  ghostFluid  on;    // ghost fluid / sharp surface force
}
```

* `ghostFluid off`:
  `F_f = sigma_f Kf snGrad(alpha1) - ghf snGrad(rho)`

* `ghostFluid on`:
  `F_f = deltaCoeff_f (H_N - H_P) (sigma kappa_G - (rho1 - rho2) gh_G)`

  where `H = Heaviside(alpha1 - 0.5)`, `theta = (alpha_P - 0.5)/(alpha_P - alpha_N)`
  and `(.)_G = (1 - theta)(.)_P + theta (.)_N`.

  This is the jump `[p_rgh]_PN = (H_N - H_P)(sigma kappa - [rho] gh)`
  moved to the right-hand side of the face pressure gradient. It is still a
  balanced-force discretisation, so the jump enters `phig` and the flux
  reconstruction consistently.

The curvature comes from whichever TwoPhaseFlow `surfaceTensionForceModel` is
selected (`constantCurvature`, `gradAlpha`, `RDF`, `fitParaboloid`,
`heightFunction`). It is read from the registered field `K_`, so the library
itself is not modified.

Files changed relative to `solver/interFlow`:

| File | Change |
|---|---|
| `interfaceForce.H` | new |
| `gfmInterFlow.C` | renamed, `#include "interfaceForce.H"` before `UEqn.H`, new help note |
| `UEqn.H` | force replaced by `interfaceForce` |
| `pEqn.H` | `phig = interfaceForce*rAUf*magSf` |
| `Make/files` | new executable name |

## Installation

```bash
# 1. TwoPhaseFlow (once)
git clone https://github.com/DLR-RY/TwoPhaseFlow
cd TwoPhaseFlow
./Allwmake

# 2. gfmInterFlow must live in TwoPhaseFlow/solver/
tar xzf gfmInterFlow.tar.gz -C solver/
cd solver/gfmInterFlow
./Allwmake
```

`Make/options` uses the relative paths `../../src/...` exactly like
`interFlow`, hence the required location.

## Verification: static drop matrix

```bash
cd tests/staticDropMatrix
./Allrun                          # all 8 cases (40^3 mesh each)
./Allrun heightFunction RDF       # subset
python3 collect.py                # summary table only
```

The cases are built from `run/velocityStaticCircle` (3D sphere, R = 0.25,
sigma = 1, rho1/rho2 and nu from the original case). In each copy the script
sets `application gfmInterFlow`, the curvature model, `curv 8` (= 2/R, used
only by `constantCurvature`) and `PIMPLE/ghostFluid`.

What to expect:

* `constantCurvature`: both CSF and GFM give velocities at round-off level.
  This checks the balanced-force property of the implementation.
* With estimated curvature, the error in max|U| is dominated by the
  curvature model, e.g. heightFunction << RDF < gradAlpha.
* GFM mainly sharpens the pressure jump: `max(p) - min(p)` should approach 8.

## Known limitations of this first version

* `rAUf` is still the linear interpolate of `rAU` with a smeared density. A
  theta-weighted (harmonic in 1/rho) face coefficient is the next step for high
  density ratios.
* The interface point is found from linear interpolation of alpha1 along
  P->N. The PLIC intersection would be more accurate.
* Non-coupled patches carry no GFM force, so there is no contact-line
  treatment. Avoid wall-attached interfaces with `ghostFluid on`.
* The explicit non-orthogonal correction of `p_rgh` does not include the jump.
  Use orthogonal meshes for verification.
* With `cyclic` patches and gravity along the cyclic direction, the
  neighbour `gh` is not geometrically adjacent. Processor patches are fine.
* The force on a face switches on and off as the interface crosses
  alpha1 = 0.5, which may cause some temporal noise in dynamic cases.

## License

GPL v3 or later, as OpenFOAM and TwoPhaseFlow.
