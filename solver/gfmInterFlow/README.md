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

## Electrohydrodynamics (leaky dielectric, DC and AC)

`ehdModel` (files `ehdModel.H/.C`) adds the Taylor-Melcher leaky-dielectric
model. Its face-normal force is added to `interfaceForce`, so it enters
`UEqn.H` and `pEqn.H` in the same balanced form as the surface tension.
Without `constant/electricProperties` the solver behaves exactly as before.

### `constant/electricProperties`

```
ehdModel        DC;          // none | DC | AC

epsilon1        [-1 -3 4 0 0 2 0] 2.0;    // permittivity of phase 1 (alpha1 = 1)
epsilon2        [-1 -3 4 0 0 2 0] 1.0;
sigma1          [-1 -3 3 0 0 2 0] 1.0;    // conductivity
sigma2          [-1 -3 3 0 0 2 0] 0.1;

epsilonMixing   harmonic;    // arithmetic | harmonic (cells and faces)
sigmaMixing     harmonic;

forceForm       maxwellStress;       // default; or kortewegHelmholtz (only for eps1 = eps2)
correctEveryOuterCorrector no;       // default: solve once per time step

nNonOrthogonalCorrectors 0;  // DC only

AC
{
    omega            [0 0 -1 0 0 0 0] 1000;   // angular frequency
    nOuterCorrectors 50;
    tolerance        1e-8;
    relaxation       1;
}
```

### DC: transient charge (charge-conservative, implicit relaxation)

```
-div(eps grad phiE) = rhoE,     d(rhoE)/dt + div(rhoE u) = div(sigma grad phiE)
rhoE*  = rhoE^n - dt div(phi, rhoE^n)
-div((eps + dt sigma) grad phiE^{n+1}) = rhoE*
rhoE^{n+1} = -div(eps grad phiE^{n+1})
F_f = rhoE_f E_n - 1/2 |E|^2_f snGrad(eps)
```

Stable for any `dt` relative to the relaxation time `eps/sigma`, and the
total charge is conserved to solver tolerance. Fields: `0/phiE` (required,
electrode BCs; time-dependent fixedValue BCs give low-frequency AC in the
time domain) and `0/rhoE` (optional, default 0).

### AC: time-averaged force from the complex potential

```
div((sigma + i omega eps) grad phiHat) = 0,   phiHat = phiERe + i phiEIm
<F>_f = 1/2 (rhoRe_f ERe_n + rhoIm_f EIm_n) - 1/4 (|ERe|^2 + |EIm|^2)_f snGrad(eps)
```

Valid when the field period is short compared with the flow time scales
(the 2 omega part of the force is dropped). The real and imaginary parts are
solved by outer iterations after rotating each matrix row by the phase of
its complex diagonal (see `ehdModel.H`). This keeps the coupling small even
for `sigma2 = 0` and either mixing law; in 1D tests it converges in 2-14
iterations for conductivity ratios up to 1e6. Fields: `0/phiERe`, `0/phiEIm`
(required; e.g. electrode `phiERe = V0`, `phiEIm = 0`, ground 0).

### `system/fvSolution` and `system/fvSchemes`

The electric equations always use the plain entries below (never
`phiEFinal` etc.), since they are solved once per outer corrector.

```
solvers
{
    phiE                          // DC: symmetric matrix
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-10;
        relTol          0;
    }
    "phiE(Re|Im)"                 // AC: row-rotated, asymmetric matrix
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-10;
        relTol          0;
    }
}
```

```
divSchemes
{
    div(phi,rhoE)   Gauss vanLeer;     // DC charge advection
}
```

Output: `phiE`, `rhoE`, `E` (DC) or `phiERe`, `phiEIm`, `rhoERe`,
`rhoEIm`, `ERe`, `EIm` (AC), and `electricForce` (cell-reconstructed force
density) in both modes.

### Force form: `maxwellStress`

The (time-averaged) Maxwell stress is built at cells from
`E = reconstruct(-snGrad(phi)|Sf|)`:

```
DC:  T = eps (E E - 1/2 |E|^2 I)
AC:  T = 1/2 eps (ERe ERe + EIm EIm) - 1/4 eps (|ERe|^2 + |EIm|^2) I
F_f = deltaCoeff_f [(n.T.n)_N - (n.T.n)_P]                (compact normal-stress jump)
    + interp_f[ n.div(T) - n_i n_j n_k d_i T_jk ]         (tangential shear, hoop terms)
```

The compact part is discretised like `snGrad(p_rgh)` (same delta
coefficient), so the jump of the electric normal stress is balanced exactly
by the pressure, as in Basilisk's `ehd/stress.h`. No free-charge density is
used for the force. In parallel it adds about ten halo exchanges per step and
no global reductions.

### Once per time step

The electric problem (potential, charge, force) is solved on the first call
of each time step; later PIMPLE outer correctors reuse the force
(`correctEveryOuterCorrector yes` restores a solve per corrector). Fields
used only for output (`E`/`ERe`/`EIm`, `rhoERe`/`rhoEIm` with the
Maxwell-stress force, `electricForce`) are computed only at write times. On
2D and wedge meshes the gradients of the out-of-plane stress components are
skipped (they multiply n_k = 0).

## Capillary time-step limit (inertial + viscous)

By default the time step is bounded by the Brackbill limit of the
surfaceForces library, `deltaT <= maxCapillaryNum * dtBKZ` with
`dtBKZ = sqrt((rho1 + rho2) Delta^3/(2 pi sigma))`. For viscous flows
(Oh = mu/sqrt(rho sigma R) >~ 1) this is far too restrictive: grid-scale
capillary waves are overdamped and the explicit surface tension is stable up
to a viscous-capillary time `~ (mu1 + mu2) Delta/sigma` (Galusinski &
Vigneaux, J. Comput. Phys. 227, 2008). Adding to `system/controlDict`

```
capillaryTimeStep
{
    cRho    0.1;    // inertial branch, cRho*tauRho (1 = Brackbill)
    cMu     0;      // viscous branch,  cMu*tauMu (off by default)
    muMode  min;    // tauMu with min(mu1, mu2) (default) or mu1 + mu2
}
maxCapillaryNum 1;  // still applied: deltaT <= maxCapillaryNum*dtCap
```

switches to (file `capillaryTimeStep.H`, evaluated per interface cell, minimum
over the interface)

```
tauRho = sqrt((rho1 + rho2) Delta^3/(2 pi sigma)),   tauMu = (mu1 + mu2) Delta/sigma
a = cMu tauMu,  b = cRho tauRho,   dtCap = (a + sqrt(a^2 + 4 b^2))/2
```

which tends to `b` for inviscid and to `a` for Stokes flow. Without the
dictionary the solver behaves exactly as before.

Delta is the smallest centre-to-centre distance over the internal and coupled
faces of the cell. The library limit uses `fvc::average(mag(mesh.delta()))`,
which on wedge meshes includes the wedge faces with patch-normal delta
`r sin(theta/2)`; for interface cells next to the axis this gives
`Delta ~ 0.17 dx` and a Brackbill step ~14 times too small. `cMu 0` gives the
Brackbill limit with the unbiased Delta.

`cMu` has to be calibrated for the scheme (PISO, explicit CSF, isoAdvection):
`tests/capillaryDtAxi` runs a relaxing mode-2 drop with the physics of the
co-flow case for several `cMu` and compares the relaxation time, `eps(t)` and
`max|U|(t)` with the `cMu = 0` run and the Stokes theory.

## Viscous stress at large viscous Fourier numbers

`turbulence->divDevRhoReff(rho, U)` treats `laplacian(mu, U)` implicitly and
the transpose part `div(mu dev2(grad(U)^T))` explicitly. For constant mu the
latter vanishes (div U = 0), but at a viscosity jump it is
`~ grad(mu).grad(U)^T`, of the same order as the implicit term, and the
explicit treatment needs a viscous Fourier number `mu dt/(rho Delta^2) = O(1)`.
In viscous capillary flows (Oh >~ 1) the capillary and Courant limits allow
far larger steps, and the run then blows up within ~50 steps
(tests/capillaryDtAxi: `mu2 dt/(rho Delta^2) = 15` with `mu1/mu2 = 0.01`).

```
PIMPLE
{
    viscousStress   explicitTranspose;    // default, as interFlow
//  viscousStress   stabilisedTranspose;  // + implicit-explicit stabiliser
//  cStab           1;                    //   -lap(cStab mu, U)^{n+1} + lap(cStab mu, U)^n
//  viscousStress   noTranspose;          // diagnostic only (wrong stress jump)
}
```

The stabiliser cancels at steady state and adds an O(dt) error
`~ cStab mu dt lap(dU/dt)`, which tests/capillaryDtAxi measures against the
small-step reference.

### Calibration (tests/capillaryDtAxi, scaled units, R/Delta = 16)

| case | dt/tauRho | result |
|---|---|---|
| mu1/mu2 = 0.01, Oh2 = 8 | 0.046, 0.069, 0.104 | stable, tau/tau_th = 1.025 (all three) |
| | 0.21, 0.42, 0.83 | unstable after 427, 57, 38 steps |
| mu1 = mu2 (Oh = 8) | 0.83 | stable |
| | 3.3, 14, 28 | unstable |
| mu1/mu2 = 0.01, nOuterCorrectors 3 | 0.83 | unstable after 13 steps |
| mu1/mu2 = 0.01, ghostFluid on | 0.21 | unstable after 427 steps (as CSF) |

The Brackbill limit is not sufficient for this scheme: use `cRho 0.1`. The
legacy limit was only stable on wedge meshes because of its Delta bias (~0.03
tauRho). The explicit transpose viscous term is not involved
(`viscousStress noTranspose` blows up at the same step).
