/*---------------------------------------------------------------------------*\
  gfmInterFlow - electrohydrodynamics (EHD) extension
-------------------------------------------------------------------------------
License
    GNU General Public License v3 or later (same as OpenFOAM / TwoPhaseFlow)

Class
    Foam::ehdModel

Description
    Leaky-dielectric (Taylor-Melcher) electrohydrodynamics for the
    two-phase VOF solver gfmInterFlow. The model returns a face-normal force
    density (N/m^3) that is added to interfaceForce, i.e. it enters the
    momentum and pressure equations in the same balanced-force form as the
    surface tension.

    Dictionary constant/electricProperties (file optional: absent = no EHD)

        ehdModel        DC;         // none | DC | AC

        epsilon1        [-1 -3 4 0 0 2 0] 2.0;   // permittivity, phase 1
        epsilon2        [-1 -3 4 0 0 2 0] 1.0;   // permittivity, phase 2
        sigma1          [-1 -3 3 0 0 2 0] 1.0;   // conductivity, phase 1
        sigma2          [-1 -3 3 0 0 2 0] 0.1;   // conductivity, phase 2

        epsilonMixing   harmonic;   // arithmetic | harmonic
        sigmaMixing     harmonic;   // arithmetic | harmonic

        forceForm       maxwellStress;  // maxwellStress (default)
                                        // | kortewegHelmholtz
        correctEveryOuterCorrector no;  // default: once per time step

        AC                          // only for ehdModel AC
        {
            omega           [0 0 -1 0 0 0 0] 1000;  // angular frequency
            nOuterCorrectors 50;    // max. Re/Im outer iterations per step
            tolerance        1e-8;  // max|change|/max|phi| to stop
            relaxation       1;     // under-relaxation of the outer loop
        }

    Material properties at cells and faces are mixed from alpha1 (faces:
    linear interpolate(alpha1)) with the selected law
        arithmetic: k = a k1 + (1 - a) k2
        harmonic  : k = k1 k2 / (a k2 + (1 - a) k1)

    DC (transient leaky dielectric, charge-conservative, implicit in the
    charge relaxation; Lopez-Herrera, Popinet & Herrada 2011):

        Gauss            : -div(eps grad phiE)            = rhoE
        charge transport : d(rhoE)/dt + div(rhoE u)        = div(sigma grad phiE)

        rhoE*  = rhoE^n - dt div(phi, rhoE^n)                (explicit advection)
        -div((eps + dt sigma) grad phiE^{n+1}) = rhoE*       (implicit relaxation)
        rhoE^{n+1} = -div(eps grad phiE^{n+1})

        Since div(eps grad) is the same discrete operator in the last two
        lines, rhoE^{n+1} - rhoE* = dt div(sigma grad phiE^{n+1}) exactly,
        and the total charge is conserved to solver tolerance. Stable for
        any dt relative to the charge relaxation time eps/sigma.
        Fields: phiE (MUST_READ, electrode BCs), rhoE (READ_IF_PRESENT,
        default 0, zeroGradient). Time-dependent electrodes: any
        time-dependent fixedValue BC (uniformFixedValue, ...).

        Force (Korteweg-Helmholtz, no electrostriction), face-normal:
            F_f = rhoE_f E_n - 1/2 |E|^2_f snGrad(eps)

    AC (time-averaged, harmonic potential phiE(t) = Re[phiHat e^{i omega t}],
    valid when the field period is short compared to the flow time scales):

        div((sigma + i omega eps) grad phiHat) = 0

        Real form with s = sigma, w = omega eps, phiHat = phiERe + i phiEIm:
            Ls phiERe - Lw phiEIm = 0,    Ls phiEIm + Lw phiERe = 0
        Each matrix row P is rotated by exp(-i theta_P), theta_P = phase of
        the complex diagonal of row P (sum over its faces of
        (s_f + i w_f) |S_f| deltaCoeff_f). With c = cos theta_P,
        s' = sin theta_P:
            M  = c Ls + s' Lw   (row-scaled, positive diagonal, asymmetric)
            B  = s' Ls - c Lw   (non-zero only near property jumps)
            M phiERe = -B phiEIm,   M phiEIm = +B phiERe
        solved by outer (block Gauss-Seidel) iterations. Any rotation gives
        the same converged solution; the diagonal phase makes the coupling
        B small in every row, also for sigma2 = 0 and either mixing law.
        M is asymmetric: use PBiCGStab (or smoothSolver) for phiE(Re|Im).

        Complex free charge rhoHat = -div(eps grad phiHat), and the
        time-averaged force
            F_f = 1/2 (rhoRe_f ERe_n + rhoIm_f EIm_n)
                - 1/4 (|ERe|^2 + |EIm|^2)_f snGrad(eps)
        Fields: phiERe, phiEIm (MUST_READ; e.g. electrode phiERe = V0,
        phiEIm = 0, ground 0).

    Force forms (forceForm):

    kortewegHelmholtz (inconsistent for eps1 != eps2, see below):
        F_f = rhoE_f E_n - 1/2 |E|^2_f snGrad(eps)          (DC)
        F_f = 1/2 (rhoRe_f ERe_n + rhoIm_f EIm_n)
            - 1/4 (|ERe|^2 + |EIm|^2)_f snGrad(eps)           (AC)

    With an eps jump and no free charge, D_n = eps E_n is continuous and
    the exact integral across the interface is
        int 1/2 E_n^2 d(eps) = 1/2 [eps] E_n1 E_n2,
    while the interpolated |E|^2_f gives about 1/2 [eps] (E_n1^2 + E_n2^2)/2:
    too large by (1 + r^2)/(2r), r = eps2/eps1, on every mesh.

    maxwellStress (default): divergence of the (time-averaged) Maxwell stress
        DC:  T = eps (E E - 1/2 |E|^2 I)
        AC:  T = 1/2 eps (ERe ERe + EIm EIm) - 1/4 eps (|ERe|^2 + |EIm|^2) I
    with cell E = reconstruct(-snGrad(phi) |Sf|) (consistent with the face
    normal fields). On the control volume centred on face f (cf. Basilisk
    ehd/stress.h), n.div(T) is split into
        F_f = deltaCoeff_f [(n.T.n)_N - (n.T.n)_P]                  compact
            + interp_f[ n.div(T) - n_i n_j n_k d_i T_jk ]            remainder
    The compact part is the jump of the normal Maxwell stress across the
    face, discretised like snGrad(p_rgh), so the electric pressure jump is
    balanced exactly. The remainder (tangential shear d_t(T_tn), hoop terms
    on wedges) is interpolated from cells: div(T) by Gauss
    (surfaceIntegrate(Sf & interpolate(T))), d_i T_jk from the gradients of
    the six components. In smooth regions the sum is n.div(T) to second
    order. No free-charge density is needed for the force.

    Output (AUTO_WRITE): E or ERe/EIm, rhoE or rhoERe/rhoEIm,
    electricForce (cell-reconstructed force density).

    The force is zero on non-coupled boundary faces, as for the surface
    tension in interfaceForce.H.

SourceFiles
    ehdModel.C

\*---------------------------------------------------------------------------*/

#ifndef ehdModel_H
#define ehdModel_H

#include "fvCFD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                          Class ehdModel Declaration
\*---------------------------------------------------------------------------*/

class ehdModel
{
public:

    //- Model types
    enum class modelType { none, DC, AC };


private:

    // Private Data

        const fvMesh& mesh_;

        const volScalarField& alpha1_;

        const surfaceScalarField& phi_;

        IOdictionary dict_;

        modelType type_;

        dimensionedScalar eps1_;
        dimensionedScalar eps2_;
        dimensionedScalar sigma1_;
        dimensionedScalar sigma2_;

        bool harmonicEps_;
        bool harmonicSigma_;

        //- Force from the Maxwell stress tensor (else Korteweg-Helmholtz)
        bool maxwellStress_;

        //- Re-solve in every PIMPLE outer corrector (default: once per
        //  time step, on the first call of the step)
        bool everyOuterCorrector_;

        //- Time index of the last solve
        label lastTimeIndex_;

        // AC controls
        dimensionedScalar omega_;
        label nOuterAC_;
        scalar tolAC_;
        scalar relaxAC_;

        // DC fields
        autoPtr<volScalarField> phiE_;
        autoPtr<volScalarField> rhoE_;
        autoPtr<volVectorField> E_;

        // AC fields
        autoPtr<volScalarField> phiERe_;
        autoPtr<volScalarField> phiEIm_;
        autoPtr<volScalarField> rhoERe_;
        autoPtr<volScalarField> rhoEIm_;
        autoPtr<volVectorField> ERe_;
        autoPtr<volVectorField> EIm_;

        //- Face-normal force density (oriented)
        surfaceScalarField force_;

        //- Cell-reconstructed force for output
        autoPtr<volVectorField> forceC_;


    // Private Member Functions

        //- Mixing law, single value
        static scalar mixValue
        (
            const scalar a,
            const scalar k1,
            const scalar k2,
            const bool harmonic
        );

        //- Property at cells
        tmp<volScalarField> mixCell
        (
            const word& name,
            const dimensionedScalar& k1,
            const dimensionedScalar& k2,
            const bool harmonic
        ) const;

        //- Property at faces (mixed from interpolate(alpha1))
        tmp<surfaceScalarField> mixFace
        (
            const word& name,
            const dimensionedScalar& k1,
            const dimensionedScalar& k2,
            const bool harmonic
        ) const;

        //- Zero the force on non-coupled boundary faces
        void zeroNonCoupled(surfaceScalarField& f) const;

        //- Face-normal force n.div(T) from a cell Maxwell stress T:
        //  compact normal-stress jump + interpolated remainder
        tmp<surfaceScalarField> maxwellFaceForce(volSymmTensorField& T) const;

        //- Cell field E consistent with the face normal fields
        tmp<volVectorField> faceConsistentE(const volScalarField& phi) const;

        void correctDC();

        void correctAC();


public:

    // Constructors

        ehdModel
        (
            const volScalarField& alpha1,
            const surfaceScalarField& phi
        );

        //- No copy construct
        ehdModel(const ehdModel&) = delete;

        //- No copy assignment
        void operator=(const ehdModel&) = delete;


    // Member Functions

        //- Any EHD model selected?
        bool active() const
        {
            return type_ != modelType::none;
        }

        //- Solve the electric problem for the current alpha1 and phi and
        //  update the force. By default this is done once per time step
        //  (later calls in the same step return immediately and keep the
        //  force); correctEveryOuterCorrector yes re-solves in every outer
        //  corrector. Uses only old-time charge, so either is consistent.
        void correct();

        //- Face-normal electric force density (N/m^3), oriented
        const surfaceScalarField& force() const
        {
            return force_;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
