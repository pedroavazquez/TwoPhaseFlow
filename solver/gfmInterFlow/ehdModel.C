/*---------------------------------------------------------------------------*\
  gfmInterFlow - electrohydrodynamics (EHD) extension
-------------------------------------------------------------------------------
License
    GNU General Public License v3 or later (same as OpenFOAM / TwoPhaseFlow)

\*---------------------------------------------------------------------------*/

#include "ehdModel.H"

// * * * * * * * * * * * * * * * Local Data  * * * * * * * * * * * * * * * //

namespace
{
    // SI: [kg m s K mol A cd]
    const Foam::dimensionSet dimPermittivity(-1, -3, 4, 0, 0, 2, 0);
    const Foam::dimensionSet dimConductivity(-1, -3, 3, 0, 0, 2, 0);
    const Foam::dimensionSet dimPotential(1, 2, -3, 0, 0, -1, 0);
    const Foam::dimensionSet dimChargeDensity(0, -3, 1, 0, 0, 1, 0);

    bool readMixing(const Foam::dictionary& dict, const Foam::word& key)
    {
        const Foam::word law =
            dict.getOrDefault<Foam::word>(key, "harmonic");

        if (law == "harmonic")
        {
            return true;
        }
        if (law != "arithmetic")
        {
            FatalIOErrorInFunction(dict)
                << key << " " << law << " unknown. Valid: arithmetic harmonic"
                << Foam::exit(Foam::FatalIOError);
        }
        return false;
    }
}


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

Foam::scalar Foam::ehdModel::mixValue
(
    const scalar a0,
    const scalar k1,
    const scalar k2,
    const bool harmonic
)
{
    const scalar a = min(max(a0, scalar(0)), scalar(1));
    const scalar ar = a*k1 + (1 - a)*k2;

    if (!harmonic)
    {
        return ar;
    }

    // k1 k2/(a k2 + (1 - a) k1); a pure phase with a zero property
    // (e.g. sigma2 = 0 at a = 1) has a zero denominator -> pure value
    const scalar d = a*k2 + (1 - a)*k1;
    if (mag(d) <= VSMALL)
    {
        return ar;
    }
    return k1*k2/d;
}


Foam::tmp<Foam::volScalarField> Foam::ehdModel::mixCell
(
    const word& name,
    const dimensionedScalar& k1,
    const dimensionedScalar& k2,
    const bool harmonic
) const
{
    tmp<volScalarField> tf
    (
        new volScalarField
        (
            IOobject
            (
                name,
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar(k1.dimensions(), Zero),
            "calculated"
        )
    );
    volScalarField& f = tf.ref();

    scalarField& fi = f.primitiveFieldRef();
    forAll(fi, i)
    {
        fi[i] = mixValue(alpha1_[i], k1.value(), k2.value(), harmonic);
    }

    // Non-coupled patches from the boundary alpha; coupled and constraint
    // patches (processor, wedge, ...) by their own evaluation
    volScalarField::Boundary& bf = f.boundaryFieldRef();
    forAll(bf, patchi)
    {
        if (bf[patchi].coupled())
        {
            continue;
        }
        const fvPatchScalarField& ap = alpha1_.boundaryField()[patchi];
        fvPatchScalarField& fp = bf[patchi];
        forAll(fp, j)
        {
            fp[j] = mixValue(ap[j], k1.value(), k2.value(), harmonic);
        }
    }
    f.correctBoundaryConditions();

    return tf;
}


Foam::tmp<Foam::surfaceScalarField> Foam::ehdModel::mixFace
(
    const word& name,
    const dimensionedScalar& k1,
    const dimensionedScalar& k2,
    const bool harmonic
) const
{
    const surfaceScalarField af(fvc::interpolate(alpha1_));

    tmp<surfaceScalarField> tf
    (
        new surfaceScalarField
        (
            IOobject
            (
                name,
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar(k1.dimensions(), Zero)
        )
    );
    surfaceScalarField& f = tf.ref();

    scalarField& fi = f.primitiveFieldRef();
    const scalarField& ai = af.primitiveField();
    forAll(fi, i)
    {
        fi[i] = mixValue(ai[i], k1.value(), k2.value(), harmonic);
    }

    surfaceScalarField::Boundary& bf = f.boundaryFieldRef();
    forAll(bf, patchi)
    {
        const fvsPatchScalarField& ap = af.boundaryField()[patchi];
        fvsPatchScalarField& fp = bf[patchi];
        forAll(fp, j)
        {
            fp[j] = mixValue(ap[j], k1.value(), k2.value(), harmonic);
        }
    }

    return tf;
}


void Foam::ehdModel::zeroNonCoupled(surfaceScalarField& f) const
{
    surfaceScalarField::Boundary& bf = f.boundaryFieldRef();
    forAll(bf, patchi)
    {
        if (!mesh_.boundary()[patchi].coupled())
        {
            bf[patchi] = scalar(0);
        }
    }
}


Foam::tmp<Foam::volVectorField> Foam::ehdModel::faceConsistentE
(
    const volScalarField& phi
) const
{
    tmp<volVectorField> tE
    (
        fvc::reconstruct(-fvc::snGrad(phi)*mesh_.magSf())
    );
    tE.ref().correctBoundaryConditions();
    return tE;
}


Foam::tmp<Foam::surfaceScalarField> Foam::ehdModel::maxwellFaceForce
(
    volSymmTensorField& T
) const
{
    T.correctBoundaryConditions();

    // Cell divergence by Gauss (includes the hoop terms on wedges)
    volVectorField divT
    (
        "ehdDivT",
        fvc::surfaceIntegrate(mesh_.Sf() & fvc::interpolate(T))
    );
    divT.correctBoundaryConditions();

    // Gradients of the independent components of T. Components with an
    // index along a non-solved direction (empty, wedge normal) only meet
    // n_k = 0 on the in-plane faces and are skipped.
    const Vector<label>& gD = mesh_.geometricD();
    const label cmptI[6] = {0, 0, 0, 1, 1, 2};
    const label cmptJ[6] = {0, 1, 2, 1, 2, 2};
    boolList activeC(6, true);
    for (label c = 0; c < 6; ++c)
    {
        activeC[c] = (gD[cmptI[c]] != -1 && gD[cmptJ[c]] != -1);
    }

    PtrList<volVectorField> gradT(symmTensor::nComponents);
    forAll(gradT, c)
    {
        if (!activeC[c])
        {
            continue;
        }
        gradT.set
        (
            c,
            new volVectorField
            (
                "ehdGradT" + Foam::name(c),
                fvc::grad(T.component(direction(c)))
            )
        );
        gradT[c].correctBoundaryConditions();
    }

    // n.T.n
    auto Tnn = [](const symmTensor& t, const vector& n) -> scalar
    {
        return n & t & n;
    };

    // n.div(T) - n.grad(n.T.n) at a cell (n fixed = face normal)
    auto restTerm =
        [](const vector& n, const vector& dT, const FixedList<vector, 6>& g)
     -> scalar
    {
        const scalar nGradTnn =
            sqr(n.x())*(n & g[symmTensor::XX])
          + 2*n.x()*n.y()*(n & g[symmTensor::XY])
          + 2*n.x()*n.z()*(n & g[symmTensor::XZ])
          + sqr(n.y())*(n & g[symmTensor::YY])
          + 2*n.y()*n.z()*(n & g[symmTensor::YZ])
          + sqr(n.z())*(n & g[symmTensor::ZZ]);

        return (n & dT) - nGradTnn;
    };

    tmp<surfaceScalarField> tF
    (
        new surfaceScalarField
        (
            IOobject
            (
                "ehdMaxwellFaceForce",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh_,
            dimensionedScalar(dimPressure/dimLength, Zero)
        )
    );
    surfaceScalarField& F = tF.ref();
    F.setOriented();

    const surfaceVectorField& Sf = mesh_.Sf();
    const surfaceScalarField& magSf = mesh_.magSf();
    const surfaceScalarField& w = mesh_.weights();
    // Same coefficient as the implicit part of laplacian(rAUf, p_rgh)
    const surfaceScalarField& dC = mesh_.nonOrthDeltaCoeffs();

    // Internal faces
    {
        const labelUList& own = mesh_.owner();
        const labelUList& nei = mesh_.neighbour();
        scalarField& Fi = F.primitiveFieldRef();

        FixedList<vector, 6> gP;
        FixedList<vector, 6> gN;

        forAll(own, facei)
        {
            const label P = own[facei];
            const label N = nei[facei];
            const vector n = Sf[facei]/magSf[facei];

            for (label c = 0; c < 6; ++c)
            {
                gP[c] = activeC[c] ? gradT[c][P] : vector::zero;
                gN[c] = activeC[c] ? gradT[c][N] : vector::zero;
            }

            Fi[facei] =
                dC[facei]*(Tnn(T[N], n) - Tnn(T[P], n))
              + w[facei]*restTerm(n, divT[P], gP)
              + (1 - w[facei])*restTerm(n, divT[N], gN);
        }
    }

    // Coupled patches (processor, cyclic); others carry no force
    surfaceScalarField::Boundary& Fb = F.boundaryFieldRef();
    forAll(Fb, patchi)
    {
        if (!mesh_.boundary()[patchi].coupled())
        {
            Fb[patchi] = scalar(0);
            continue;
        }

        const fvPatchSymmTensorField& Tp = T.boundaryField()[patchi];
        const symmTensorField TP(Tp.patchInternalField());
        const symmTensorField TN(Tp.patchNeighbourField());

        const fvPatchVectorField& dTp = divT.boundaryField()[patchi];
        const vectorField dP(dTp.patchInternalField());
        const vectorField dN(dTp.patchNeighbourField());

        List<vectorField> gPp(6);
        List<vectorField> gNp(6);
        for (label c = 0; c < 6; ++c)
        {
            if (!activeC[c])
            {
                gPp[c].setSize(Fb[patchi].size(), Zero);
                gNp[c].setSize(Fb[patchi].size(), Zero);
                continue;
            }
            const fvPatchVectorField& gp = gradT[c].boundaryField()[patchi];
            gPp[c] = gp.patchInternalField();
            gNp[c] = gp.patchNeighbourField();
        }

        const vectorField& Sfp = Sf.boundaryField()[patchi];
        const scalarField& magSfp = magSf.boundaryField()[patchi];
        const scalarField& wp = w.boundaryField()[patchi];
        const scalarField& dCp = dC.boundaryField()[patchi];
        fvsPatchScalarField& Fp = Fb[patchi];

        FixedList<vector, 6> gP;
        FixedList<vector, 6> gN;

        forAll(Fp, i)
        {
            const vector n = Sfp[i]/magSfp[i];
            for (label c = 0; c < 6; ++c)
            {
                gP[c] = gPp[c][i];
                gN[c] = gNp[c][i];
            }

            Fp[i] =
                dCp[i]*(Tnn(TN[i], n) - Tnn(TP[i], n))
              + wp[i]*restTerm(n, dP[i], gP)
              + (1 - wp[i])*restTerm(n, dN[i], gN);
        }
    }

    return tF;
}


void Foam::ehdModel::correctDC()
{
    volScalarField& phiE = phiE_.ref();
    volScalarField& rhoE = rhoE_.ref();

    const dimensionedScalar dt = mesh_.time().deltaT();

    const surfaceScalarField epsf
    (
        mixFace("ehdEpsilonf", eps1_, eps2_, harmonicEps_)
    );
    const surfaceScalarField sigmaf
    (
        mixFace("ehdSigmaf", sigma1_, sigma2_, harmonicSigma_)
    );

    // Explicit advection of the old-time charge with the current flux
    const volScalarField rhoStar
    (
        "rhoEStar",
        rhoE.oldTime()
      - dt*fvc::div(phi_, rhoE.oldTime(), "div(phi,rhoE)")
    );

    // Implicit charge relaxation: -div((eps + dt sigma) grad phiE) = rhoStar
    // Solver settings from fvSolution/solvers/phiE (no '...Final' entry:
    // the electric problem is solved once per outer corrector)
    const surfaceScalarField gammaf("ehdGammaf", epsf + dt*sigmaf);

    const label nNonOrth =
        dict_.getOrDefault<label>("nNonOrthogonalCorrectors", 0);

    for (label nonOrth = 0; nonOrth <= nNonOrth; ++nonOrth)
    {
        fvScalarMatrix phiEEqn
        (
            fvm::laplacian(gammaf, phiE) == -rhoStar
        );
        phiEEqn.solve(mesh_.solver(phiE.name()));
    }

    // Charge consistent with Gauss' law: rhoE - rhoStar = dt div(sigma grad)
    rhoE = -fvc::laplacian(epsf, phiE);
    rhoE.correctBoundaryConditions();

    // Output field, also used by the Korteweg-Helmholtz force
    if (!maxwellStress_ || mesh_.time().writeTime())
    {
        E_.ref() = -fvc::grad(phiE);
    }

    const volScalarField epsC
    (
        mixCell("ehdEpsilonC", eps1_, eps2_, harmonicEps_)
    );

    if (maxwellStress_)
    {
        // T = eps (E E - 1/2 |E|^2 I)
        const volVectorField Ec(faceConsistentE(phiE));
        volSymmTensorField T
        (
            "ehdMaxwellT",
            epsC*(sqr(Ec) - 0.5*magSqr(Ec)*dimensionedSymmTensor(dimless, symmTensor::I))
        );
        force_ = maxwellFaceForce(T);
    }
    else
    {
        // F_f = rhoE_f E_n - 1/2 |E|^2_f snGrad(eps)
        force_ =
            fvc::interpolate(rhoE)*(-fvc::snGrad(phiE))
          - 0.5*fvc::interpolate(magSqr(E_()))*fvc::snGrad(epsC);
    }

    zeroNonCoupled(force_);

    const scalar Qtot = gSum(rhoE.primitiveField()*mesh_.V().field());
    const scalar Qabs = gSum(mag(rhoE.primitiveField())*mesh_.V().field());

    Info<< "EHD DC: total charge = " << Qtot
        << ", total |charge| = " << Qabs << endl;
}


void Foam::ehdModel::correctAC()
{
    volScalarField& pr = phiERe_.ref();
    volScalarField& pi = phiEIm_.ref();

    const surfaceScalarField epsf
    (
        mixFace("ehdEpsilonf", eps1_, eps2_, harmonicEps_)
    );
    const surfaceScalarField sigmaf
    (
        mixFace("ehdSigmaf", sigma1_, sigma2_, harmonicSigma_)
    );
    // omega eps: same units as sigma
    const surfaceScalarField wf("ehdOmegaEpsf", omega_*epsf);

    // Row rotation: phase of the complex diagonal of each row
    scalarField cP(mesh_.nCells(), scalar(1));
    scalarField sP(mesh_.nCells(), Zero);
    {
        tmp<fvScalarMatrix> tLs(fvm::laplacian(sigmaf, pr));
        tmp<fvScalarMatrix> tLw(fvm::laplacian(wf, pr));
        const scalarField& ds = tLs().diag();
        const scalarField& dw = tLw().diag();

        forAll(cP, i)
        {
            const scalar a = -ds[i];
            const scalar b = -dw[i];
            const scalar m = Foam::sqrt(sqr(a) + sqr(b));
            if (m > VSMALL)
            {
                cP[i] = a/m;
                sP[i] = b/m;
            }
        }
    }

    const volScalarField::Internal cosP
    (
        IOobject("ehdCosP", mesh_.time().timeName(), mesh_),
        mesh_,
        dimless,
        cP
    );
    const volScalarField::Internal sinP
    (
        IOobject("ehdSinP", mesh_.time().timeName(), mesh_),
        mesh_,
        dimless,
        sP
    );

    label iter = 0;
    scalar change = GREAT;

    for (iter = 0; iter < nOuterAC_; ++iter)
    {
        const scalarField prevRe(pr.primitiveField());
        const scalarField prevIm(pi.primitiveField());

        // M phiRe = -B phiIm
        {
            const volScalarField::Internal Bi
            (
                sinP*fvc::laplacian(sigmaf, pi)()()
              - cosP*fvc::laplacian(wf, pi)()()
            );

            fvScalarMatrix ReEqn
            (
                cosP*fvm::laplacian(sigmaf, pr)
              + sinP*fvm::laplacian(wf, pr)
              + Bi
            );
            ReEqn.solve(mesh_.solver(pr.name()));
        }

        // M phiIm = +B phiRe
        {
            const volScalarField::Internal Br
            (
                sinP*fvc::laplacian(sigmaf, pr)()()
              - cosP*fvc::laplacian(wf, pr)()()
            );

            fvScalarMatrix ImEqn
            (
                cosP*fvm::laplacian(sigmaf, pi)
              + sinP*fvm::laplacian(wf, pi)
              - Br
            );
            ImEqn.solve(mesh_.solver(pi.name()));
        }

        if (relaxAC_ < 1)
        {
            pr.primitiveFieldRef() =
                relaxAC_*pr.primitiveField() + (1 - relaxAC_)*prevRe;
            pi.primitiveFieldRef() =
                relaxAC_*pi.primitiveField() + (1 - relaxAC_)*prevIm;
            pr.correctBoundaryConditions();
            pi.correctBoundaryConditions();
        }

        const scalar scale =
            max
            (
                gMax(mag(pr.primitiveField())),
                gMax(mag(pi.primitiveField()))
            ) + VSMALL;

        change =
            max
            (
                gMax(mag(pr.primitiveField() - prevRe)),
                gMax(mag(pi.primitiveField() - prevIm))
            )/scale;

        if (change < tolAC_)
        {
            ++iter;
            break;
        }
    }

    Info<< "EHD AC: " << iter << " outer iterations, relative change "
        << change << endl;

    if (change >= tolAC_)
    {
        WarningInFunction
            << "AC potential not converged in " << nOuterAC_
            << " outer iterations (change " << change << ")" << endl;
    }

    // Complex free charge and field: used by the Korteweg-Helmholtz force,
    // otherwise output only
    if (!maxwellStress_ || mesh_.time().writeTime())
    {
        rhoERe_.ref() = -fvc::laplacian(epsf, pr);
        rhoEIm_.ref() = -fvc::laplacian(epsf, pi);
        rhoERe_.ref().correctBoundaryConditions();
        rhoEIm_.ref().correctBoundaryConditions();

        ERe_.ref() = -fvc::grad(pr);
        EIm_.ref() = -fvc::grad(pi);
    }

    const volScalarField epsC
    (
        mixCell("ehdEpsilonC", eps1_, eps2_, harmonicEps_)
    );

    if (maxwellStress_)
    {
        // <T> = 1/2 eps (ERe ERe + EIm EIm) - 1/4 eps (|ERe|^2 + |EIm|^2) I
        const volVectorField Er(faceConsistentE(pr));
        const volVectorField Ei(faceConsistentE(pi));
        volSymmTensorField T
        (
            "ehdMaxwellT",
            epsC
           *(
                0.5*(sqr(Er) + sqr(Ei))
              - 0.25*(magSqr(Er) + magSqr(Ei))
               *dimensionedSymmTensor(dimless, symmTensor::I)
            )
        );
        force_ = maxwellFaceForce(T);
    }
    else
    {
        // Time average: 1/2 Re(rhoHat conj(EHat)) - 1/4 |EHat|^2 grad(eps)
        force_ =
            0.5
           *(
                fvc::interpolate(rhoERe_())*(-fvc::snGrad(pr))
              + fvc::interpolate(rhoEIm_())*(-fvc::snGrad(pi))
            )
          - 0.25
           *fvc::interpolate(magSqr(ERe_()) + magSqr(EIm_()))
           *fvc::snGrad(epsC);
    }

    zeroNonCoupled(force_);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ehdModel::ehdModel
(
    const volScalarField& alpha1,
    const surfaceScalarField& phi
)
:
    mesh_(alpha1.mesh()),
    alpha1_(alpha1),
    phi_(phi),
    dict_
    (
        IOobject
        (
            "electricProperties",
            alpha1.mesh().time().constant(),
            alpha1.mesh(),
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        )
    ),
    type_(modelType::none),
    eps1_("epsilon1", dimPermittivity, Zero),
    eps2_("epsilon2", dimPermittivity, Zero),
    sigma1_("sigma1", dimConductivity, Zero),
    sigma2_("sigma2", dimConductivity, Zero),
    harmonicEps_(true),
    harmonicSigma_(true),
    maxwellStress_(true),
    everyOuterCorrector_(false),
    lastTimeIndex_(-1),
    omega_("omega", dimless/dimTime, Zero),
    nOuterAC_(50),
    tolAC_(1e-8),
    relaxAC_(1),
    force_
    (
        IOobject
        (
            "electricFaceForce",
            alpha1.mesh().time().timeName(),
            alpha1.mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        alpha1.mesh(),
        dimensionedScalar(dimPressure/dimLength, Zero)
    )
{
    force_.setOriented();

    const word typeName = dict_.getOrDefault<word>("ehdModel", "none");

    if (typeName == "none")
    {
        Info<< "EHD: none" << endl;
        return;
    }
    else if (typeName == "DC")
    {
        type_ = modelType::DC;
    }
    else if (typeName == "AC")
    {
        type_ = modelType::AC;
    }
    else
    {
        FatalIOErrorInFunction(dict_)
            << "ehdModel " << typeName << " unknown. Valid: none DC AC"
            << exit(FatalIOError);
    }

    eps1_ = dimensionedScalar("epsilon1", dimPermittivity, dict_);
    eps2_ = dimensionedScalar("epsilon2", dimPermittivity, dict_);
    sigma1_ = dimensionedScalar("sigma1", dimConductivity, dict_);
    sigma2_ = dimensionedScalar("sigma2", dimConductivity, dict_);

    harmonicEps_ = readMixing(dict_, "epsilonMixing");
    harmonicSigma_ = readMixing(dict_, "sigmaMixing");

    everyOuterCorrector_ =
        dict_.getOrDefault<Switch>("correctEveryOuterCorrector", false);

    {
        const word form =
            dict_.getOrDefault<word>("forceForm", "maxwellStress");
        if (form == "maxwellStress")
        {
            maxwellStress_ = true;
        }
        else if (form == "kortewegHelmholtz")
        {
            maxwellStress_ = false;
            if (mag(eps1_.value() - eps2_.value()) > SMALL*mag(eps1_.value()))
            {
                WarningInFunction
                    << "forceForm kortewegHelmholtz with epsilon1 != epsilon2:"
                    << " the interpolated |E|^2 in 1/2 |E|^2 grad(eps)"
                    << " overestimates the normal polarisation stress by"
                    << " (1 + r^2)/(2r), r = eps2/eps1, independent of the"
                    << " mesh. Use forceForm maxwellStress." << endl;
            }
        }
        else
        {
            FatalIOErrorInFunction(dict_)
                << "forceForm " << form << " unknown. Valid: "
                << "kortewegHelmholtz maxwellStress"
                << exit(FatalIOError);
        }
    }

    const Time& runTime = mesh_.time();

    if (type_ == modelType::DC)
    {
        phiE_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "phiE",
                    runTime.timeName(),
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh_
            )
        );

        IOobject rhoIO
        (
            "rhoE",
            runTime.timeName(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        );

        if (rhoIO.typeHeaderOk<volScalarField>(true))
        {
            rhoE_.reset(new volScalarField(rhoIO, mesh_));
        }
        else
        {
            rhoIO.readOpt(IOobject::NO_READ);
            rhoE_.reset
            (
                new volScalarField
                (
                    rhoIO,
                    mesh_,
                    dimensionedScalar(dimChargeDensity, Zero),
                    "zeroGradient"
                )
            );
        }
        rhoE_->oldTime();

        E_.reset
        (
            new volVectorField
            (
                IOobject
                (
                    "E",
                    runTime.timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                -fvc::grad(phiE_())
            )
        );

        Info<< "EHD: leaky dielectric, DC (transient charge)" << nl;
    }
    else
    {
        const dictionary& acDict = dict_.subDict("AC");
        omega_ = dimensionedScalar("omega", dimless/dimTime, acDict);
        nOuterAC_ = acDict.getOrDefault<label>("nOuterCorrectors", 50);
        tolAC_ = acDict.getOrDefault<scalar>("tolerance", 1e-8);
        relaxAC_ = acDict.getOrDefault<scalar>("relaxation", 1);

        if (omega_.value() <= 0)
        {
            FatalIOErrorInFunction(acDict)
                << "omega must be positive" << exit(FatalIOError);
        }

        auto readPhi = [&](const word& name)
        {
            return new volScalarField
            (
                IOobject
                (
                    name,
                    runTime.timeName(),
                    mesh_,
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh_
            );
        };
        phiERe_.reset(readPhi("phiERe"));
        phiEIm_.reset(readPhi("phiEIm"));

        auto newRho = [&](const word& name)
        {
            return new volScalarField
            (
                IOobject
                (
                    name,
                    runTime.timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh_,
                dimensionedScalar(dimChargeDensity, Zero),
                "zeroGradient"
            );
        };
        rhoERe_.reset(newRho("rhoERe"));
        rhoEIm_.reset(newRho("rhoEIm"));

        ERe_.reset
        (
            new volVectorField
            (
                IOobject
                (
                    "ERe",
                    runTime.timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                -fvc::grad(phiERe_())
            )
        );
        EIm_.reset
        (
            new volVectorField
            (
                IOobject
                (
                    "EIm",
                    runTime.timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                -fvc::grad(phiEIm_())
            )
        );

        Info<< "EHD: leaky dielectric, AC (time-averaged), omega = "
            << omega_.value() << nl;
    }

    forceC_.reset
    (
        new volVectorField
        (
            IOobject
            (
                "electricForce",
                runTime.timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedVector(dimPressure/dimLength, Zero),
            "calculated"
        )
    );

    Info<< "    epsilon1 " << eps1_.value() << ", epsilon2 " << eps2_.value()
        << ", sigma1 " << sigma1_.value() << ", sigma2 " << sigma2_.value()
        << nl
        << "    mixing: epsilon "
        << (harmonicEps_ ? "harmonic" : "arithmetic")
        << ", sigma " << (harmonicSigma_ ? "harmonic" : "arithmetic")
        << nl
        << "    force: "
        << (maxwellStress_ ? "Maxwell stress (compact normal jump)"
                           : "Korteweg-Helmholtz")
        << nl
        << "    solved "
        << (everyOuterCorrector_ ? "in every outer corrector" : "once per time step")
        << endl;
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::ehdModel::correct()
{
    if (type_ == modelType::none)
    {
        return;
    }

    // Once per time step unless requested otherwise
    const label timeIndex = mesh_.time().timeIndex();
    if (!everyOuterCorrector_ && timeIndex == lastTimeIndex_)
    {
        return;
    }
    lastTimeIndex_ = timeIndex;

    if (type_ == modelType::DC)
    {
        correctDC();
    }
    else if (type_ == modelType::AC)
    {
        correctAC();
    }
    else
    {
        return;
    }

    // Output-only field
    if (mesh_.time().writeTime())
    {
        forceC_.ref() = fvc::reconstruct(force_*mesh_.magSf());
    }
}


// ************************************************************************* //
