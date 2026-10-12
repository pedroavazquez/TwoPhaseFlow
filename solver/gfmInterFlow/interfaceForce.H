/*---------------------------------------------------------------------------*\
  gfmInterFlow - ghost-fluid extension of TwoPhaseFlow's interFlow
-------------------------------------------------------------------------------
License
    GNU General Public License v3 or later (same as OpenFOAM / TwoPhaseFlow)

File
    interfaceForce.H

Description
    Face force (pressure-gradient units, kg m^-2 s^-2) entering UEqn.H and
    pEqn.H in place of
        surfForces.surfaceTensionForce() + surfForces.accelerationForce()

    PIMPLE { ghostFluid off; }  (default) -> CSF, identical to interFlow:
        F_f = sigma_f*Kf*snGrad(alpha1) - ghf*snGrad(rho)

    PIMPLE { ghostFluid on; }   -> ghost fluid / sharp surface force:
        F_f = deltaCoeff_f*(H_N - H_P)*( sigma*kappa_G - (rho1 - rho2)*gh_G )

    with
        H       = Heaviside(alpha1 - 0.5)
        theta   = (alpha_P - 0.5)/(alpha_P - alpha_N), position of the
                  alpha1 = 0.5 crossing along the line P->N
        (.)_G   = (1 - theta)*(.)_P + theta*(.)_N

    This is the jump [p_rgh]_PN = (H_N - H_P)(sigma*kappa - [rho] gh)
    moved to the right-hand side of the face pressure gradient. The
    curvature is the one computed by the selected TwoPhaseFlow
    surfaceTensionForceModel (registered on the mesh as "K_"), so all
    curvature models (constantCurvature, gradAlpha, RDF, fitParaboloid,
    heightFunction) can be combined with either force discretisation.

    The force is a face-normal quantity (its sign flips with the face
    orientation, like snGrad), so the field is marked oriented; otherwise
    ESI OpenFOAM refuses phiHbyA += phig in pEqn.H.

    Non-coupled patches carry no force (no contact-line treatment).
    The face delta coefficient is mesh.nonOrthDeltaCoeffs(), i.e. the one of
    the implicit part of laplacian(rAUf, p_rgh).

\*---------------------------------------------------------------------------*/

const bool useGFM =
    pimple.dict().getOrDefault<Switch>("ghostFluid", false);

surfaceScalarField interfaceForce
(
    IOobject
    (
        "interfaceForce",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimPressure/dimLength, Zero)
);

// Face-normal quantity: same orientation semantics as snGrad and fluxes
interfaceForce.setOriented();

if (!useGFM)
{
    interfaceForce =
        surfForces.surfaceTensionForce()
      + surfForces.accelerationForce();
}
else
{
    const scalar alphaC = 0.5;

    const volScalarField& Kc = mesh.lookupObject<volScalarField>("K_");

    volScalarField sigmaK("sigmaK", surfForces.sigma()*Kc);
    sigmaK.correctBoundaryConditions();  // exchange on coupled patches

    const scalar drho = (rho1 - rho2).value();

    // Same orthogonal coefficient as the implicit part of laplacian(rAUf, p_rgh)
    const surfaceScalarField& dC = mesh.nonOrthDeltaCoeffs();

    auto faceForce = [&]
    (
        const scalar aP, const scalar aN,
        const scalar sKP, const scalar sKN,
        const scalar ghP, const scalar ghN,
        const scalar delta
    ) -> scalar
    {
        const scalar HP = pos0(aP - alphaC);
        const scalar HN = pos0(aN - alphaC);

        if (HP == HN)
        {
            return 0;
        }

        // aP != aN guaranteed when HP != HN
        const scalar theta =
            min(max((aP - alphaC)/(aP - aN), scalar(0)), scalar(1));

        const scalar sigmaKG = (1 - theta)*sKP + theta*sKN;
        const scalar ghG     = (1 - theta)*ghP + theta*ghN;

        return delta*(HN - HP)*(sigmaKG - drho*ghG);
    };

    // Internal faces
    {
        const labelUList& own = mesh.owner();
        const labelUList& nei = mesh.neighbour();
        scalarField& fI = interfaceForce.primitiveFieldRef();

        forAll(own, facei)
        {
            const label P = own[facei];
            const label N = nei[facei];

            fI[facei] = faceForce
            (
                alpha1[P], alpha1[N],
                sigmaK[P], sigmaK[N],
                gh[P], gh[N],
                dC[facei]
            );
        }
    }

    // Coupled boundaries (processor, cyclic); other patches carry no force
    auto& fBf = interfaceForce.boundaryFieldRef();

    forAll(fBf, patchi)
    {
        const fvPatchScalarField& aPf = alpha1.boundaryField()[patchi];

        if (!aPf.coupled())
        {
            continue;
        }

        const scalarField aP(aPf.patchInternalField());
        const scalarField aN(aPf.patchNeighbourField());

        const fvPatchScalarField& sKpf = sigmaK.boundaryField()[patchi];
        const scalarField sKP(sKpf.patchInternalField());
        const scalarField sKN(sKpf.patchNeighbourField());

        const fvPatchScalarField& ghpf = gh.boundaryField()[patchi];
        const scalarField ghP(ghpf.patchInternalField());
        const scalarField ghN(ghpf.patchNeighbourField());

        const scalarField& dCp = dC.boundaryField()[patchi];
        fvsPatchScalarField& fp = fBf[patchi];

        forAll(fp, i)
        {
            fp[i] = faceForce
            (
                aP[i], aN[i], sKP[i], sKN[i], ghP[i], ghN[i], dCp[i]
            );
        }
    }
}

// ************************************************************************* //