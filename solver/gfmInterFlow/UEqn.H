    MRF.correctBoundaryVelocity(U);

    // Viscous stress -div(tau), tau = mu (grad(U) + grad(U)^T) (laminar).
    // PIMPLE { viscousStress <option>; cStab <c>; }
    //   explicitTranspose   (default) turbulence->divDevRhoReff: implicit
    //                       laplacian(mu, U), explicit div(mu dev2(grad(U)^T)).
    //                       With a viscosity jump the explicit part is
    //                       ~ grad(mu).grad(U)^T and limits the viscous
    //                       Fourier number mu dt/(rho Delta^2) to O(1).
    //   stabilisedTranspose same, plus the implicit-explicit stabiliser
    //                       -laplacian(cStab mu, U)^{n+1} + laplacian(cStab mu, U)^n
    //                       (zero at steady state, O(dt) in time)
    //   noTranspose         drops the transpose term (diagnostic only: wrong
    //                       stress balance at a viscosity jump)
    const word viscousStress
    (
        pimple.dict().getOrDefault<word>("viscousStress", "explicitTranspose")
    );

    tmp<fvVectorMatrix> tdivTau;

    if (viscousStress == "explicitTranspose")
    {
        tdivTau = turbulence->divDevRhoReff(rho, U);
    }
    else if (viscousStress == "stabilisedTranspose")
    {
        const scalar cStab = pimple.dict().getOrDefault<scalar>("cStab", 1.0);
        const volScalarField muS("muS", cStab*rho*turbulence->nuEff());

        tdivTau =
        (
          - fvc::div((rho*turbulence->nuEff())*dev2(T(fvc::grad(U))))
          - fvm::laplacian(rho*turbulence->nuEff(), U)
          - fvm::laplacian(muS, U)
          + fvc::laplacian(muS, U)
        );
    }
    else if (viscousStress == "noTranspose")
    {
        tdivTau = -fvm::laplacian(rho*turbulence->nuEff(), U);
    }
    else
    {
        FatalErrorInFunction
            << "Unknown PIMPLE/viscousStress " << viscousStress
            << ", valid: explicitTranspose stabilisedTranspose noTranspose"
            << exit(FatalError);
    }

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + tdivTau()
     ==
        fvOptions(rho, U)
    );

    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    // CSF or ghost-fluid face force, see interfaceForce.H
                    interfaceForce
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );

        fvOptions.correct(U);
    }
