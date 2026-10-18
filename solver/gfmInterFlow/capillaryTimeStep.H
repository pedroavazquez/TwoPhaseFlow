/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Global
    capillaryTimeStep

Description
    Capillary time-step limit with inertial and viscous branches
    (Galusinski & Vigneaux, J. Comput. Phys. 227, 2008), used when
    system/controlDict contains

        capillaryTimeStep
        {
            cRho    1;      // inertial (Brackbill) branch
            cMu     0.5;    // viscous-capillary branch
            muMode  min;    // min (default) | sum
        }

    For every interface cell (1e-3 < alpha1 < 1 - 1e-3)

        tauRho = sqrt((rho1 + rho2) Delta^3 / (2 pi sigma))
        tauMu  = muCap Delta / sigma,   muCap = min(mu1, mu2) (muMode min)
                                         or mu1 + mu2   (muMode sum)
        a = cMu tauMu,  b = cRho tauRho
        dtCap = (a + sqrt(a^2 + 4 b^2))/2

    dtCap -> b for a -> 0 (inviscid, Brackbill with cRho = 1, same
    definition as surfaceForces::capillaryDt), dtCap -> a for b -> 0
    (Stokes), and max(a, b) <= dtCap <= a + b. The limit is the minimum
    over interface cells and capillaryNum = deltaT/dtCap is then bounded by
    maxCapillaryNum in setDeltaT.H, exactly as before.

    Delta is the smallest centre-to-centre distance across the internal and
    coupled faces of the cell. Unlike fvc::average(mag(mesh.delta())), it
    ignores wedge faces, whose patch-normal delta r sin(theta/2) goes to
    zero at the axis and made the Brackbill limit too small on wedge meshes.

    Input:  mesh, alpha1, mixture, rho1, rho2, surfForces, runTime
    Output: capillaryNum

\*---------------------------------------------------------------------------*/

{
    const dictionary& capDict = *capDictPtr;

    const scalar cRho = capDict.getOrDefault<scalar>("cRho", 1.0);
    const scalar cMu = capDict.getOrDefault<scalar>("cMu", 0.5);

    // min: the less viscous phase controls the grid-scale capillary modes
    // (tests/capillaryDtAxi: mu1/mu2 = 0.01 is unstable at dt = 0.2 tauRho,
    // mu1 = mu2 is stable at 0.8 tauRho); sum: Galusinski-Vigneaux as is
    const word muMode = capDict.getOrDefault<word>("muMode", "min");
    if (muMode != "min" && muMode != "sum")
    {
        FatalIOErrorInFunction(capDict)
            << "muMode must be min or sum, not " << muMode
            << exit(FatalIOError);
    }
    const bool muMin = (muMode == "min");

    // Cell size: smallest centre-to-centre distance over internal and
    // coupled (processor, cyclic) faces. Wedge/empty/walls are ignored.
    const vectorField& Cc = mesh.cellCentres();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();

    scalarField Delta(mesh.nCells(), GREAT);

    forAll(own, facei)
    {
        const scalar d = mag(Cc[nei[facei]] - Cc[own[facei]]);
        Delta[own[facei]] = min(Delta[own[facei]], d);
        Delta[nei[facei]] = min(Delta[nei[facei]], d);
    }

    forAll(mesh.boundary(), patchi)
    {
        const fvPatch& pp = mesh.boundary()[patchi];

        if (pp.coupled())
        {
            // coupled delta = full vector from this cell to the neighbour
            const vectorField pd(pp.delta());
            const labelUList& fc = pp.faceCells();

            forAll(fc, i)
            {
                Delta[fc[i]] = min(Delta[fc[i]], mag(pd[i]));
            }
        }
    }

    // Phase properties (cell values, so non-Newtonian models also work)
    const scalarField mu1
    (
        rho1.value()*mixture.nuModel1().nu()().primitiveField()
    );
    const scalarField mu2
    (
        rho2.value()*mixture.nuModel2().nu()().primitiveField()
    );
    const tmp<volScalarField> tsigmaCap = surfForces.sigma();
    const scalarField& sigmaCap = tsigmaCap().primitiveField();

    const scalarField& a1 = alpha1.primitiveField();
    const scalar rhoSum = rho1.value() + rho2.value();
    const scalar twoPi = constant::mathematical::twoPi;

    scalar dtCap = GREAT;
    scalar dtRho = GREAT;
    scalar dtMu = GREAT;

    forAll(a1, celli)
    {
        if (a1[celli] > 1e-3 && a1[celli] < 1 - 1e-3 && Delta[celli] < GREAT)
        {
            const scalar d = Delta[celli];
            const scalar s = max(sigmaCap[celli], SMALL);

            const scalar b = cRho*Foam::sqrt(rhoSum*pow3(d)/(twoPi*s));
            const scalar muCap =
                muMin
              ? min(mu1[celli], mu2[celli])
              : mu1[celli] + mu2[celli];
            const scalar a = cMu*muCap*d/s;

            dtCap = min(dtCap, 0.5*(a + Foam::sqrt(sqr(a) + 4*sqr(b))));
            dtRho = min(dtRho, b);
            dtMu = min(dtMu, a);
        }
    }

    reduce(dtCap, minOp<scalar>());
    reduce(dtRho, minOp<scalar>());
    reduce(dtMu, minOp<scalar>());

    capillaryNum = runTime.deltaTValue()/dtCap;

    if (dtCap < GREAT)
    {
        Info<< "Capillary time step: " << dtCap
            << " (inertial cRho*tauRho " << dtRho
            << ", viscous cMu*tauMu " << dtMu << ")" << endl;
    }
}

// ************************************************************************* //
