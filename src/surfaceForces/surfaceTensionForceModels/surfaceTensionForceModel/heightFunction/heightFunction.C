/*---------------------------------------------------------------------------*\
            Copyright (c) 2017-2019, German Aerospace Center (DLR)
-------------------------------------------------------------------------------
License
    This file is part of the VoFLibrary source code library, which is an
	unofficial extension to OpenFOAM.
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

\*---------------------------------------------------------------------------*/

#include "heightFunction.H"
#include "addToRunTimeSelectionTable.H"

#include "alphaContactAngleTwoPhaseFvPatchScalarField.H"
#include "mathematicalConstants.H"
#include "surfaceInterpolate.H"
#include "fvcDiv.H"
#include "fvcGrad.H"
#include "reconstructionSchemes.H"
#include "SortableList.H"
#include "leastSquareFitParabolid.H"
#include "wedgePolyPatch.H"
#include "unitConversion.H"
#include "vector2D.H"


// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(heightFunction, 0);
    addToRunTimeSelectionTable(surfaceTensionForceModel,heightFunction, components);
}



void Foam::heightFunction::getStencilValues
(
    const Map<scalar>& MapAlpha,
    const labelList& stencil,
    DynamicField<scalar>& alphaValues
)
{
    alphaValues.clear();
    zoneDistribute& zDist = zoneDistribute::New(mesh_);
    forAll(stencil,i)
    {
        const label& gblIdx = stencil[i];
        alphaValues.append(zDist.getValue(*alphaHFPtr_,MapAlpha,gblIdx));
    }
}

bool Foam::heightFunction::fullColumn(const scalar avgHeight,const scalar tol)
{
    return (avgHeight >= (1-tol) || avgHeight < tol);
}

void Foam::heightFunction::nextCell
(
    const label celli,
    const HFStencil::orientation orientation,
    twoDimFDStencil& HFCol
)
{
    label idx = stencilHF_.getCellLabel(celli,HFCol.nextCell(orientation));
    HFCol.status[orientation].gblIdx = idx;

}

void Foam::heightFunction::computeColumns
(
    const label dir,
    const Map<scalar>& mapAlpha,
    const label celli,
    const HFStencil::orientation orientation,
    twoDimFDStencil& HFCol
)
{
    const labelListList& stencil = stencilHF_.getStencil();
    const globalIndex& globalIdx = zoneDistribute::New(mesh_).globalNumbering();

    label& nextPos = HFCol.status[orientation].gblIdx;
    scalar columnHeightTol = 1e-6;

    nextCell
    (
        celli,
        orientation,
        HFCol
    );

    if (!globalIdx.isLocal(nextPos))
    {
        return;
    }

    DynamicField<scalar > alphaValues(100); // should be big enough avoids resizing

    for (label i = HFCol.status[orientation].iterI;i<7;i++) // move four times in both direction
    {
        if (globalIdx.isLocal(nextPos))
        {
            label localIdx = globalIdx.toLocal(nextPos);
            if (localIdx >= mesh_.nCells() || stencil[localIdx].size() == 0 )
            {
                // boundary face cannot compute height
                HFCol.status[orientation].iterI = i;
                HFCol.status[orientation].avgColVal = 0.5;
                return;
            }

            getStencilValues(mapAlpha,stencil[localIdx],alphaValues);

            HFCol.status[orientation].avgColVal = HFCol.addColumnHeight(alphaValues);

            if (!fullColumn(HFCol.status[orientation].avgColVal ,columnHeightTol))
            {
                HFCol.status[orientation].iterI = i;

                nextCell
                (
                    localIdx,
                    orientation,
                    HFCol
                );
            }
        }
		else
		{
			return;
		}

        if (fullColumn(HFCol.status[orientation].avgColVal,columnHeightTol) )
        {
            return;
        }

    }
}

void Foam::heightFunction::exchangeStencils
(
    const List<List<twoDimFDStencil>>& sendStencil,
    List<twoDimFDStencil>& recvStencil
)
{
    PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);

    // Stream data into buffer
    for (label domain = 0; domain < Pstream::nProcs(); domain++)
    {
        if (domain != Pstream::myProcNo())
        {
            // Put data into send buffer
            UOPstream toDomain(domain, pBufs);

            toDomain << sendStencil[domain];
        }
    }

    // wait until everything is written.
    pBufs.finishedSends();

    for (label domain = 0; domain < Pstream::nProcs(); domain++)
    {
        if (domain != Pstream::myProcNo())
        {
            List<twoDimFDStencil> stencilFromDomain;
            // get data from send buffer
            UIPstream fromDomain(domain, pBufs);

            fromDomain >> stencilFromDomain;
            recvStencil.append(stencilFromDomain);

        }
    }
}

Foam::scalar Foam::heightFunction::calcCurvature
(
    const scalarField& fit
)
{
    if (fit.size() == 2)
    {
        return (2*fit[1])/pow(1+sqr(fit[0]),1.5);
    }
    else
    {
        // c0*x + c1*y  + c2*x^2 + c3*y^2  + c4*x*y
        // curvature at x=0, y=0
        // 1st derivative = c1, c2
        // 2nd derivative = 2*c3, 2*c4, c5
        // equation: Jibben et al, A Paraboloid Fitting Technique for
        // Calculating Curvature from Piecewise-Linear
        // Interface Reconstructions on 3D
        // Unstructured Meshes
        return (2*fit[2] + 2*fit[3] + 2*fit[2]*sqr(fit[1]) + 2*fit[3]*sqr(fit[0]) - 2*fit[4]*fit[0]*fit[1])/pow(1+sqr(fit[0])+sqr(fit[1]),1.5);
    }
}

void Foam::heightFunction::setWedgeGeometry()
{
    const polyBoundaryMesh& pbm = mesh_.boundaryMesh();

    DynamicList<label> wedgeIDs;
    forAll(pbm, patchi)
    {
        if (isA<wedgePolyPatch>(pbm[patchi]))
        {
            wedgeIDs.append(patchi);
        }
    }

    if (wedgeIDs.empty())
    {
        axi_ = false;
        return;
    }

    if (wedgeIDs.size() != 2)
    {
        FatalErrorInFunction
            << "Axisymmetric height function: expected exactly 2 wedge "
            << "patches, found " << wedgeIDs.size()
            << exit(FatalError);
    }

    // Area vector and area-weighted centre of both (planar) wedge patches
    vector n[2];
    scalar d[2];
    for (label k = 0; k < 2; k++)
    {
        const polyPatch& pp = pbm[wedgeIDs[k]];
        const vectorField Sf(pp.faceAreas());
        const vectorField Cf(pp.faceCentres());
        const scalarField magSf(mag(Sf));

        const vector S = gSum(Sf);
        const scalar A = gSum(magSf);
        const vector AC = gSum(magSf*Cf);

        if (mag(S) < VSMALL || A < VSMALL)
        {
            FatalErrorInFunction
                << "Wedge patch " << pp.name() << " has zero area"
                << exit(FatalError);
        }

        n[k] = S/mag(S);
        d[k] = n[k] & (AC/A);
    }

    // Outward normals of the two wedge planes: n0 & n1 = -cos(theta)
    const scalar c = n[0] & n[1];
    const scalar theta = acos(min(max(-c, scalar(-1)), scalar(1)));

    axisDir_ = n[0] ^ n[1];
    if (mag(axisDir_) < SMALL)
    {
        FatalErrorInFunction
            << "Wedge patches are parallel, cannot determine the axis"
            << exit(FatalError);
    }
    axisDir_ /= mag(axisDir_);

    // Point on the intersection line of the two wedge planes
    const scalar det = 1 - sqr(c);
    axisPoint_ = ((d[0] - d[1]*c)*n[0] + (d[1] - d[0]*c)*n[1])/det;

    wedgeWidth_ = 2*tan(0.5*theta);
    cosHalfAngle_ = cos(0.5*theta);

    // The two geometric (meridional) directions, as used by the stencils
    label k = 0;
    forAll(mesh_.geometricD(), cmpt)
    {
        if (mesh_.geometricD()[cmpt] != -1 && k < 2)
        {
            meridionalCmpt_[k++] = cmpt;
        }
    }

    if (k != 2)
    {
        FatalErrorInFunction
            << "Axisymmetric height function: expected two geometric "
            << "directions, geometricD = " << mesh_.geometricD()
            << exit(FatalError);
    }

    axialDir_ =
        (mag(axisDir_[meridionalCmpt_[0]]) > mag(axisDir_[meridionalCmpt_[1]]))
      ? 0 : 1;

    axi_ = true;

    Info<< "heightFunction: axisymmetric (wedge) mesh" << nl
        << "    axis point     " << axisPoint_ << nl
        << "    axis direction " << axisDir_ << nl
        << "    wedge angle    " << radToDeg(theta) << " deg" << endl;
}


Foam::vector Foam::heightFunction::radialVector(const point& p) const
{
    const vector dp = p - axisPoint_;
    return dp - (dp & axisDir_)*axisDir_;
}


Foam::scalar Foam::heightFunction::meridionalCentre(const label celli) const
{
    // V = A_m r_A 2tan(theta/2), r_A = area centroid of the meridional
    // section. For a square section r_A = r_c - A_m/(12 r_c), with r_c the
    // cell centroid radius (one correction step is sufficient).
    const scalar V = mesh_.V()[celli];
    const scalar rc = max(mag(radialVector(mesh_.C()[celli])), VSMALL);
    const scalar A0 = V/(wedgeWidth_*rc);

    return max(rc - A0/(12*rc), 0.5*rc);
}


Foam::scalar Foam::heightFunction::meridionalDelta(const label celli) const
{
    return sqrt(mesh_.V()[celli]/(wedgeWidth_*meridionalCentre(celli)));
}


void Foam::heightFunction::meridionalFraction
(
    const boolList& interfaceCells,
    const volVectorField& normal,
    const volVectorField& gradAlpha,
    volScalarField& alphaM
) const
{
    // For every interface cell: meridional section = rectangle in (r, z);
    // a straight interface with the given normal is positioned such that
    // the r-weighted (wedge volume) fraction equals alpha1, then the
    // fraction of the rectangle AREA on the phase-1 side is returned.
    // Only the normal is taken from outside, the position comes from alpha1.
    const pointField& pts = mesh_.points();
    const labelListList& cellPts = mesh_.cellPoints();
    scalarField& aM = alphaM.primitiveFieldRef();

    DynamicList<vector2D> poly(8);

    forAll(interfaceCells, celli)
    {
        if (!interfaceCells[celli] || mag(normal[celli]) < VSMALL)
        {
            continue;
        }

        scalar r0 = GREAT, r1 = -GREAT, z0 = GREAT, z1 = -GREAT;
        for (const label pti : cellPts[celli])
        {
            const vector dp = pts[pti] - axisPoint_;
            const scalar z = dp & axisDir_;
            const scalar r = mag(dp - z*axisDir_)*cosHalfAngle_;
            r0 = min(r0, r);
            r1 = max(r1, r);
            z0 = min(z0, z);
            z1 = max(z1, z);
        }

        if ((r1 - r0)*(z1 - z0) < VSMALL)
        {
            continue;
        }

        const vector rvC = radialVector(mesh_.C()[celli]);
        if (mag(rvC) < VSMALL)
        {
            continue;
        }
        const vector er = rvC/mag(rvC);

        vector nIn = normal[celli];
        if ((nIn & gradAlpha[celli]) < 0)
        {
            nIn = -nIn;
        }
        const scalar nr = nIn & er;
        const scalar nz = nIn & axisDir_;
        if (sqr(nr) + sqr(nz) < VSMALL)
        {
            continue;
        }

        const vector2D rect[4] =
        {
            vector2D(r0, z0),
            vector2D(r1, z0),
            vector2D(r1, z1),
            vector2D(r0, z1)
        };

        // Area and first r-moment of the part with nr r + nz z >= c
        auto clipped = [&](const scalar c, scalar& A, scalar& Mr)
        {
            poly.clear();
            for (label i = 0; i < 4; i++)
            {
                const vector2D& pa = rect[i];
                const vector2D& pb = rect[(i + 1) % 4];
                const scalar fa = nr*pa.x() + nz*pa.y() - c;
                const scalar fb = nr*pb.x() + nz*pb.y() - c;
                if (fa >= 0)
                {
                    poly.append(pa);
                }
                if ((fa >= 0) != (fb >= 0))
                {
                    poly.append(pa + (fa/(fa - fb))*(pb - pa));
                }
            }
            A = 0;
            Mr = 0;
            forAll(poly, i)
            {
                const vector2D& pa = poly[i];
                const vector2D& pb = poly[(i + 1) % poly.size()];
                const scalar cr = pa.x()*pb.y() - pb.x()*pa.y();
                A += cr;
                Mr += (pa.x() + pb.x())*cr;
            }
            A = mag(0.5*A);
            Mr = mag(Mr/6.0);
        };

        scalar Arect, Mrect;
        scalar cLo = GREAT, cHi = -GREAT;
        for (const vector2D& p : rect)
        {
            const scalar f = nr*p.x() + nz*p.y();
            cLo = min(cLo, f);
            cHi = max(cHi, f);
        }
        clipped(cLo - 1, Arect, Mrect);
        if (Mrect < VSMALL)
        {
            continue;
        }

        // bisection: r-weighted fraction of {f >= c} decreases with c
        const scalar target = alpha1_[celli];
        scalar A = 0, Mr = 0;
        for (label iter = 0; iter < 60; iter++)
        {
            const scalar c = 0.5*(cLo + cHi);
            clipped(c, A, Mr);
            if (Mr/Mrect > target)
            {
                cLo = c;
            }
            else
            {
                cHi = c;
            }
        }
        clipped(0.5*(cLo + cHi), A, Mr);

        aM[celli] = min(max(A/Arect, scalar(0)), scalar(1));
    }

    alphaM.correctBoundaryConditions();
}


void Foam::heightFunction::meridionalRect
(
    const label celli,
    scalar& r0,
    scalar& r1,
    scalar& z0,
    scalar& z1
) const
{
    const pointField& pts = mesh_.points();
    r0 = GREAT; r1 = -GREAT; z0 = GREAT; z1 = -GREAT;
    for (const label pti : mesh_.cellPoints()[celli])
    {
        const vector dp = pts[pti] - axisPoint_;
        const scalar z = dp & axisDir_;
        const scalar r = mag(dp - z*axisDir_)*cosHalfAngle_;
        r0 = min(r0, r);
        r1 = max(r1, r);
        z0 = min(z0, z);
        z1 = max(z1, z);
    }
}


Foam::scalar Foam::heightFunction::quadraticAreaFraction
(
    const label celli,
    const label dir,
    const scalar hx,
    const scalar q,
    const vector& gradAlphai
) const
{
    // Interface inside the cell modelled by the height-function parabola
    // s_d(u) = c + slope u + q u^2/2 along the column direction, positioned
    // so that its wedge-volume (r-weighted) fraction equals alpha1. Returns
    // the meridional AREA fraction. hx, q: height slope (cell units) and
    // second derivative (1/length) along the global tangential coordinate.
    scalar r0, r1, z0, z1;
    meridionalRect(celli, r0, r1, z0, z1);
    if ((r1 - r0)*(z1 - z0) < VSMALL)
    {
        return alpha1_[celli];
    }

    const vector rvC = radialVector(mesh_.C()[celli]);
    if (mag(rvC) < VSMALL)
    {
        return alpha1_[celli];
    }
    const vector er = rvC/mag(rvC);

    const label cmptD = meridionalCmpt_[dir];
    const label cmptT = meridionalCmpt_[1 - dir];

    vector ed(Zero);
    ed[cmptD] = 1;
    // +1: phase 1 at low global s_d
    const scalar sigma = ((gradAlphai & ed) < 0) ? 1.0 : -1.0;

    const bool axialColumns = (dir == axialDir_);
    const scalar target = alpha1_[celli];
    const label nq = 200;

    // fraction (volume, area) for interface offset c
    auto fractions = [&](const scalar c, scalar& fv, scalar& fa)
    {
        scalar sv = 0, sw = 0, sa = 0;
        if (axialColumns)
        {
            // z_int(r) = c + sz (r - rm) + qz (r - rm)^2/2
            const scalar sAx = (axisDir_[cmptD] > 0) ? 1.0 : -1.0;
            const scalar sEr = (er[cmptT] > 0) ? 1.0 : -1.0;
            const scalar sz = sAx*sigma*hx*sEr;
            const scalar qz = sAx*sigma*q;
            const bool below = ((gradAlphai & axisDir_) < 0);
            const scalar rm = 0.5*(r0 + r1);
            for (label k = 0; k < nq; k++)
            {
                const scalar r = r0 + (r1 - r0)*(k + 0.5)/nq;
                const scalar u = r - rm;
                const scalar zi = c + sz*u + 0.5*qz*sqr(u);
                scalar L = min(max((zi - z0)/(z1 - z0), scalar(0)), scalar(1));
                if (!below)
                {
                    L = 1 - L;
                }
                sv += L*r;
                sw += r;
                sa += L;
            }
            fv = sv/sw;
            fa = sa/nq;
        }
        else
        {
            // r_int(z) = c + sr (z - zm) + qr (z - zm)^2/2
            const scalar sEr = (er[cmptD] > 0) ? 1.0 : -1.0;
            const scalar sAxT = (axisDir_[cmptT] > 0) ? 1.0 : -1.0;
            const scalar sr = sEr*sigma*hx*sAxT;
            const scalar qr = sEr*sigma*q;
            const bool inside = ((gradAlphai & er) < 0);
            const scalar zm = 0.5*(z0 + z1);
            const scalar Vr = sqr(r1) - sqr(r0);
            for (label k = 0; k < nq; k++)
            {
                const scalar z = z0 + (z1 - z0)*(k + 0.5)/nq;
                const scalar u = z - zm;
                const scalar ri =
                    min(max(c + sr*u + 0.5*qr*sqr(u), r0), r1);
                scalar Lv = (sqr(ri) - sqr(r0))/Vr;
                scalar La = (ri - r0)/(r1 - r0);
                if (!inside)
                {
                    Lv = 1 - Lv;
                    La = 1 - La;
                }
                sv += Lv;
                sa += La;
            }
            fv = sv/nq;
            fa = sa/nq;
        }
    };

    // the volume fraction is monotonic in c; bracket generously
    const scalar span = 2*((r1 - r0) + (z1 - z0));
    scalar cLo = (axialColumns ? z0 : r0) - span;
    scalar cHi = (axialColumns ? z1 : r1) + span;
    scalar fvLo, faLo, fvHi, faHi;
    fractions(cLo, fvLo, faLo);
    fractions(cHi, fvHi, faHi);
    const bool increasing = (fvHi > fvLo);

    scalar fv = 0, fa = target;
    for (label iter = 0; iter < 60; iter++)
    {
        const scalar c = 0.5*(cLo + cHi);
        fractions(c, fv, fa);
        if ((fv < target) == increasing)
        {
            cLo = c;
        }
        else
        {
            cHi = c;
        }
    }
    fractions(0.5*(cLo + cHi), fv, fa);

    return min(max(fa, scalar(0)), scalar(1));
}


bool Foam::heightFunction::localColumns
(
    const label startCell,
    const label dir,
    const Map<scalar>& MapAlpha,
    twoDimFDStencil& cols
)
{
    const labelListList& stencil = stencilHF_.getStencil();
    const globalIndex& gi = zoneDistribute::New(mesh_).globalNumbering();

    if (stencil[startCell].size() == 0)
    {
        return false;
    }

    cols.status.first().iterI = 1;
    cols.status.second().iterI = 1;

    DynamicField<scalar> vals(100);
    getStencilValues(MapAlpha, stencil[startCell], vals);
    cols.addColumnHeight(vals);

    computeColumns(dir, MapAlpha, startCell, HFStencil::orientation::pos, cols);
    computeColumns(dir, MapAlpha, startCell, HFStencil::orientation::neg, cols);

    return
        cols.foundHeight()
     && gi.isLocal(cols.status[0].gblIdx)
     && gi.isLocal(cols.status[1].gblIdx);
}


Foam::label Foam::heightFunction::axisNeighbour
(
    const label celli,
    const vector& nHat
)
{
    // Interface cell on the axis (prism) with an interface mostly normal to
    // the axis: return its radial (outer) face neighbour if that one has a
    // height-function stencil, else -1
    const vector rvC = radialVector(mesh_.C()[celli]);
    const scalar deltaM = meridionalDelta(celli);
    if (mag(rvC) > deltaM || mag(rvC) < VSMALL)
    {
        return -1;
    }
    const vector er = rvC/mag(rvC);

    if (mag(nHat & axisDir_) < mag(nHat & er))
    {
        return -1;
    }

    const labelUList& own = mesh_.owner();
    const labelUList& nei = mesh_.neighbour();

    label n1 = -1;
    scalar best = 0;
    for (const label facei : mesh_.cells()[celli])
    {
        if (!mesh_.isInternalFace(facei))
        {
            continue;
        }
        const label other = (own[facei] == celli) ? nei[facei] : own[facei];
        const vector d = mesh_.C()[other] - mesh_.C()[celli];
        const scalar dr = d & er;
        if (dr > best && mag(d & axisDir_) < 0.5*deltaM)
        {
            best = dr;
            n1 = other;
        }
    }

    if (n1 < 0 || !stencilHF_.isCuboid()[n1])
    {
        return -1;
    }

    return n1;
}


Foam::vector Foam::heightFunction::heightNormal
(
    const label dir,
    const scalar hx,
    const vector& gradAlphai
) const
{
    // Heights count phase-1 cells along e_d; the interface is
    // s_d = s_start + h (phase 1 at low s_d) or s_end - h (phase 1 at high
    // s_d), so its slope ds_d/ds_t is +hx or -hx. Normal = e_d - slope e_t
    // (orientation is fixed later with grad(alpha)).
    vector ed(Zero);
    vector et(Zero);
    ed[meridionalCmpt_[dir]] = 1;
    et[meridionalCmpt_[1 - dir]] = 1;

    const scalar slope = ((gradAlphai & ed) < 0) ? hx : -hx;

    return ed - slope*et;
}


Foam::scalar Foam::heightFunction::azimuthalCurvature
(
    const scalar r,
    const vector& er,
    const scalar magNr,
    const vector& gradAlphai,
    const scalar kappaM,
    const scalar rAxis
) const
{
    // Close to the axis: symmetry limit kappa_theta = kappa_m
    if (r < rAxis)
    {
        return kappaM;
    }

    // n = grad(alpha)/|grad(alpha)| points into phase1, kappa = -div(n)
    const scalar sgn = ((gradAlphai & er) >= 0) ? 1.0 : -1.0;

    return -sgn*magNr/r;
}


Foam::scalar Foam::heightFunction::azimuthalCurvatureHF
(
    const label celli,
    const point& centre,
    const vector& normal,
    twoDimFDStencil& cols,
    const vector& gradAlphai,
    const scalar kappaM,
    const scalar deltaX
) const
{
    const vector rvI = radialVector(centre);
    scalar rI = mag(rvI);

    const vector rvC = radialVector(mesh_.C()[celli]);
    if (mag(rvC) < VSMALL)
    {
        return kappaM;
    }
    const vector er = rvC/mag(rvC);

    // Interface s_d = h(s_t) in cell units: normal ~ (e_d - h' e_t)
    const scalarField& h = cols.heights();
    const scalar hx = 0.5*(h[2] - h[0]);

    const label dir = cols.direction();
    const scalar erD = mag(er[meridionalCmpt_[dir]]);
    const scalar erT = mag(er[meridionalCmpt_[1 - dir]]);

    const scalar magNr = (erD + mag(hx)*erT)/sqrt(1 + sqr(hx));

    // Radius where the height (and its slope) is defined:
    // - axial columns: column (cell) centre radius
    // - radial columns: interface radius at the axial position of the
    //   column centre, r(z_c) = r_I - n_z (z_c - z_I)/n_r from the PLIC line
    //   (the PLIC centroid itself can be up to Delta/2 away along the
    //   interface, which made kappa_theta first order)
    scalar r = meridionalCentre(celli);
    if (erD >= erT)
    {
        r = rI;
        if (mag(normal) > VSMALL)
        {
            const vector nHat = normal/mag(normal);
            const scalar nr = nHat & er;
            const scalar nz = nHat & axisDir_;
            const scalar dz = (mesh_.C()[celli] - centre) & axisDir_;
            if (mag(nr) > mag(nz))
            {
                r = max(rI - nz*dz/nr, SMALL);
            }
        }
    }

    return azimuthalCurvature(r, er, magNr, gradAlphai, kappaM, 0.5*deltaX);
}


Foam::scalar Foam::heightFunction::azimuthalCurvaturePLIC
(
    const point& centre,
    const vector& normal,
    const vector& gradAlphai,
    const scalar kappaM,
    const scalar deltaX
) const
{
    const vector rv = radialVector(centre);
    const scalar r = mag(rv);

    if (r < 2*deltaX || mag(normal) < VSMALL)
    {
        return kappaM;
    }

    const vector er = rv/r;
    const scalar magNr = mag((normal/mag(normal)) & er);

    return azimuthalCurvature(r, er, magNr, gradAlphai, kappaM, 2*deltaX);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::heightFunction::heightFunction
(
    const dictionary& dict,
    const volScalarField& alpha1,
    const surfaceScalarField& phi,
    const volVectorField& U
)
:
    surfaceTensionForceModel
    (
        typeName,
        dict,
        alpha1,
        phi,
        U
    ),
    deltaN_
    (
        "deltaN",
        1e-8/pow(average(alpha1.mesh().V()), 1.0/3.0)
    ),
    mesh_(alpha1.mesh()),
    stencilHF_(alpha1.mesh(),dict.lookupOrDefault<scalar>("angleTol",0.001)),
    IFRegion_(alpha1.mesh()),
    twoDim_(false),
    axi_(false),
    axisPoint_(Zero),
    axisDir_(Zero),
    wedgeWidth_(0),
    meridionalCmpt_(-1),
    cosHalfAngle_(1),
    alphaHFPtr_(&alpha1),
    debugHF_(dict.getOrDefault<bool>("debugHF", false)),
    axialDir_(-1),
    nAxiIter_(dict.getOrDefault<label>("axiFractionIterations", 3))

{
    label dimensions = 0;
    for (direction cmpt = 0; cmpt < vector::nComponents; cmpt++)
    {
        dimensions += pos0(mesh_.geometricD()[cmpt]) * mesh_.geometricD()[cmpt];
    }

    twoDim_ = (dimensions == 2);

    if (dimensions <= 1)
    {
        FatalErrorInFunction
        << "The Height Function Method only works on grids with atleast " << nl
        << "two dimensions" << nl
        << endl
        << exit(FatalError);
    }

    setWedgeGeometry();
}

// * * * * * * * * * * * * * * Public Access Member Functions  * * * * * * * * * * * * * * //

void Foam::heightFunction::correctContactAngle
(
    surfaceVectorField::Boundary& nHatb,
    surfaceVectorField::Boundary& heightFunctionf
)
{
    scalar convertToRad = Foam::constant::mathematical::pi/180.0;

    const fvMesh& mesh = alpha1_.mesh();
    const volScalarField::Boundary& abf = alpha1_.boundaryField();

    const fvBoundaryMesh& boundary = mesh.boundary();

    forAll(boundary, patchi)
    {
        if (isA<alphaContactAngleTwoPhaseFvPatchScalarField>(abf[patchi]))
        {
            alphaContactAngleTwoPhaseFvPatchScalarField& acap =
                const_cast<alphaContactAngleTwoPhaseFvPatchScalarField&>
                (
                    refCast<const alphaContactAngleTwoPhaseFvPatchScalarField>
                    (
                        abf[patchi]
                    )
                );

            fvsPatchVectorField& nHatp = nHatb[patchi];
            const scalarField theta
            (
                convertToRad*acap.theta(U_.boundaryField()[patchi], nHatp)
            );

            const vectorField nf
            (
                boundary[patchi].nf()
            );

            // Reset nHatp to correspond to the contact angle

            const scalarField a12(nHatp & nf);
            const scalarField b1(cos(theta));

            scalarField b2(nHatp.size());
            forAll(b2, facei)
            {
                b2[facei] = cos(acos(a12[facei]) - theta[facei]);
            }

            const scalarField det(1.0 - a12*a12);

            scalarField a((b1 - a12*b2)/det);
            scalarField b((b2 - a12*b1)/det);

            nHatp = a*nf + b*nHatp;
            nHatp /= (mag(nHatp) + deltaN_.value());

            acap.gradient() = (nf & nHatp)*mag(heightFunctionf[patchi]);
            acap.evaluate();
        }
    }
}


void Foam::heightFunction::correct()
{
    deltaFunctionModel_->correct();

    const fvMesh& mesh = alpha1_.mesh();

    reconstructionSchemes& surf =
        mesh.lookupObjectRef<reconstructionSchemes>("reconstructionScheme");

    // can also be an isosurface
    surf.reconstruct(false);

    const boolList& interfaceCells = surf.interfaceCell();
    const volVectorField& faceCentre = surf.centre();
    const volVectorField& faceNormal = surf.normal();

    // Axisymmetric meshes: sign of the radial component of the normal
    tmp<volVectorField> tgradAlpha;
    if (axi_)
    {
        tgradAlpha = fvc::grad(alpha1_);
    }

    const globalIndex& globalIdx = zoneDistribute::New(mesh_).globalNumbering();

    boolList nextToInterface(mesh.nCells(),false);

    volScalarField cellDistField
    (
        IOobject
        (
            "cellDistField",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("scalar", dimless, 0),
        "calculated"   //alpha1.boundaryField().types()
    );

    IFRegion_.markCellsNearSurf(interfaceCells,5,nextToInterface,cellDistField);

    if (mesh_.time().outputTime())
    {
        cellDistField.write();
    }
    // updateStencil;
    stencilHF_.updateStencil(nextToInterface);

    const boolList& isCuboid = stencilHF_.isCuboid();

    volScalarField isCuboidField
    (
        IOobject
        (
            "isCuboidField",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("scalar", dimless, 0),
        "calculated"   //alpha1.boundaryField().types()
    );

    forAll(isCuboid,i)
    {
        if (isCuboid[i])
        {
            isCuboidField[i] = 1;
        }
        else
        {
            isCuboidField[i] = 0;
        }
    }

    if (mesh_.time().outputTime())
    {
        isCuboidField.write();
    }

    // Field the heights are built from: on wedge meshes the meridional
    // area fraction, elsewhere alpha1
    volScalarField alphaM("alphaMeridional", alpha1_);
    alphaHFPtr_ = &alpha1_;
    if (axi_)
    {
        // Pass 1: area fractions with the PLIC normal
        meridionalFraction(interfaceCells, faceNormal, tgradAlpha(), alphaM);
        alphaHFPtr_ = &alphaM;

        // Iterations: heights from the current area fractions give, per
        // cell, the column direction, slope and second derivative; the area
        // fractions are then recomputed with this parabolic interface
        // (matched to alpha1 in volume). The straight PLIC line is only
        // first order and misses the O(kappa Delta) curvature term of the
        // volume -> area conversion, which does not vanish next to the axis.
        const boolList& isCub = stencilHF_.isCuboid();
        const labelListList& st = stencilHF_.getStencil();

        for (label it = 0; it < nAxiIter_; it++)
        {
            const Map<scalar> MapAlphaIt =
                stencilHF_.getDatafromOtherProc(nextToInterface, alphaM);

            labelList hfDir(mesh.nCells(), -1);
            scalarField hfHx(mesh.nCells(), Zero);
            scalarField hfQ(mesh.nCells(), Zero);

            forAll(interfaceCells, celli)
            {
                if (!interfaceCells[celli] || mag(faceNormal[celli]) < VSMALL)
                {
                    continue;
                }
                const vector nP = faceNormal[celli]/mag(faceNormal[celli]);

                if (isCub[celli] && st[celli].size())
                {
                    const label d0 =
                        (
                            mag(nP[meridionalCmpt_[0]])
                         >= mag(nP[meridionalCmpt_[1]])
                        ) ? 0 : 1;

                    for (label k = 0; k < 2; k++)
                    {
                        const label d = (k == 0) ? d0 : 1 - d0;
                        twoDimFDStencil cols(twoDim_, d, globalIdx.toGlobal(celli));
                        if (localColumns(celli, d, MapAlphaIt, cols))
                        {
                            const scalarField& h = cols.heights();
                            hfDir[celli] = d;
                            hfHx[celli] = 0.5*(h[2] - h[0]);
                            hfQ[celli] =
                                (h[2] - 2*h[1] + h[0])/meridionalDelta(celli);
                            break;
                        }
                    }
                }
                else
                {
                    // axis cell: mirrored heights along the global
                    // tangential coordinate
                    const label n1 = axisNeighbour(celli, nP);
                    if (n1 < 0)
                    {
                        continue;
                    }
                    twoDimFDStencil cols(twoDim_, axialDir_, globalIdx.toGlobal(n1));
                    if (localColumns(n1, axialDir_, MapAlphaIt, cols))
                    {
                        const scalarField& h = cols.heights();
                        const label radCmpt = meridionalCmpt_[1 - axialDir_];
                        const bool outerPositive =
                            (radialVector(mesh.C()[n1])[radCmpt] > 0);
                        const scalar hRow0 = outerPositive ? h[0] : h[2];
                        const scalar hRow1 = h[1];
                        hfDir[celli] = axialDir_;
                        hfHx[celli] =
                            outerPositive
                          ? 0.5*(hRow1 - hRow0)
                          : 0.5*(hRow0 - hRow1);
                        hfQ[celli] = (hRow1 - hRow0)/meridionalDelta(n1);
                    }
                }
            }

            scalarField& aM = alphaM.primitiveFieldRef();
            forAll(hfDir, celli)
            {
                if (hfDir[celli] >= 0)
                {
                    aM[celli] = quadraticAreaFraction
                    (
                        celli,
                        hfDir[celli],
                        hfHx[celli],
                        hfQ[celli],
                        tgradAlpha()[celli]
                    );
                }
            }
            alphaM.correctBoundaryConditions();
        }
    }

    volScalarField kappaTheta
    (
        IOobject
        (
            "kappaTheta",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless/dimLength, Zero),
        "calculated"
    );

    Map<scalar> MapAlpha = stencilHF_.getDatafromOtherProc
    (
        nextToInterface,
        *alphaHFPtr_
    );

    volScalarField foundHeightField
    (
        IOobject
        (
            "foundHeightField",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("scalar", dimless, 0),
        "calculated"   //alpha1.boundaryField().types()
    );
    Vector<label> geomDir = mesh_.geometricD();
    label dirs = twoDim_ ? 2 : 3;

    scalar thickness = 1.0;
    if (twoDim_)
    {
        forAll(geomDir, iDir)
        {
            if (geomDir[iDir] == -1)
            {
                thickness = mesh.bounds().span()[iDir];
                break;
            }
        }
    }

    twoDimStencilMap parallelStencil;
    List<List<twoDimFDStencil>> sendStencil(Pstream::nProcs());

    DynamicField<scalar > alphaValues(100); // should be big enough avoids resizing

    const labelListList& stencil = stencilHF_.getStencil();

    forAll(interfaceCells,celli)
    {
        vector n = faceNormal[celli];
        if (interfaceCells[celli] && mag(n) != 0)
        {
            if (isCuboid[celli])
            {
                n /= mag(n);

                SortableList<scalar> sortedDirs(0);
                for (int dirI=0;dirI<3;dirI++)
                {
                    if (geomDir[dirI] != -1)
                        sortedDirs.append(mag(n[dirI]));
                }
                sortedDirs.reverseSort();
                foundHeightField[celli] = 0;

                for (int dirI=0;dirI<dirs;dirI++)
                {
                    label direction = sortedDirs.indices()[dirI];
                    twoDimFDStencil cols(twoDim_,direction,globalIdx.toGlobal(celli));
                    cols.status.first().iterI = 1;
                    cols.status.second().iterI = 1;

                    getStencilValues(MapAlpha,stencil[celli],alphaValues);
                    cols.addColumnHeight(alphaValues);

                    computeColumns
                    (
                        direction,
                        MapAlpha,
                        celli,
                        HFStencil::orientation::pos,
                        cols
                    );

                    computeColumns
                    (
                        direction,
                        MapAlpha,
                        celli,
                        HFStencil::orientation::neg,
                        cols
                    );

                    if (cols.foundHeight())
                    {
                        foundHeightField[celli] = 1;
                        scalar deltaX = twoDim_? sqrt(mesh.V()[celli]/thickness) : cbrt(mesh.V()[celli]);
                        if (axi_)
                        {
                            deltaX = meridionalDelta(celli);
                        }
                        K_[celli] = cols.calcCurvature(deltaX);

                        if (axi_)
                        {
                            kappaTheta[celli] = azimuthalCurvatureHF
                            (
                                celli,
                                faceCentre[celli],
                                faceNormal[celli],
                                cols,
                                tgradAlpha()[celli],
                                K_[celli],
                                deltaX
                            );
                            K_[celli] += kappaTheta[celli];
                        }

                        if (debugHF_)
                        {
                            // Stencil slots (i,j) -> i + 3j in 2D
                            const labelList& st = stencil[celli];
                            Pout<< "HFdebug cell " << celli
                                << " C " << mesh.C()[celli]
                                << " dir " << direction
                                << " deltaX " << deltaX
                                << " h " << cols.heights()
                                << " K " << K_[celli]
                                << " kTheta " << kappaTheta[celli]
                                << " iter " << cols.status[0].iterI
                                << ' ' << cols.status[1].iterI
                                << nl << "    slots";
                            forAll(st, sloti)
                            {
                                const label g = st[sloti];
                                if
                                (
                                    g >= 0 && globalIdx.isLocal(g)
                                 && globalIdx.toLocal(g) < mesh.nCells()
                                )
                                {
                                    const label l = globalIdx.toLocal(g);
                                    const vector dc =
                                        (mesh.C()[l] - mesh.C()[celli])/deltaX;
                                    Pout<< ' ' << sloti << ':' << l
                                        << '(' << dc.x() << ',' << dc.y()
                                        << ',' << dc.z() << ")a="
                                        << (*alphaHFPtr_)[l];
                                }
                                else
                                {
                                    Pout<< ' ' << sloti << ":remote" << g;
                                }
                            }
                            Pout<< endl;
                        }

                        break;
                    }

                    if (!globalIdx.isLocal(cols.status[0].gblIdx))
                    {
                        parallelStencil.insert
                        (
                            Vector2D<label>(celli,direction),
                            cols
                        );
                        label procI = globalIdx.whichProcID(cols.status[0].gblIdx);
                        sendStencil[procI].append(cols);

						break;
                    }
                    if (!globalIdx.isLocal(cols.status[1].gblIdx))
                    {
                        parallelStencil.insert
                        (
                            Vector2D<label>(celli,direction),
                            cols
                        );
                        label procI = globalIdx.whichProcID(cols.status[1].gblIdx);
                        sendStencil[procI].append(cols);

						break;
                    }

                }
            }
            else
            {
                // fitParaboloid
            }
        }
        else
        {
            // fitParaboloid
            K_[celli] = 0;
        }

    }

    List<twoDimFDStencil> recvStencil;
    exchangeStencils(sendStencil,recvStencil);

    sendStencil.clear();
    sendStencil.resize(Pstream::nProcs());

    forAll(recvStencil,i)
    {
        twoDimFDStencil& HFcols = recvStencil[i];
        HFcols.heights() = 0.0;

        if (globalIdx.isLocal(HFcols.status[0].gblIdx))
        {
            const label localIdx = globalIdx.toLocal(HFcols.status[0].gblIdx);
            HFcols.status[1].gblIdx = -1;

            getStencilValues(MapAlpha,stencil[localIdx],alphaValues);
            HFcols.status[0].avgColVal = HFcols.addColumnHeight(alphaValues);

            if (!fullColumn(HFcols.status[0].avgColVal, 1e-6))
            {
                // moves straight to nextPos.
                computeColumns
                (
                    HFcols.direction(),
                    MapAlpha,
                    localIdx,
                    HFStencil::orientation::pos,
                    HFcols
                );
            }

            if (fullColumn(HFcols.status[0].avgColVal, 1e-6))
            {
                label origProc = globalIdx.whichProcID(HFcols.gblcelli());
                sendStencil[origProc].append(HFcols);
            }
        }

        if (globalIdx.isLocal(HFcols.status[1].gblIdx))
        {
            const label localIdx = globalIdx.toLocal(HFcols.status[1].gblIdx);
            HFcols.status[0].gblIdx = -1;

            getStencilValues(MapAlpha,stencil[localIdx],alphaValues);
            HFcols.status[1].avgColVal = HFcols.addColumnHeight(alphaValues);

            if (!fullColumn(HFcols.status[1].avgColVal, 1e-6))
            {
                computeColumns
                (
                    HFcols.direction(),
                    MapAlpha,
                    localIdx,
                    HFStencil::orientation::neg,
                    HFcols
                );
            }

            if (fullColumn(HFcols.status[1].avgColVal, 1e-6))
            {
                label origProc = globalIdx.whichProcID(HFcols.gblcelli());
                sendStencil[origProc].append(HFcols);
            }
        }

        // send back otherProc
    }

    List<twoDimFDStencil> sendBackToOriginalProcess;
    exchangeStencils(sendStencil,sendBackToOriginalProcess);

    forAll(sendBackToOriginalProcess,i)
    {
        twoDimFDStencil& neiCols = sendBackToOriginalProcess[i];
        if (neiCols.status[0].gblIdx != -1)
        {
            label celli = globalIdx.toLocal(neiCols.gblcelli());
            Vector2D<label> celliAndDir(celli,neiCols.direction());
            if (parallelStencil.found(celliAndDir))
            {
                parallelStencil[celliAndDir].status[0] = neiCols.status[0];
                parallelStencil[celliAndDir].heights() += neiCols.heights();
            }
        }
        if (neiCols.status[1].gblIdx != -1)
        {
            label celli = globalIdx.toLocal(neiCols.gblcelli());
            Vector2D<label> celliAndDir(celli,neiCols.direction());
            if (parallelStencil.found(celliAndDir))
            {
                parallelStencil[celliAndDir].status[1] = neiCols.status[1];
                parallelStencil[celliAndDir].heights() += neiCols.heights();
            }
        }
    }

    forAllIters(parallelStencil,iter)
    {
        if (iter().foundHeight())
        {
            label celli = globalIdx.toLocal(iter().gblcelli());
            foundHeightField[celli] = 1;

            scalar deltaX = twoDim_? sqrt(mesh.V()[celli]/thickness) : cbrt(mesh.V()[celli]);
            if (axi_)
            {
                deltaX = meridionalDelta(celli);
            }
            K_[celli] = iter().calcCurvature(deltaX);

            if (axi_)
            {
                kappaTheta[celli] = azimuthalCurvatureHF
                (
                    celli,
                    faceCentre[celli],
                    faceNormal[celli],
                    iter(),
                    tgradAlpha()[celli],
                    K_[celli],
                    deltaX
                );
                K_[celli] += kappaTheta[celli];
            }
        }

    }

    // Axisymmetric: interface cells on the axis (prisms without a stencil)
    // with an interface mostly normal to the axis (drop poles). Build the
    // axial height columns from the stencil of the radial neighbour (row 1),
    // whose stencil holds row 0 (axis), row 1 and row 2; then use the
    // symmetry h(-r) = h(r): heights (h_0, h_0, h_1) centred on row 0.
    if (axi_)
    {
        forAll(interfaceCells, celli)
        {
            if
            (
                !interfaceCells[celli]
             || foundHeightField[celli] == 1
             || mag(faceNormal[celli]) < VSMALL
            )
            {
                continue;
            }

            const label n1 =
                axisNeighbour(celli, faceNormal[celli]/mag(faceNormal[celli]));
            if (n1 < 0)
            {
                continue;
            }

            twoDimFDStencil cols(twoDim_, axialDir_, globalIdx.toGlobal(n1));
            if (!localColumns(n1, axialDir_, MapAlpha, cols))
            {
                continue;
            }

            // which end of the stencil (slot 0 or 2) is the axis row
            const vector er1 = radialVector(mesh.C()[n1]);
            const label radCmpt = meridionalCmpt_[1 - axialDir_];
            const scalarField& h = cols.heights();
            const scalar h0 = (er1[radCmpt] > 0) ? h[0] : h[2];
            const scalar h1 = h[1];

            // heights at r = -Delta/2 (mirror), Delta/2, 3 Delta/2
            scalarField hMirror(3);
            hMirror[0] = h0;
            hMirror[1] = h0;
            hMirror[2] = h1;
            twoDimFDStencil axisCols(twoDim_, hMirror, axialDir_, globalIdx.toGlobal(celli));

            const scalar dX = meridionalDelta(n1);
            const scalar kappaM = axisCols.calcCurvature(dX);

            // kappa_theta = n_r/r at the column centre r = Delta/2
            const scalar hx = 0.5*(h1 - h0);
            const scalar magNr = mag(hx)/sqrt(1 + sqr(hx));
            // on the axis kappa_theta -> kappa_m (symmetry), same sign
            K_[celli] = kappaM;
            kappaTheta[celli] = sign(kappaM)*magNr/(0.5*dX);
            K_[celli] += kappaTheta[celli];
            foundHeightField[celli] = 1;

            if (debugHF_)
            {
                Pout<< "HFdebug axis cell " << celli << " via " << n1
                    << " h " << h << " h0 " << h0 << " h1 " << h1
                    << " K " << K_[celli] << " kTheta " << kappaTheta[celli]
                    << endl;
            }
        }
    }

    // fitParaboloid

    Vector<label> explicitDim(1,1,-1);

    label nDims = 0;

    forAll(mesh.geometricD(),i)
    {
        if (mesh.geometricD()[i] == 1)
        {
            nDims++;
        }
    }

    if (nDims == 2)
    {
        explicitDim.y() = -1;
    }

    leastSquareFitParabolid paraboloid(geomDir,explicitDim);

    zoneDistribute& zDist = zoneDistribute::New(mesh_);

    zDist.setUpCommforZone(interfaceCells);

    Map<Field <vector > > mapCentres = zDist.getFields(interfaceCells,faceCentre);
    Map<Field <vector > > mapNormal = zDist.getFields(interfaceCells,faceNormal);

    forAll(interfaceCells,cellI)
    {
        if (interfaceCells[cellI] && foundHeightField[cellI] == 0)
        {
            if (mag(faceNormal[cellI]) == 0)
            {
                K_[cellI] = 0;
                continue;
            }
            vector n = faceNormal[cellI]/mag(faceNormal[cellI]);
            point c = faceCentre[cellI];

            const vectorField& neiNormal =  mapNormal[cellI];
            const vectorField& neiCentre =  mapCentres[cellI];

            DynamicField< vector > centres(neiCentre[cellI].size());
            DynamicField< scalar > weight(neiNormal[cellI].size());

            forAll(neiNormal,i)
            {
                if (mag(neiNormal[i]) != 0)
                {
                    centres.append(neiCentre[i]);
                    weight.append(pow(mag(neiNormal[i]),0.25));
                }
            }

            // Axisymmetric: close to the axis, add the interface centres
            // mirrored across the axis (the meridian continues smoothly)
            scalar deltaXAxi = 0;
            if (axi_)
            {
                deltaXAxi = meridionalDelta(cellI);

                if (mag(radialVector(c)) < 3*deltaXAxi)
                {
                    const label nOrig = centres.size();
                    for (label i = 0; i < nOrig; i++)
                    {
                        const vector ci = centres[i];
                        const scalar wi = weight[i];
                        const vector rv = radialVector(ci);

                        if (mag(rv) > 1e-3*deltaXAxi)
                        {
                            centres.append(ci - 2*rv);
                            weight.append(wi);
                        }
                    }
                }
            }

            if (centres.size() >= paraboloid.nCoeffs())
            {
                K_[cellI] =  calcCurvature(paraboloid.fitParaboloid(c,n,centres,weight));

                if (axi_)
                {
                    kappaTheta[cellI] = azimuthalCurvaturePLIC
                    (
                        c,
                        n,
                        tgradAlpha()[cellI],
                        K_[cellI],
                        deltaXAxi
                    );
                    K_[cellI] += kappaTheta[cellI];
                }
            }
            else
            {
                K_[cellI] = 0;
            }
        }
    }

    if (mesh_.time().outputTime())
    {
        foundHeightField.write();
        if (axi_)
        {
            kappaTheta.write();
            alphaM.write();
        }
    }

    // alphaM goes out of scope at the end of correct()
    alphaHFPtr_ = &alpha1_;
    nextToInterface = false;

    IFRegion_.markCellsNearSurf(interfaceCells,1,nextToInterface,cellDistField);

    zDist.setUpCommforZone(nextToInterface);

    Map<Field<scalar>> mapCurv = zDist.getFields(nextToInterface,K_);
    mapCentres = zDist.getFields(nextToInterface,faceCentre);

    forAll(nextToInterface,celli)
    {
        if (nextToInterface[celli] && !interfaceCells[celli])
        {
            const point cc = mesh.C()[celli];
            scalar smallDist = GREAT;
            label smallestDistIdx = -1;
            Field < vector> centreField = mapCentres[celli];
            forAll(centreField,i)
            {
                if (centreField[i] != vector::zero && i > 0)
                {
                    scalar dist = mag(cc-centreField[i]);
                    if (smallDist > dist)
                    {
                        smallDist = dist;
                        smallestDistIdx = i;
                    }
                }
            }
            K_[celli] = mapCurv[celli][smallestDistIdx];
        }
    }
    K_.correctBoundaryConditions();

    Kf_ = fvc::interpolate(K_);

}



// ************************************************************************* //
