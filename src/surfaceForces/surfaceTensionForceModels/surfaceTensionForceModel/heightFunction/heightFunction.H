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

Class
    Foam::heightFunction

Description
    Height Function method for the curvature compuation only works on grids
    with cuboid cells

    Axisymmetric (wedge) meshes: detected from the two wedge patches.
    - Interface cells on the axis (prisms, no stencil of their own) whose
      interface is mostly normal to the axis use axial height columns built
      from the stencil of their radial neighbour, with the symmetry
      h(-r) = h(r) across the axis (as Basilisk's symmetric ghost cells).
    - Heights are built from meridional AREA fractions, not from the
      r-weighted volume fractions (bias O(Delta^2/r), blowing up near the
      axis). Area fraction of a cell: first a straight interface with the
      PLIC normal matched to alpha1 in wedge volume; then
      axiFractionIterations (default 3) passes with the parabola given by
      the heights (slope and second derivative), again matched in volume.
    - kappa = kappa_m + kappa_theta, kappa_m from the 2D heights with the
      meridional cell size, and
          kappa_theta = -n_r/r,
      n = grad(alpha)/|grad(alpha)| the normal pointing into phase1, n_r its
      radial component from the height slope, r the radius at which the
      height is defined (column centre for axial columns, interface for
      radial columns).
    - Cells without a valid height: 2D paraboloid fit (interface centres
      mirrored across the axis close to it) and n_r from the PLIC normal.
    - Close to the axis (r < Delta/2 with heights, r < 2 Delta without) the
      symmetry limit kappa_theta = kappa_m is used, since n_r/r with r ~ Delta
      would amplify O(1) normal errors.
    The azimuthal part is written as field kappaTheta at output times.

SourceFiles
    heightFunction.C

\*---------------------------------------------------------------------------*/

#ifndef heightFunction_H
#define heightFunction_H

#include "typeInfo.H"
#include "volFields.H"
#include "dimensionedScalar.H"
#include "autoPtr.H"

#include "surfaceTensionForceModel.H"
#include "HFStencil.H"
#include "markInterfaceRegion.H"
#include "twoDimFDStencil.H"
#include "HashTable.H"
#include "FixedList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                Class heightFunction Declaration
\*---------------------------------------------------------------------------*/

class heightFunction
    : public surfaceTensionForceModel
{

private:

    //-
    //- Stabilisation for normalisation of the interface normal
    typedef HashTable<twoDimFDStencil,Vector2D<label>,Hash<Vector2D<label>>> twoDimStencilMap;

    const dimensionedScalar deltaN_;

    const fvMesh& mesh_;

    HFStencil stencilHF_;

    markInterfaceRegion IFRegion_;

    bool twoDim_;

    // Axisymmetric (wedge) mesh data

        //- Mesh has wedge patches
        bool axi_;

        //- Point on the wedge axis
        point axisPoint_;

        //- Unit vector along the wedge axis
        vector axisDir_;

        //- Wedge width per unit radius, 2 tan(theta/2)
        scalar wedgeWidth_;

        //- Global components of the two geometric (meridional) directions
        FixedList<label, 2> meridionalCmpt_;

        //- cos(theta/2): radius of a wedge-plane point in the centre plane
        scalar cosHalfAngle_;

        //- Field the heights are built from (alpha1 or meridional fraction)
        const volScalarField* alphaHFPtr_;

        //- Print stencil and heights of every height-function cell
        bool debugHF_;

        //- Index (0/1) of the axial direction among the meridional ones
        label axialDir_;

        //- Iterations of the parabolic area-fraction correction
        //  (dictionary keyword axiFractionIterations, default 3)
        label nAxiIter_;

    bool fullColumn(const scalar avgHeight,const scalar tol);

    void nextCell
    (
        const label celli,
        const HFStencil::orientation orientation,
        twoDimFDStencil& HFCol
    );

    void exchangeStencils
    (
        const List<List<twoDimFDStencil>>& sendStencil,
        List<twoDimFDStencil>& recvStencil
    );

    // Private Member Functions

        virtual void correctContactAngle
        (
            surfaceVectorField::Boundary& nHat,
            surfaceVectorField::Boundary& heightFunctionf
        );

        void computeColumns
        (
            const label dir,
            const Map<scalar>& mapAlpha,
            const label celli,
            const HFStencil::orientation orientation,
            twoDimFDStencil& HFCol
        );

        void getStencilValues
        (
            const Map<scalar>& MapAlpha,
            const labelList& stencil,
            DynamicField<scalar>& alphaValues
        );

        scalar calcCurvature
        (
            const scalarField& fit
        );

        // Axisymmetric (wedge) meshes

            //- Find axis and wedge angle from the two wedge patches
            void setWedgeGeometry();

            //- Vector from the axis to p, perpendicular to the axis
            vector radialVector(const point& p) const;

            //- Meridional cell size sqrt(A_m), A_m from V = A_m r_A 2tan(theta/2)
            scalar meridionalDelta(const label celli) const;

            //- Radius of the centroid of the meridional cell section
            scalar meridionalCentre(const label celli) const;

            //- Meridional area fraction of the interface cells: straight
            //  interface with the given normal, positioned so that its
            //  wedge-volume fraction equals alpha1 (other cells: alpha1)
            void meridionalFraction
            (
                const boolList& interfaceCells,
                const volVectorField& normal,
                const volVectorField& gradAlpha,
                volScalarField& alphaM
            ) const;

            //- Meridional rectangle [r0,r1]x[z0,z1] of a wedge cell
            void meridionalRect
            (
                const label celli,
                scalar& r0,
                scalar& r1,
                scalar& z0,
                scalar& z1
            ) const;

            //- Area fraction of a parabolic (height-function) interface
            //  matched to alpha1 in wedge volume
            scalar quadraticAreaFraction
            (
                const label celli,
                const label dir,
                const scalar hx,
                const scalar q,
                const vector& gradAlphai
            ) const;

            //- Height columns from startCell in direction dir, both ends
            //  local; false if no height was found
            bool localColumns
            (
                const label startCell,
                const label dir,
                const Map<scalar>& MapAlpha,
                twoDimFDStencil& cols
            );

            //- Radial neighbour of an axis cell whose interface is mostly
            //  normal to the axis (-1 otherwise)
            label axisNeighbour(const label celli, const vector& nHat);

            //- Interface normal from the height slope hx (cell units)
            vector heightNormal
            (
                const label dir,
                const scalar hx,
                const vector& gradAlphai
            ) const;

            //- Azimuthal curvature -n_r/r, kappaM for r < rAxis
            scalar azimuthalCurvature
            (
                const scalar r,
                const vector& er,
                const scalar magNr,
                const vector& gradAlphai,
                const scalar kappaM,
                const scalar rAxis
            ) const;

            //- kappa_theta of a cell with a valid height column
            scalar azimuthalCurvatureHF
            (
                const label celli,
                const point& centre,
                const vector& normal,
                twoDimFDStencil& cols,
                const vector& gradAlphai,
                const scalar kappaM,
                const scalar deltaX
            ) const;

            //- kappa_theta of a fallback (paraboloid) cell
            scalar azimuthalCurvaturePLIC
            (
                const point& centre,
                const vector& normal,
                const vector& gradAlphai,
                const scalar kappaM,
                const scalar deltaX
            ) const;

        //- Disallow copy construct
        heightFunction(const heightFunction&);

        //- Disallow default bitwise assignment
        void operator=(const heightFunction&);

public:

    //- Runtime type information
    TypeName("heightFunction");


    // Constructors

        //- Construct from components
        heightFunction
        (
            const dictionary& dict,
            const volScalarField& alpha1,
            const surfaceScalarField& phi,
            const volVectorField& U
        );

    //- Destructor
    virtual ~heightFunction()
    {}

    virtual void correct();


};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
