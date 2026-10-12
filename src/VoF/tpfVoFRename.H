/*---------------------------------------------------------------------------*\
    tpfVoFRename.H

    Forced into every TwoPhaseFlow compilation unit through
        EXE_INC = -include <path>/tpfVoFRename.H ...
    in Make/options (see tpf_vof_rename.sh).

    TwoPhaseFlow's src/VoF is a modified copy of OpenFOAM-ESI's geometricVoF.
    Both libraries export the same symbols, and when a run-time loaded ESI
    library (e.g. libfieldFunctionObjects.so) pulls in libgeometricVoF.so,
    the static objects of both are constructed in, and destroyed from, the
    same storage: heap corruption ("corrupted double-linked list" at exit).

    Renaming the shared class names (and the namespace 'reconstruction' of
    plicRDF, isoAlpha, gradAlpha, isoSurface) gives TwoPhaseFlow its own
    symbols. TypeName strings, hence dictionary keywords, are unchanged.
\*---------------------------------------------------------------------------*/

#ifndef tpfVoFRename_H
#define tpfVoFRename_H

#define cutCell                         tpfCutCell
#define cutCellIso                      tpfCutCellIso
#define cutCellPLIC                     tpfCutCellPLIC
#define cutFace                         tpfCutFace
#define cutFaceIso                      tpfCutFaceIso
#define cutFacePLIC                     tpfCutFacePLIC
#define cutFaceAdvect                   tpfCutFaceAdvect
#define surfaceIteratorIso              tpfSurfaceIteratorIso
#define surfaceIteratorPLIC             tpfSurfaceIteratorPLIC
#define reconstructionSchemes           tpfReconstructionSchemes
#define reconstructedDistanceFunction   tpfReconstructedDistanceFunction

// Namespace of plicRDF, isoAlpha, gradAlpha (and isoSurface)
#define reconstruction                  tpfReconstruction

#endif

// ************************************************************************* //
