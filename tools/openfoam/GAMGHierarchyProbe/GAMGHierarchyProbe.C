/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | www.openfoam.com
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2026 Dmitry Mishura
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

Application
    GAMGHierarchyProbe

Description
    Construct the WavefrontGaussSeidel synthetic symmetric LDU matrix on an
    undecomposed fvMesh, invoke the stock OpenFOAM v2606 algebraicPair GAMG
    agglomerator, print hierarchy sizes, and exit without solving.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "GAMGAgglomeration.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::noParallel();

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const lduAddressing& addressing = mesh.lduAddr();
    const labelUList& owners = addressing.lowerAddr();
    const labelUList& neighbours = addressing.upperAddr();
    const label nCells = addressing.size();
    const label nFaces = owners.size();

    if (neighbours.size() != owners.size())
    {
        FatalErrorInFunction
            << "LDU owner/neighbour addressing size mismatch"
            << exit(FatalError);
    }

    lduMatrix matrix(mesh);
    scalarField& diagonal = matrix.diag(nCells);
    scalarField& upper = matrix.upper(nFaces);
    diagonal = Zero;

    for (label face=0; face<nFaces; ++face)
    {
        const label owner = owners[face];
        const label neighbour = neighbours[face];
        const scalar coefficient =
            1.0 + 0.001*((17*owner + 13*neighbour) % 101);

        upper[face] = -coefficient;
        diagonal[owner] += coefficient;
        diagonal[neighbour] += coefficient;
    }

    scalar diagonalSum = 0;
    for (label cell=0; cell<nCells; ++cell)
    {
        diagonalSum += diagonal[cell];
    }
    const scalar meanDiagonal = diagonalSum/scalar(nCells);
    for (label cell=0; cell<nCells; ++cell)
    {
        diagonal[cell] += 0.25*meanDiagonal;
    }

    dictionary controls;
    controls.add("agglomerator", word("algebraicPair"));
    controls.add("nCellsInCoarsestLevel", label(10));
    controls.add("mergeLevels", label(1));
    controls.add("renumber", false);

    Info<< "GAMGHierarchyProbe" << nl
        << "meshCells=" << nCells << nl
        << "meshFaces=" << nFaces << nl
        << "agglomerator=algebraicPair" << nl
        << "nCellsInCoarsestLevel=10" << nl
        << "mergeLevels=1" << nl
        << "renumber=false" << nl
        << "maxLevels=50" << nl
        << "parallel=false" << nl
        << "matrix=symmetric-upper-storage" << nl
        << "weight=abs(upper)" << nl
        << endl;

    const GAMGAgglomeration& hierarchy =
        GAMGAgglomeration::New(matrix, controls);

    const label totalLevels = hierarchy.size() + 1;
    for (label level=0; level<totalLevels; ++level)
    {
        const lduAddressing& levelAddressing =
            hierarchy.meshLevel(level).lduAddr();

        Info<< "level " << level
            << " cells=" << levelAddressing.size()
            << " faces=" << levelAddressing.upperAddr().size()
            << nl;
    }

    Info<< nl << "levels=" << totalLevels << nl << endl;
    Info<< "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
