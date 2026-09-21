#include "symGaussSeidelSmoother.H"

namespace Foam
{
    defineTypeNameAndDebug(symGaussSeidelSmoother, 0);

    lduMatrix::smoother::addsymMatrixConstructorToTable<symGaussSeidelSmoother>
        addsymGaussSeidelSmootherSymMatrixConstructorToTable_;

    lduMatrix::smoother::addasymMatrixConstructorToTable<symGaussSeidelSmoother>
        addsymGaussSeidelSmootherAsymMatrixConstructorToTable_;
}


Foam::symGaussSeidelSmoother::symGaussSeidelSmoother
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces
)
:
    lduMatrix::smoother
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces
    )
{}


void Foam::symGaussSeidelSmoother::smooth
(
    const word& fieldName_,
    scalarField& psi,
    const lduMatrix& matrix_,
    const scalarField& source,
    const FieldField<Field, scalar>& interfaceBouCoeffs_,
    const lduInterfaceFieldPtrsList& interfaces_,
    const direction cmpt,
    const label nSweeps
)
{
    scalar* __restrict__ psiPtr = psi.begin();
    const label nCells = psi.size();
    scalarField bPrime(nCells);
    scalar* __restrict__ bPrimePtr = bPrime.begin();

    const scalar* const __restrict__ diagPtr = matrix_.diag().begin();
    const scalar* const __restrict__ upperPtr = matrix_.upper().begin();
    const scalar* const __restrict__ lowerPtr = matrix_.lower().begin();
    const label* const __restrict__ uPtr =
        matrix_.lduAddr().upperAddr().begin();
    const label* const __restrict__ ownStartPtr =
        matrix_.lduAddr().ownerStartAddr().begin();

    // Retain the OpenFOAM-14 coupled-interface handling exactly. The local
    // test currently supplies empty interface lists as its MPI placeholder.
    FieldField<Field, scalar>& mBouCoeffs =
        const_cast<FieldField<Field, scalar>&>(interfaceBouCoeffs_);
    forAll(mBouCoeffs, patchi)
    {
        if (interfaces_.set(patchi)) mBouCoeffs[patchi].negate();
    }

    for (label sweep=0; sweep<nSweeps; ++sweep)
    {
        bPrime = source;
        matrix_.initMatrixInterfaces
        (
            mBouCoeffs, interfaces_, psi, bPrime, cmpt
        );
        matrix_.updateMatrixInterfaces
        (
            mBouCoeffs, interfaces_, psi, bPrime, cmpt
        );

        scalar psii;
        label fStart;
        label fEnd = ownStartPtr[0];

        for (label celli=0; celli<nCells; ++celli)
        {
            fStart = fEnd;
            fEnd = ownStartPtr[celli + 1];
            psii = bPrimePtr[celli];
            for (label facei=fStart; facei<fEnd; ++facei)
                psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
            psii /= diagPtr[celli];
            for (label facei=fStart; facei<fEnd; ++facei)
                bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
            psiPtr[celli] = psii;
        }

        fStart = ownStartPtr[nCells];
        for (label celli=nCells - 1; celli>=0; --celli)
        {
            fEnd = fStart;
            fStart = ownStartPtr[celli];
            psii = bPrimePtr[celli];
            for (label facei=fStart; facei<fEnd; ++facei)
                psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
            psii /= diagPtr[celli];
            for (label facei=fStart; facei<fEnd; ++facei)
                bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
            psiPtr[celli] = psii;
        }
    }

    forAll(mBouCoeffs, patchi)
    {
        if (interfaces_.set(patchi)) mBouCoeffs[patchi].negate();
    }
}


void Foam::symGaussSeidelSmoother::smooth
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt,
    const label nSweeps
) const
{
    smooth
    (
        fieldName_, psi, matrix_, source,
        interfaceBouCoeffs_, interfaces_, cmpt, nSweeps
    );
}
