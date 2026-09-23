#ifndef Foam_IndexKahnDirectSmoother_H
#define Foam_IndexKahnDirectSmoother_H

#include "lduMatrix.H"

namespace Foam
{

struct IndexKahnSchedule;

class IndexKahnDirectSmoother
:
    public lduMatrix::smoother
{
    label width_;
    label minCells_;
    bool useIndexKahn_;
    const IndexKahnSchedule* schedule_;

    void smoothInternal
    (
        solveScalarField& psi,
        const solveScalarField& source,
        direction cmpt,
        label nSweeps
    ) const;

public:

    TypeName("indexKahnDirect");

    IndexKahnDirectSmoother
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    virtual void smooth
    (
        solveScalarField& psi,
        const scalarField& source,
        direction cmpt,
        label nSweeps
    ) const;

    virtual void scalarSmooth
    (
        solveScalarField& psi,
        const solveScalarField& source,
        direction cmpt,
        label nSweeps
    ) const;
};

} // namespace Foam

#endif
