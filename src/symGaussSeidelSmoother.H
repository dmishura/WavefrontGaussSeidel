#ifndef symGaussSeidelSmoother_H
#define symGaussSeidelSmoother_H

#include "lduMatrix.H"

namespace Foam
{

class symGaussSeidelSmoother
:
    public lduMatrix::smoother
{
public:

    TypeName("symGaussSeidel");

    symGaussSeidelSmoother
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces
    );

    static void smooth
    (
        const word& fieldName,
        scalarField& psi,
        const lduMatrix& matrix,
        const scalarField& source,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        direction cmpt,
        label nSweeps
    );

    virtual void smooth
    (
        scalarField& psi,
        const scalarField& source,
        direction cmpt,
        label nSweeps
    ) const;
};

} // End namespace Foam

#endif
