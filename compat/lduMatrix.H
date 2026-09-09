#ifndef lduMatrix_H
#define lduMatrix_H

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace Foam
{

using scalar = double;
using label = int;
using direction = unsigned int;
using word = std::string;

template<class Type>
class Field : public std::vector<Type>
{
    using Base = std::vector<Type>;

public:
    using Base::Base;
    using Base::operator=;

    Field() = default;
    Field(const Field&) = default;
    Field(Field&&) = default;
    Field& operator=(const Field&) = default;
    Field& operator=(Field&&) = default;

    Type* begin() noexcept { return Base::data(); }
    const Type* begin() const noexcept { return Base::data(); }
    Type* end() noexcept { return Base::data() + Base::size(); }
    const Type* end() const noexcept { return Base::data() + Base::size(); }

    Field& operator=(const Type value)
    {
        std::fill(Base::begin(), Base::end(), value);
        return *this;
    }

    void negate()
    {
        for (Type& value : *this) value = -value;
    }
};

using scalarField = Field<scalar>;
using labelField = Field<label>;

template<template<class> class FieldType, class Type>
class FieldField : public std::vector<FieldType<Type>>
{
public:
    using std::vector<FieldType<Type>>::vector;
};

class lduInterfaceFieldPtrsList
{
    std::vector<const void*> values_;

public:
    explicit lduInterfaceFieldPtrsList(const label size = 0)
    : values_(static_cast<std::size_t>(size), nullptr)
    {}

    label size() const { return static_cast<label>(values_.size()); }
    bool set(const label i) const
    {
        return values_[static_cast<std::size_t>(i)] != nullptr;
    }
};

class lduAddressing
{
    labelField upperAddr_;
    labelField ownerStartAddr_;

public:
    lduAddressing(labelField upperAddr, labelField ownerStartAddr)
    :
        upperAddr_(std::move(upperAddr)),
        ownerStartAddr_(std::move(ownerStartAddr))
    {}

    const labelField& upperAddr() const { return upperAddr_; }
    const labelField& ownerStartAddr() const { return ownerStartAddr_; }
};

class lduMatrix
{
public:
    class smoother;

private:
    lduAddressing addressing_;
    scalarField diag_;
    scalarField upper_;
    scalarField lower_;

public:
    lduMatrix
    (
        lduAddressing addressing,
        scalarField diag,
        scalarField upper,
        scalarField lower
    )
    :
        addressing_(std::move(addressing)),
        diag_(std::move(diag)),
        upper_(std::move(upper)),
        lower_(std::move(lower))
    {}

    const lduAddressing& lduAddr() const { return addressing_; }
    const scalarField& diag() const { return diag_; }
    const scalarField& upper() const { return upper_; }
    const scalarField& lower() const { return lower_; }

    void initMatrixInterfaces
    (
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,
        const scalarField&,
        scalarField&,
        direction
    ) const
    {}

    void updateMatrixInterfaces
    (
        const FieldField<Field, scalar>&,
        const lduInterfaceFieldPtrsList&,
        const scalarField&,
        scalarField&,
        direction
    ) const
    {}
};

class lduMatrix::smoother
{
protected:
    const word& fieldName_;
    const lduMatrix& matrix_;
    const FieldField<Field, scalar>& interfaceBouCoeffs_;
    const FieldField<Field, scalar>& interfaceIntCoeffs_;
    const lduInterfaceFieldPtrsList& interfaces_;

public:
    smoother
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces
    )
    :
        fieldName_(fieldName),
        matrix_(matrix),
        interfaceBouCoeffs_(interfaceBouCoeffs),
        interfaceIntCoeffs_(interfaceIntCoeffs),
        interfaces_(interfaces)
    {}

    virtual ~smoother() = default;

    virtual void smooth(scalarField&, const scalarField&, direction, label) const = 0;

    template<class> struct addsymMatrixConstructorToTable {};
    template<class> struct addasymMatrixConstructorToTable {};
};

}

#define TypeName(Name) inline static constexpr const char* typeName = Name
#define defineTypeNameAndDebug(Type, DebugSwitch)
#define forAll(Container, Index) \
    for (Foam::label Index = 0; Index < static_cast<Foam::label>((Container).size()); ++Index)

#endif
