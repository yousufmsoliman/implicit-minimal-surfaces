#pragma once
#include "geometrycentral/numerical/linear_algebra_utilities.h"

extern "C" {
#include "lobpcg.h"
#include "multivector.h"
#include "temp_multivector.h"
}

extern "C" {
BlopexInt zpotrf_(char* uplo, BlopexInt* n, komplex* a, BlopexInt* lda, BlopexInt* info);
BlopexInt zhegv_(BlopexInt* itype,
                 char* jobz,
                 char* uplo,
                 BlopexInt* n,
                 komplex* a,
                 BlopexInt* lda,
                 komplex* b,
                 BlopexInt* ldb,
                 double* w,
                 komplex* work,
                 BlopexInt* lwork,
                 double* rwork,
                 BlopexInt* info);
}

using Complex = std::complex<double>;
using ComplexVector = geometrycentral::Vector<Complex>;
using ComplexMatvec = std::function<ComplexVector(const ComplexVector&)>;

struct BlopexComplexVector
{
    ComplexVector values;
};

struct BlopexMatvecOperator
{
    ComplexMatvec apply;
};

static komplex toBlopex(Complex z)
{
    return komplex{z.real(), z.imag()};
}

static Complex fromBlopex(const komplex& z)
{
    return {z.real, z.imag};
}

static void* blopexCreateVector(void* sample)
{
    auto* sampleVector = static_cast<BlopexComplexVector*>(sample);
    return new BlopexComplexVector{ComplexVector::Zero(sampleVector->values.size())};
}

static BlopexInt blopexDestroyVector(void* vector)
{
    delete static_cast<BlopexComplexVector*>(vector);
    return 0;
}

static BlopexInt blopexInnerProd(void* x, void* y, void* result)
{
    const auto& xValues = static_cast<BlopexComplexVector*>(x)->values;
    const auto& yValues = static_cast<BlopexComplexVector*>(y)->values;

    // The BLOPEX temporary complex multivector calls InnerProd(y, x) when forming x^* y.
    *static_cast<komplex*>(result) = toBlopex(yValues.dot(xValues));
    return 0;
}

static BlopexInt blopexCopyVector(void* x, void* y)
{
    static_cast<BlopexComplexVector*>(y)->values = static_cast<BlopexComplexVector*>(x)->values;
    return 0;
}

static BlopexInt blopexClearVector(void* x)
{
    static_cast<BlopexComplexVector*>(x)->values.setZero();
    return 0;
}

static BlopexInt blopexSetRandomValues(void* x, BlopexInt seed)
{
    auto& values = static_cast<BlopexComplexVector*>(x)->values;
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (Eigen::Index i = 0; i < values.size(); ++i)
    {
        values(i) = Complex(dist(rng), dist(rng));
    }
    return 0;
}

static BlopexInt blopexScaleVector(double alpha, void* x)
{
    static_cast<BlopexComplexVector*>(x)->values *= alpha;
    return 0;
}

static BlopexInt blopexAxpy(void* alpha, void* x, void* y)
{
    const Complex a = fromBlopex(*static_cast<komplex*>(alpha));
    static_cast<BlopexComplexVector*>(y)->values += a * static_cast<BlopexComplexVector*>(x)->values;
    return 0;
}

static BlopexInt blopexVectorSize(void* x)
{
    return static_cast<BlopexInt>(static_cast<BlopexComplexVector*>(x)->values.size());
}

static void blopexApplyMatvec(void* operatorData, void* x, void* y)
{
    const auto* op = static_cast<BlopexMatvecOperator*>(operatorData);
    auto* xMultivector = static_cast<mv_TempMultiVector*>(x);
    auto* yMultivector = static_cast<mv_TempMultiVector*>(y);

    BlopexInt iyActive = 0;
    for (BlopexInt ix = 0; ix < xMultivector->numVectors; ++ix)
    {
        if (xMultivector->mask != nullptr && xMultivector->mask[ix] == 0)
        {
            continue;
        }

        while (yMultivector->mask != nullptr &&
            iyActive < yMultivector->numVectors &&
            yMultivector->mask[iyActive] == 0)
        {
            ++iyActive;
        }

        assert(iyActive < yMultivector->numVectors);
        auto* xVector = static_cast<BlopexComplexVector*>(xMultivector->vector[ix]);
        auto* yVector = static_cast<BlopexComplexVector*>(yMultivector->vector[iyActive]);
        yVector->values = op->apply(xVector->values);

        ++iyActive;
    }
}

static mv_InterfaceInterpreter makeBlopexComplexInterpreter()
{
    mv_InterfaceInterpreter interpreter{};

    interpreter.CreateVector = blopexCreateVector;
    interpreter.DestroyVector = blopexDestroyVector;
    interpreter.InnerProd = blopexInnerProd;
    interpreter.CopyVector = blopexCopyVector;
    interpreter.ClearVector = blopexClearVector;
    interpreter.SetRandomValues = blopexSetRandomValues;
    interpreter.ScaleVector = blopexScaleVector;
    interpreter.Axpy = blopexAxpy;
    interpreter.VectorSize = blopexVectorSize;

    interpreter.CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
    interpreter.CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
    interpreter.DestroyMultiVector = mv_TempMultiVectorDestroy;
    interpreter.Width = mv_TempMultiVectorWidth;
    interpreter.Height = mv_TempMultiVectorHeight;
    interpreter.SetMask = mv_TempMultiVectorSetMask;
    interpreter.CopyMultiVector = mv_TempMultiVectorCopy;
    interpreter.ClearMultiVector = mv_TempMultiVectorClear;
    interpreter.SetRandomVectors = mv_TempMultiVectorSetRandom;
    interpreter.MultiInnerProd = mv_TempMultiVectorByMultiVector_complex;
    interpreter.MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag_complex;
    interpreter.MultiVecMat = mv_TempMultiVectorByMatrix_complex;
    interpreter.MultiVecMatDiag = mv_TempMultiVectorByDiagonal_complex;
    interpreter.MultiAxpy = mv_TempMultiVectorAxpy_complex;
    interpreter.MultiXapy = mv_TempMultiVectorXapy_complex;
    interpreter.Eval = mv_TempMultiVectorEval;
    interpreter.MultiPrint = nullptr;

    return interpreter;
}

inline ComplexVector smallestEigenvectorBLOPEX(const ComplexMatvec& energyMatvec,
                                               const ComplexMatvec& massMatvec,
                                               size_t n,
                                               size_t maxIterations,
                                               double tol = 1e-6,
                                               bool verbose = true,
                                               size_t blockSize = 1)
{
    assert(n <= static_cast<size_t>(std::numeric_limits<BlopexInt>::max()));
    blockSize = std::max<size_t>(1, std::min(blockSize, n));
    assert(blockSize <= static_cast<size_t>(std::numeric_limits<BlopexInt>::max()));

    mv_InterfaceInterpreter interpreter = makeBlopexComplexInterpreter();
    BlopexComplexVector sample{ComplexVector::Zero(static_cast<Eigen::Index>(n))};
    auto* multivectorData = static_cast<mv_TempMultiVector*>(
        mv_TempMultiVectorCreateFromSampleVector(&interpreter, static_cast<BlopexInt>(blockSize), &sample));
    mv_TempMultiVectorSetRandom(multivectorData, 1);

    mv_MultiVectorPtr x = mv_MultiVectorWrap(&interpreter, multivectorData, 1);

    BlopexMatvecOperator energyOperator{energyMatvec};
    BlopexMatvecOperator massOperator{massMatvec};

    lobpcg_BLASLAPACKFunctions lapack{};
    lapack.zpotrf = zpotrf_;
    lapack.zhegv = zhegv_;

    lobpcg_Tolerance tolerance{};
    tolerance.absolute = tol;
    tolerance.relative = 1e-50;

    int iterations = 0;
    std::vector<komplex> lambda(blockSize);
    std::vector<double> residual(blockSize);

    lobpcg_solve_complex(
        x,
        &energyOperator,
        blopexApplyMatvec,
        massMatvec ? static_cast<void*>(&massOperator) : nullptr,
        massMatvec ? blopexApplyMatvec : nullptr,
        nullptr,
        nullptr,
        nullptr,
        lapack,
        tolerance,
        static_cast<int>(maxIterations),
        verbose ? 1 : 0,
        &iterations,
        lambda.data(),
        nullptr,
        0,
        residual.data(),
        nullptr,
        0);

    size_t bestIndex = 0;
    for (size_t i = 1; i < blockSize; ++i)
    {
        if (lambda[i].real < lambda[bestIndex].real)
        {
            bestIndex = i;
        }
    }

    auto* resultData = static_cast<mv_TempMultiVector*>(mv_MultiVectorGetData(x));
    ComplexVector result = static_cast<BlopexComplexVector*>(resultData->vector[bestIndex])->values;
    mv_MultiVectorDestroy(x);

    if (verbose)
    {
        std::cerr << "BLOPEX block size: " << blockSize
            << "\tBLOPEX # iters: " << iterations
            << "\tBLOPEX smallest lambda: " << fromBlopex(lambda[bestIndex])
            << "\tBLOPEX residual: " << residual[bestIndex] << std::endl;
    }

    return result;
}

inline ComplexVector smallestEigenvectorBLOPEX(const ComplexMatvec& energyMatvec,
                                               size_t n,
                                               size_t maxIterations,
                                               double tol = 1e-6,
                                               bool verbose = true,
                                               size_t blockSize = 8)
{
    return smallestEigenvectorBLOPEX(energyMatvec, ComplexMatvec{}, n, maxIterations, tol, verbose, blockSize);
}

inline ComplexVector smallestEigenvectorBLOPEX(SparseMatrix<Complex>& energyMatrix,
                                               SparseMatrix<Complex>& massMatrix,
                                               size_t maxIterations,
                                               double tol = 1e-6,
                                               bool verbose = true,
                                               size_t blockSize = 8)
{
    const size_t n = static_cast<size_t>(energyMatrix.rows());
    return smallestEigenvectorBLOPEX(
        [&energyMatrix](const ComplexVector& x) { return energyMatrix * x; },
        [&massMatrix](const ComplexVector& x) { return massMatrix * x; },
        n,
        maxIterations,
        tol,
        verbose,
        blockSize);
}
