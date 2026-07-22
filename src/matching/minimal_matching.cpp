#include "minimal_matching.h"

#include "linebundle/connection_laplacian.h"
#include "linebundle/singularity_locator.h"

#include "geometrycentral/numerical/linear_algebra_utilities.h"

#include <exception>
#include <queue>

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/utilities/timing.h"
#include "lbfgs.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <limits>
#include <random>
#include <vector>

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

namespace
{
    using Complex = std::complex<double>;
    using ComplexVector = geometrycentral::Vector<Complex>;
    using ComplexMatvec = std::function<ComplexVector(const ComplexVector&)>;

    struct BlopexComplexVector {
        ComplexVector values;
    };

    struct BlopexMatvecOperator {
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
        for (Eigen::Index i = 0; i < values.size(); ++i) {
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
        for (BlopexInt ix = 0; ix < xMultivector->numVectors; ++ix) {
            if (xMultivector->mask != nullptr && xMultivector->mask[ix] == 0) {
                continue;
            }

            while (yMultivector->mask != nullptr &&
                   iyActive < yMultivector->numVectors &&
                   yMultivector->mask[iyActive] == 0) {
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

    ComplexVector smallestEigenvectorBLOPEX(const ComplexMatvec& energyMatvec,
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
        for (size_t i = 1; i < blockSize; ++i) {
            if (lambda[i].real < lambda[bestIndex].real) {
                bestIndex = i;
            }
        }

        auto* resultData = static_cast<mv_TempMultiVector*>(mv_MultiVectorGetData(x));
        ComplexVector result = static_cast<BlopexComplexVector*>(resultData->vector[bestIndex])->values;
        mv_MultiVectorDestroy(x);

        if (verbose) {
            std::cerr << "BLOPEX block size: " << blockSize
                      << "\tBLOPEX # iters: " << iterations
                      << "\tBLOPEX smallest lambda: " << fromBlopex(lambda[bestIndex])
                      << "\tBLOPEX residual: " << residual[bestIndex] << std::endl;
        }

        return result;
    }

    ComplexVector smallestEigenvectorBLOPEX(const ComplexMatvec& energyMatvec,
                                            size_t n,
                                            size_t maxIterations,
                                            double tol = 1e-6,
                                            bool verbose = true,
                                            size_t blockSize = 8)
    {
        return smallestEigenvectorBLOPEX(energyMatvec, ComplexMatvec{}, n, maxIterations, tol, verbose, blockSize);
    }

    ComplexVector smallestEigenvectorBLOPEX(SparseMatrix<Complex>& energyMatrix,
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

    ComplexVector slicewiseMatrixVectorProduct(const std::vector<SparseMatrix<Complex>>& matrixASlice,
                                     const std::vector<SparseMatrix<Complex>>& matrixBSlice,
                                     const Vector<double>& massA,
                                     const Vector<double>& massB,
                                     const ComplexVector* potentialDiag,
                                     double potentialScale,
                                     size_t nA,
                                     size_t nB,
                                     const ComplexVector& x)
    {
        ComplexVector y = ComplexVector::Zero(static_cast<Eigen::Index>(nA * nB));

        ComplexVector xB(static_cast<Eigen::Index>(nB));
        for (size_t iA = 0; iA < nA; ++iA) {
            for (size_t iB = 0; iB < nB; ++iB) {
                xB(static_cast<Eigen::Index>(iB)) = x(static_cast<Eigen::Index>(iA + iB * nA));
            }

            ComplexVector yB = matrixASlice[iA] * xB;
            for (size_t iB = 0; iB < nB; ++iB) {
                y(static_cast<Eigen::Index>(iA + iB * nA)) += massA(static_cast<Eigen::Index>(iA)) * yB(static_cast<Eigen::Index>(iB));
            }
        }

        for (size_t iB = 0; iB < nB; ++iB) {
            Eigen::Index offset = static_cast<Eigen::Index>(iB * nA);
            ComplexVector xA = x.segment(offset, static_cast<Eigen::Index>(nA));
            ComplexVector yA = matrixBSlice[iB] * xA;
            y.segment(offset, static_cast<Eigen::Index>(nA)) += massB(static_cast<Eigen::Index>(iB)) * yA;
        }

        if (potentialDiag != nullptr && potentialScale != 0.) {
            y += potentialScale * potentialDiag->cwiseProduct(x);
        }

        return y;
    }
}

namespace minimalmatching
{
    MinimalMatchingSolver::MinimalMatchingSolver(
        ManifoldSurfaceMesh& mesh_A_,
        ManifoldSurfaceMesh& mesh_B_,

        IntrinsicGeometryInterface& geometry_A_,
        IntrinsicGeometryInterface& geometry_B_,

        VertexPositionGeometry& extrinsic_geometry_A_,
        VertexPositionGeometry& extrinsic_geometry_B_
        ) :
    mesh_A(mesh_A_), mesh_B(mesh_B_), geometry_A(geometry_A_), geometry_B(geometry_B_), extrinsic_geometry_A(extrinsic_geometry_A_), extrinsic_geometry_B(extrinsic_geometry_B_), productMesh(mesh_A, mesh_B)
    {
        geometry_A.requireDECOperators();
        geometry_B.requireDECOperators();

        dδ_A = geometry_A.d1 * geometry_A.d1.transpose();
        dδ_B = geometry_B.d1 * geometry_B.d1.transpose();

        dδ_A_solver = std::make_unique<SquareSolver<double>>(dδ_A);
        dδ_B_solver = std::make_unique<SquareSolver<double>>(dδ_B);

        buildSpinConnections();
        if (options.bundleType == BundleType::SKYSCRAPER || mesh_A.genus() != 0)
            buildSkyscraperConnections();

        if (mesh_A.genus() != 0)
        {
            throw std::runtime_error("Only genus zero surfaces are supported");
        }

        computeSingularityPinningPotential();
    }


    HalfedgeData<Vector2> liftToSpinConnection(ManifoldSurfaceMesh &mesh, HalfedgeData<Vector2> &leviCivita, FaceData<double> &curvature)
    {
        HalfedgeData<Vector2> spin(mesh, Vector2{1,0});
        for (Halfedge h : mesh.interiorHalfedges())
        {
            if (!leviCivita[h].isFinite() || !leviCivita[h].isDefined())
                std::cerr << "Non-finite Levi-Civita connection" << std::endl;
            spin[h] = leviCivita[h].pow(0.5);
            if (!spin[h].isFinite() || !spin[h].isDefined())
                std::cerr << "Non-finite spin connection" << std::endl;
        }

        FaceData<int> q(mesh, 0);
        FaceData<int> q_total(mesh, 0);

        for (Face f : mesh.faces())
        {
            Vector2 monodromy{1,0};
            for (Halfedge ij : f.adjacentHalfedges())
                monodromy *= spin[ij];

            Vector2 exp_iK = Vector2::fromAngle(curvature[f] / 2.);
            double angle = (monodromy / exp_iK).arg();
            if (fabs(angle) < 1e-1)
                q[f] = 0;
            else
                q[f] = 1;

            q_total[f] = q[f];
        }

        long max_dist = 0;
        FaceData<long> dual_tree_dist(mesh, -1);
        FaceData<Face> triangle_parent(mesh, Face());
        for (Face f : mesh.faces())
            triangle_parent[f] = f;

        Face root = mesh.face(0);
        dual_tree_dist[root] = 0;

        std::queue<Face> queue;
        queue.push(root);

        while (!queue.empty())
        {
            Face f = queue.front();
            queue.pop();

            for (Halfedge ij : f.adjacentHalfedges())
            {
                Halfedge ji = ij.twin();

                Face g = ji.face();
                if (triangle_parent[g] == g && g != root)
                {
                    dual_tree_dist[g] = dual_tree_dist[f] + 1;
                    max_dist = std::max(max_dist, dual_tree_dist[g]);
                    triangle_parent[g] = f;
                    queue.push(g);
                }
            }
        }

        // Count the sum of q_A (mod 2) for the descendents of an edge in a
        // dual spanning tree
        for (long d = max_dist; d >= 0; d--)
        {
            for (Face f : mesh.faces())
                if (dual_tree_dist[f] == d)
                {
                    Face g = triangle_parent[f];
                    q_total[g] += q_total[f];
                }
        }

        for (Face f : mesh.faces())
        {
            for (Halfedge ij : f.adjacentHalfedges())
            {
                Halfedge ji = ij.twin();
                Face g = ji.face();

                if (g == triangle_parent[f])
                {
                    if (q_total[f] % 2 == 1)
                    {
                        spin[ij] *= -1;
                        spin[ji] *= -1;
                    }
                    break;
                }
            }
        }

        return spin;
    }

    void MinimalMatchingSolver::buildSpinConnections()
    {
        geometry_A.requireTransportVectorsAlongHalfedge();
        geometry_B.requireTransportVectorsAlongHalfedge();

        geometry_A.requireFaceGaussianCurvatures();
        geometry_B.requireFaceGaussianCurvatures();

        leviCivita_A = geometry_A.transportVectorsAlongHalfedge;
        leviCivita_B = geometry_B.transportVectorsAlongHalfedge;

        gaussCurvature_A = geometry_A.faceGaussianCurvatures;
        gaussCurvature_B = geometry_B.faceGaussianCurvatures;

        for (Face f : mesh_A.faces())
        {
            Vector2 monodromy{1,0};
            for (Halfedge ij : f.adjacentHalfedges())
            {
                monodromy *= leviCivita_A[ij];
            }

            Vector2 diff = monodromy / Vector2::fromAngle(gaussCurvature_A[f]);
            if (diff.arg() > 1e-2)
                std::cerr << "Levi-Civita curvature is not compatible with the connection! " << diff.arg() << std::endl;
        }

        spinConnection_A = liftToSpinConnection(mesh_A, leviCivita_A, gaussCurvature_A);
        spinConnection_B = liftToSpinConnection(mesh_B, leviCivita_B, gaussCurvature_B);

        spinCurvature_A = FaceData<double>(mesh_A, gaussCurvature_A.raw() / 2.);
        spinCurvature_B = FaceData<double>(mesh_B, gaussCurvature_B.raw() / 2.);

        geometry_A.requireFaceAreas();
        geometry_B.requireFaceAreas();

        areaCurvature_A = geometry_A.faceAreas;
        areaCurvature_A *= (2. * M_PI) / (areaCurvature_A.raw().sum());
        areaCurvature_B = geometry_B.faceAreas;
        areaCurvature_B *= (2. * M_PI) / (areaCurvature_B.raw().sum());

        correspondenceConnection_A = spinConnection_A;
        correspondenceConnection_B = spinConnection_B;

        correspondenceCurvature_A = spinCurvature_A;
        correspondenceCurvature_B = spinCurvature_B;

        if (options.curvatureAType == CurvatureType::AREA)
        {
            geometrycentral::Vector<double> curvature_change = areaCurvature_A.raw() - spinCurvature_A.raw();
            geometrycentral::Vector<double> u = dδ_A_solver->solve(curvature_change);
            geometrycentral::Vector<double> star_du = geometry_A.d1.transpose() * u;

            for (Halfedge ij : mesh_A.halfedges())
            {
                size_t iIJ = productMesh.eInd_A[ij.edge()];
                int s = ij.edge().halfedge() == ij ? 1 : -1;
                correspondenceConnection_A[ij] = Vector2::fromAngle(s * star_du(iIJ)) * spinConnection_A[ij];
            }

            correspondenceCurvature_A = areaCurvature_A;
        }

        if (options.curvatureBType == CurvatureType::AREA)
        {
            geometrycentral::Vector<double> curvature_change = areaCurvature_B.raw() - spinCurvature_B.raw();
            geometrycentral::Vector<double> u = dδ_B_solver->solve(curvature_change);
            geometrycentral::Vector<double> star_du = geometry_B.d1.transpose() * u;

            for (Halfedge ij : mesh_B.halfedges())
            {
                size_t iIJ = productMesh.eInd_B[ij.edge()];
                int s = ij.edge().halfedge() == ij ? 1 : -1;
                correspondenceConnection_B[ij] = Vector2::fromAngle(s * star_du(iIJ)) * spinConnection_B[ij];
            }

            correspondenceCurvature_B = areaCurvature_B;
        }

        buildProductConnectionLaplacian();
    }

    void MinimalMatchingSolver::buildProductConnectionLaplacian()
    {
        for (Face f : mesh_A.faces())
        {
            Vector2 monodromy{1,0};
            for (Halfedge ij : f.adjacentHalfedges())
            {
                monodromy *= correspondenceConnection_A[ij];
            }

            Vector2 diff = monodromy / Vector2::fromAngle(correspondenceCurvature_A[f]);
            if (diff.arg() > 1e-2)
                std::cerr << "Curvature is not compatible with the connection! " << diff.arg() << std::endl;
        }

        if (options.laplacianType == FEM)
        {
            connectionLaplacian_A = complexBundleLaplacian(mesh_A, geometry_A, correspondenceConnection_A, correspondenceCurvature_A, false);
            connectionLaplacian_B = complexBundleLaplacian(mesh_B, geometry_B, correspondenceConnection_B, correspondenceCurvature_B, false);
        }
        else if (options.laplacianType == DEC)
        {
            connectionLaplacian_A = complexBundleLaplacianDEC(mesh_A, geometry_A, correspondenceConnection_A, correspondenceCurvature_A);
            connectionLaplacian_B = complexBundleLaplacianDEC(mesh_B, geometry_B, correspondenceConnection_B, correspondenceCurvature_B);

        }

        if (options.massMatrixType == FEM)
        {
            massMatrix_A = complexBundleMassMatrix(mesh_A, geometry_A, correspondenceConnection_A, correspondenceCurvature_A);
            massMatrix_B = complexBundleMassMatrix(mesh_B, geometry_B, correspondenceConnection_B, correspondenceCurvature_B);
        }
        else
        {
            massMatrix_A = complexBundleMassMatrixDEC(mesh_A, geometry_A, correspondenceConnection_A, correspondenceCurvature_A);
            massMatrix_B = complexBundleMassMatrixDEC(mesh_B, geometry_B, correspondenceConnection_B, correspondenceCurvature_B);
        }

        geometry_A.requireVertexLumpedMassMatrix();
        geometry_B.requireVertexLumpedMassMatrix();
        massMatrixReal_A = geometry_A.vertexLumpedMassMatrix;
        massMatrixReal_B = geometry_B.vertexLumpedMassMatrix;

        geometrycentral::Vector<std::complex<double>> psi_A = geometrycentral::smallestEigenvectorPositiveDefinite(connectionLaplacian_A, massMatrix_A);
        geometrycentral::Vector<std::complex<double>> psi_B = geometrycentral::smallestEigenvectorPositiveDefinite(connectionLaplacian_B, massMatrix_B);

        double lambda_A = (psi_A.adjoint() * (connectionLaplacian_A * psi_A))(0).real();
        double lambda_B = (psi_B.adjoint() * (connectionLaplacian_B * psi_B))(0).real();

        std::cerr << "lambda_A = " << lambda_A << " / " << (psi_A.adjoint() * (connectionLaplacian_A * psi_A))(0) << std::endl;
        std::cerr << "lambda_B = " << lambda_B << " / " << (psi_B.adjoint() * (connectionLaplacian_B * psi_B))(0) <<  std::endl;

        geometry_A.requireMeshLengthScale();
        geometry_B.requireMeshLengthScale();
        double h = (geometry_A.meshLengthScale + geometry_B.meshLengthScale) / 2.;
        EPSILON0 = (lambda_A + lambda_B);
    }


    void MinimalMatchingSolver::buildSkyscraperConnections()
    {
        skyscraperConnection_A = HalfedgeData<Vector2>(mesh_A, Vector2{1,0});
        skyscraperConnection_B = HalfedgeData<Vector2>(mesh_B, Vector2{0,1});
        geometrycentral::Vector<double> curvature_A = spinCurvature_A.raw();
        if (options.curvatureAType == CurvatureType::AREA)
            curvature_A = areaCurvature_A.raw();

        Face f_A = mesh_A.face(0);

        Face root_A = f_A;
        curvature_A(root_A.getIndex()) -= 2. * M_PI;

        geometrycentral::Vector<double> u_A = dδ_A_solver->solve(curvature_A);
        geometrycentral::Vector<double> star_du_A = geometry_A.d1.transpose() * u_A;

        for (Halfedge ij : mesh_A.halfedges())
        {
            size_t iIJ = productMesh.eInd_A[ij.edge()];
            int s = ij.edge().halfedge() == ij ? 1 : -1;
            skyscraperConnection_A[ij] = Vector2::fromAngle(s * star_du_A(iIJ));
        }

        geometrycentral::Vector<double> curvature_B = spinCurvature_B.raw();
        if (options.curvatureBType == CurvatureType::AREA)
            curvature_B = areaCurvature_B.raw();

        Face f_B = mesh_B.face(0);

        Face root_B = f_B;
        curvature_B(root_B.getIndex()) -= 2. * M_PI;

        geometrycentral::Vector<double> u_B = dδ_B_solver->solve(curvature_B);
        geometrycentral::Vector<double> star_du_B = geometry_B.d1.transpose() * u_B;

        for (Halfedge ij : mesh_B.halfedges())
        {
            size_t iIJ = productMesh.eInd_B[ij.edge()];
            int s = ij.edge().halfedge() == ij ? 1 : -1;
            skyscraperConnection_B[ij] = Vector2::fromAngle(s * star_du_B(iIJ));
        }

        correspondenceConnection_A = skyscraperConnection_A;
        correspondenceConnection_B = skyscraperConnection_B;

        if (options.curvatureAType == CurvatureType::AREA)
            correspondenceCurvature_A = areaCurvature_A;
        else
            correspondenceCurvature_A = spinCurvature_A;

        if (options.curvatureBType == CurvatureType::AREA)
            correspondenceCurvature_B = areaCurvature_B;
        else
            correspondenceCurvature_B = spinCurvature_B;

        buildProductConnectionLaplacian();
    }

    void MinimalMatchingSolver::closestPointInitialization(VertexData<Vector3>& pos_A_, VertexData<Vector3>& pos_B_)
    {
        auto pos_A = pos_A_.reinterpretTo(mesh_A);
        auto pos_B = pos_B_.reinterpretTo(mesh_B);

        VertexData<SurfacePoint> A_to_B(mesh_A);
        VertexData<SurfacePoint> B_to_A(mesh_B);

            // We use a slicewise trivial connection
            closestPointConnection_A = VertexData<HalfedgeData<Vector2>>(mesh_A, HalfedgeData<Vector2>(mesh_B, Vector2{1,0}));
            closestPointCurvature_A = VertexData<FaceData<double>>(mesh_A, FaceData<double>(mesh_B, 0));
            for (Vertex v_A : mesh_A.vertices())
            {
                Face target_face;
                Vector3 target_point{};

                std::tie(target_face, target_point) = closestPoint(pos_A[v_A], mesh_B, pos_B);

                std::vector<Vertex> vc_A = {target_face.halfedge().vertex(), target_face.halfedge().next().vertex(), target_face.halfedge().next().next().vertex()};
                Vector3 bc = computeBarycentricCoords(
                    extrinsic_geometry_B.vertexPositions[vc_A[0].getIndex()],
                    extrinsic_geometry_B.vertexPositions[vc_A[1].getIndex()],
                    extrinsic_geometry_B.vertexPositions[vc_A[2].getIndex()],
                    target_point
                );

                SurfacePoint target_surface_point(target_face,bc);
                A_to_B[v_A] = target_surface_point;

                double t = options.trivialConnectionBlendingWeight;

                double total_curvature = correspondenceCurvature_B.raw().sum();
                FaceData<double> target_curvature(mesh_B, 0);
                target_curvature[target_face] += total_curvature;

                target_curvature = t * target_curvature + (1-t) * correspondenceCurvature_B;

                geometrycentral::Vector<double> curvature_change = target_curvature.raw() - correspondenceCurvature_B.raw();
                closestPointCurvature_A[v_A] = target_curvature;

                geometrycentral::Vector<double> u = dδ_B_solver->solve(curvature_change);
                geometrycentral::Vector<double> star_du = geometry_B.d1.transpose() * u;

                for (Halfedge ij : mesh_B.halfedges())
                {
                    size_t iIJ = productMesh.eInd_B[ij.edge()];
                    int s = ij.edge().halfedge() == ij ? 1 : -1;
                    closestPointConnection_A[v_A][ij] = Vector2::fromAngle(s * star_du(iIJ)) * correspondenceConnection_B[ij];
                }
            }

            // We use a slicewise trivial connection
            closestPointConnection_B = VertexData<HalfedgeData<Vector2>>(mesh_B, HalfedgeData<Vector2>(mesh_A, Vector2{1,0}));
            closestPointCurvature_B = VertexData<FaceData<double>>(mesh_B, FaceData<double>(mesh_A, 0));

            for (Vertex v_B : mesh_B.vertices())
            {
                Face target_face;
                Vector3 target_point{};

                std::tie(target_face, target_point) = closestPoint(pos_B[v_B], mesh_A, pos_A);

                std::vector<Vertex> vc = {target_face.halfedge().vertex(), target_face.halfedge().next().vertex(), target_face.halfedge().next().next().vertex()};
                Vector3 bc = computeBarycentricCoords(
                    extrinsic_geometry_A.vertexPositions[vc[0].getIndex()],
                    extrinsic_geometry_A.vertexPositions[vc[1].getIndex()],
                    extrinsic_geometry_A.vertexPositions[vc[2].getIndex()],
                    target_point
                );

                SurfacePoint target_surface_point(target_face,bc);
                B_to_A[v_B] = target_surface_point;

                double t = options.trivialConnectionBlendingWeight;

                double total_curvature = correspondenceCurvature_A.raw().sum();

                FaceData<double> target_curvature(mesh_A, 0);
                target_curvature[target_face] += total_curvature;
                target_curvature = t * target_curvature + (1-t) * correspondenceCurvature_A;

                geometrycentral::Vector<double> curvature_change = target_curvature.raw() - correspondenceCurvature_A.raw();
                closestPointCurvature_B[v_B] = target_curvature;

                geometrycentral::Vector<double> u = dδ_A_solver->solve(curvature_change);
                geometrycentral::Vector<double> star_du = geometry_A.d1.transpose() * u;

                for (Halfedge ij : mesh_A.halfedges())
                {
                    size_t iIJ = productMesh.eInd_A[ij.edge()];
                    int s = ij.edge().halfedge() == ij ? 1 : -1;
                    closestPointConnection_B[v_B][ij] = Vector2::fromAngle(s * star_du(iIJ)) * correspondenceConnection_A[ij];
                }
            }

            size_t nv_prod = productMesh.nCells(0);
            size_t nA = mesh_A.nVertices();
            size_t nB = mesh_B.nVertices();

            geometrycentral::Vector<double> massA(nA);
            geometrycentral::Vector<double> massB(nB);
            std::vector<SparseMatrix<std::complex<double>>> connectionLaplacianASlices(nA);
            std::vector<SparseMatrix<std::complex<double>>> connectionLaplacianBSlices(nB);

            for (Vertex v_A : mesh_A.vertices())
            {
                size_t iA = productMesh.vInd_A[v_A];
                massA(static_cast<Eigen::Index>(iA)) = geometry_A.vertexDualAreas[v_A];
                connectionLaplacianASlices[iA] =
                    complexBundleLaplacianDEC(mesh_B, geometry_B, closestPointConnection_A[v_A], closestPointCurvature_A[v_A]);
            }

            for (Vertex v_B : mesh_B.vertices())
            {
                size_t iB = productMesh.vInd_B[v_B];
                massB(static_cast<Eigen::Index>(iB)) = geometry_B.vertexDualAreas[v_B];
                connectionLaplacianBSlices[iB] =
                    complexBundleLaplacianDEC(mesh_A, geometry_A, closestPointConnection_B[v_B], closestPointCurvature_B[v_B]);
            }

            DenseMatrix<double> V(mesh_A.nVertices(), mesh_B.nVertices());
            V.setConstant(std::numeric_limits<double>::infinity());

            HeatMethodDistanceSolver distance_solver_A(geometry_A);
            HeatMethodDistanceSolver distance_solver_B(geometry_B);

            geometry_A.requireVertexIndices();
            geometry_B.requireVertexIndices();

            for (Vertex v : mesh_A.vertices())
            {
                VertexData<double> dist_a = distance_solver_A.computeDistance(v);
                VertexData<double> dist_b = distance_solver_B.computeDistance(A_to_B[v]);

                for (Vertex v_A : mesh_A.vertices())
                    for (Vertex v_B : mesh_B.vertices())
                    {
                        size_t iV_A = geometry_A.vertexIndices[v_A];
                        size_t iV_B = geometry_B.vertexIndices[v_B];

                        double dist2 = dist_a[v_A] * dist_a[v_A] + dist_b[v_B] * dist_b[v_B];
                        dist2 *= options.doubleWellCoeff;
                        double V0 = 1. - exp(-dist2);
                        V(iV_A, iV_B) = fmin(V(iV_A, iV_B), V0);
                    }
            }

            for (Vertex v : mesh_B.vertices())
            {
                VertexData<double> dist_a = distance_solver_A.computeDistance(B_to_A[v]);
                VertexData<double> dist_b = distance_solver_B.computeDistance(v);

                for (Vertex v_A : mesh_A.vertices())
                    for (Vertex v_B : mesh_B.vertices())
                    {
                        size_t iV_A = geometry_A.vertexIndices[v_A];
                        size_t iV_B = geometry_B.vertexIndices[v_B];

                        double dist2 = dist_a[v_A] * dist_a[v_A] + dist_b[v_B] * dist_b[v_B];
                        dist2 *= options.doubleWellCoeff;
                        double V0 = 1. - exp(-dist2);
                        V(iV_A, iV_B) = fmin(V(iV_A, iV_B), V0);
                    }
            }

            V /= V.maxCoeff();

            geometrycentral::Vector<std::complex<double>> mass_diag(nv_prod);
            geometrycentral::Vector<std::complex<double>> V_diag(nv_prod);
            for (Vertex v_A : mesh_A.vertices())
                for (Vertex v_B : mesh_B.vertices())
                {
                    size_t v_AxB = productMesh.vertexIndex(v_A, v_B);
                    mass_diag(productMesh.vertexIndex(v_A, v_B)) = geometry_A.vertexDualAreas[v_A] * geometry_B.vertexDualAreas[v_B];
                    V_diag(productMesh.vertexIndex(v_A, v_B)) = geometry_A.vertexDualAreas[v_A] * geometry_B.vertexDualAreas[v_B]
                    * (V(productMesh.vInd_A[v_A], productMesh.vInd_B[v_B]));
                }

            // Now we compute an actual eigenvector of our slicewise trivial connection
            auto energyMatvec = [&](const geometrycentral::Vector<std::complex<double>>& x) {
                return slicewiseMatrixVectorProduct(connectionLaplacianASlices, connectionLaplacianBSlices, massA, massB, nullptr, 0., nA, nB, x);
            };
            auto massMatvec = [&](const geometrycentral::Vector<std::complex<double>>& x) {
                return mass_diag.cwiseProduct(x);
            };
            initializationSection = smallestEigenvectorBLOPEX(energyMatvec, massMatvec, nv_prod, 1000);
            currentSection = initializationSection;
    }

    std::pair<VertexData<SurfacePoint>, VertexData<SurfacePoint>> MinimalMatchingSolver::extractCorrespondence()
    {
        auto &psi = currentSection;

        VertexData<SurfacePoint> pos_A(mesh_A);
        VertexData<SurfacePoint> pos_B(mesh_B);

        // Find the positions inside of B where the points of A are mapped
        for (Vertex v_A : mesh_A.vertices())
        {
            bool found = false;
            for (Face f_B : mesh_B.faces())
            {
                geometrycentral::Vector<double> singularity_locator_aux_data(7);
                double index_twoform = 0;

                int h_count = 0;
                for (Halfedge ij : f_B.adjacentHalfedges())
                {
                    Vertex vi_B = ij.tailVertex();
                    Vertex vj_B = ij.tipVertex();

                    size_t src_AxB = productMesh.vertexIndex(v_A, vi_B);
                    size_t dst_AxB = productMesh.vertexIndex(v_A, vj_B);

                    std::complex<double> rho = correspondenceConnection_B[ij];
                    double xi_ij = std::arg(psi(dst_AxB) / (rho * psi(src_AxB)));
                    index_twoform += xi_ij;

                    singularity_locator_aux_data(h_count) = std::abs(psi(src_AxB));
                    singularity_locator_aux_data(h_count + 3) = xi_ij;

                    h_count++;
                }

                singularity_locator_aux_data(6) = correspondenceCurvature_B[f_B];
                index_twoform += correspondenceCurvature_B[f_B];
                index_twoform /= 2. * M_PI;

                int degree = std::rint(index_twoform);
                if (degree != 0 && degree != +1)
                    std::cerr << "Found a singularity on face " << v_A << " x " << f_B << " with degree = " << degree << std::endl;

                if (degree != 0)
                {
                    found = true;

                    Vector3 barycoords = locateSingularity( singularity_locator_aux_data(3), singularity_locator_aux_data(4), singularity_locator_aux_data(5),
                        singularity_locator_aux_data(6), singularity_locator_aux_data(0), singularity_locator_aux_data(1), singularity_locator_aux_data(2));
                    pos_A[v_A] = SurfacePoint(f_B, barycoords);
                }
            }

            if (!found)
                std::cerr << "Failed to find a point where the vertex " << v_A << " is mapped to inside of B" << std::endl;
        }

        // Find the positions inside of A where the points of B are mapped
        for (Vertex v_B : mesh_B.vertices())
        {
            bool found = false;
            for (Face f_A : mesh_A.faces())
            {
                geometrycentral::Vector<double> singularity_locator_aux_data(7);
                double index_twoform = 0;

                int h_count = 0;
                for (Halfedge ij : f_A.adjacentHalfedges())
                {
                    Vertex vi_A = ij.tailVertex();
                    Vertex vj_A = ij.tipVertex();

                    size_t src_AxB = productMesh.vertexIndex(vi_A, v_B);
                    size_t dst_AxB = productMesh.vertexIndex(vj_A, v_B);

                    std::complex<double> rho = correspondenceConnection_A[ij];
                    double xi_ij = std::arg(psi(dst_AxB) / (rho * psi(src_AxB)));
                    index_twoform += xi_ij;

                    singularity_locator_aux_data(h_count) = std::abs(psi(src_AxB));
                    singularity_locator_aux_data(h_count + 3) = xi_ij;

                    h_count++;
                }

                singularity_locator_aux_data(6) = correspondenceCurvature_A[f_A];
                index_twoform += correspondenceCurvature_A[f_A];
                index_twoform /= 2. * M_PI;

                int degree = std::rint(index_twoform);
                if (degree != 0 && degree != +1)
                    std::cerr << "Found a singularity on face " << f_A << " x " << v_B << " with degree = " << degree << std::endl;

                if (degree != 0)
                {
                    found = true;

                    Vector3 barycoords = locateSingularity( singularity_locator_aux_data(3), singularity_locator_aux_data(4), singularity_locator_aux_data(5),
                        singularity_locator_aux_data(6), singularity_locator_aux_data(0), singularity_locator_aux_data(1), singularity_locator_aux_data(2));
                    pos_B[v_B] = SurfacePoint(f_A, barycoords);
                }
            }

            if (!found)
                std::cerr << "Failed to find a point where the vertex " << v_B << " is mapped to inside of A" << std::endl;
        }


        return std::make_pair(pos_A, pos_B);
    }

    std::pair<EdgeData<std::vector<SurfacePoint>>, EdgeData<std::vector<SurfacePoint>>> MinimalMatchingSolver::extractEdgeCorrespondence()
    {
        auto &psi = currentSection;

        EdgeData<std::vector<SurfacePoint>> edgePath_A(mesh_A);
        EdgeData<std::vector<SurfacePoint>> edgePath_B(mesh_B);

        auto [pos_A, pos_B] = extractCorrespondence();

        auto edgeEdgeIntersection = [&](Edge e_A, Edge e_B)
        {
            SurfacePoint p{};

            Halfedge ij_A = e_A.halfedge();
            Vertex i_A = ij_A.tailVertex();
            Vertex j_A = ij_A.tipVertex();

            Halfedge xy_B = e_B.halfedge();
            Vertex x_B = xy_B.tailVertex();
            Vertex y_B = xy_B.tipVertex();

            double index_twoform = 0;

            size_t i_AxB = productMesh.vertexIndex(i_A, x_B);
            size_t j_AxB = productMesh.vertexIndex(j_A, x_B);
            size_t k_AxB = productMesh.vertexIndex(j_A, y_B);
            size_t l_AxB = productMesh.vertexIndex(i_A, y_B);

            std::complex<double> rho;
            double xi;

            // ij_A
            rho = correspondenceConnection_A[ij_A];
            xi = std::arg(psi(j_AxB) / (rho * psi(i_AxB)));
            index_twoform += xi;

            // xy_B
            rho = correspondenceConnection_B[xy_B];
            xi = std::arg(psi(k_AxB) / (rho * psi(j_AxB)));
            index_twoform += xi;

            // -ij_A
            rho = correspondenceConnection_A[ij_A].conj();
            xi = std::arg(psi(l_AxB) / (rho * psi(k_AxB)));
            index_twoform += xi;

            // -xy_B
            rho = correspondenceConnection_B[xy_B].conj();
            xi = std::arg(psi(i_AxB) / (rho * psi(l_AxB)));
            index_twoform += xi;

            index_twoform /= 2. * M_PI;

            int degree = std::round(index_twoform);
            if (degree != 0)
                std::cerr << "Found a singularity on edge-edge face " << e_A << " x " << e_B << " with degree = " << degree << std::endl;

            if (degree == 0) return SurfacePoint();

            Vector2 rij = correspondenceConnection_A[ij_A];
            Vector2 rxy = correspondenceConnection_B[xy_B];

            Vector2 z0 = Vector2::fromComplex(psi(i_AxB));
            Vector2 z1 = Vector2::fromComplex(psi(j_AxB));
            Vector2 z2 = Vector2::fromComplex(psi(k_AxB));
            Vector2 z3 = Vector2::fromComplex(psi(l_AxB));

            Vector2 st = locateEdgeEdgeSingularity(rij, rxy, z0, z1, z2, z3);

            double t = st.y;
            p = SurfacePoint(e_B, t);

            return p;
        };

        for (Edge e_A : mesh_A.edges())
        {
            Vertex i_A = e_A.firstVertex();
            Vertex j_A = e_A.secondVertex();

            SurfacePoint p_i = pos_A[i_A];
            SurfacePoint p_j = pos_A[j_A];

            std::vector<SurfacePoint> path;
            path.emplace_back(p_i);

            if (p_i.face == p_j.face)
            {
                path.emplace_back(p_j);
                edgePath_A[e_A] = path;
                continue;
            }

            Face f = p_i.face;

            // Find the first intersection
            SurfacePoint p;
            Halfedge h_p;

            for (Edge e_B : f.adjacentEdges())
            {
                p = edgeEdgeIntersection(e_A,e_B);
                if ( p != SurfacePoint() )
                {
                    h_p = e_B.halfedge();
                    break;
                }
            }

            if (p == SurfacePoint())
            {
                std::cerr << "Failed to find a first intersection" << std::endl;
                continue;
            }

            path.emplace_back(p);

            // Continue looking for intersections until we reach the endpoint
            if (h_p.face() == f)
                h_p = h_p.twin();

            while (h_p.face() != p_j.face)
            {
                f = h_p.face();
                for (Edge e_B : f.adjacentEdges())
                {
                    if (e_B == h_p.edge()) continue;

                    p = edgeEdgeIntersection(e_A,e_B);
                    if ( p != SurfacePoint() )
                    {
                        std::cerr << "Face " << f << std::endl;
                        h_p = e_B.halfedge();
                        break;
                    }
                }

                if (p == SurfacePoint())
                {
                    std::cerr << "Failed to find an edge-edge intersection" << std::endl;
                    break;
                }

                path.emplace_back(p);

                if (h_p.face() == f)
                    h_p = h_p.twin();
            }

            path.emplace_back(p_j);
            edgePath_A[e_A] = path;

        }

        return std::make_pair(edgePath_A, edgePath_B);
    }


    double MinimalMatchingSolver::dirichletEnergy(geometrycentral::Vector<std::complex<double>>& psi)
    {
        auto Z = psi.reshaped(productMesh.nV_A, productMesh.nV_B);

        double e = 0;

        e += 0.5 * ((Z * massMatrix_B.conjugate()).adjoint() * (connectionLaplacian_A * Z)).trace().real();
        e += 0.5 * ((Z * connectionLaplacian_B.conjugate()).adjoint() * (massMatrix_A * Z)).trace().real();

        if (e < 0) std::cerr << "SOMETHING IS CATASTROPICALLY WRONG WITH THE DIRICHLET ENERGY, SINCE IT IS NEGATIVE: " << e << std::endl;
        return e;
    }

    double MinimalMatchingSolver::dirichletEnergy(DenseMatrix<std::complex<double>>& psi)
    {
        double e = 0;

        e += 0.5 * ((psi * massMatrix_B.conjugate()).adjoint() * (connectionLaplacian_A * psi)).trace().real();
        e += 0.5 * ((psi * connectionLaplacian_B.conjugate()).adjoint() * (massMatrix_A * psi)).trace().real();

        if (e < 0) std::cerr << "SOMETHING IS CATASTROPICALLY WRONG WITH THE DIRICHLET ENERGY, SINCE IT IS NEGATIVE: " << e << std::endl;
        return e;
    }

    geometrycentral::Vector<std::complex<double>> MinimalMatchingSolver::dirichletEnergyGradient(geometrycentral::Vector<std::complex<double>>& psi)
    {
        auto Z = psi.reshaped(productMesh.nV_A, productMesh.nV_B);
        DenseMatrix<std::complex<double>> g = (connectionLaplacian_A * Z * massMatrix_B.transpose() + massMatrix_A * Z * connectionLaplacian_B.transpose());
        return g.reshaped();
    }

    DenseMatrix<std::complex<double>> MinimalMatchingSolver::dirichletEnergyGradient(DenseMatrix<std::complex<double>>& psi)
    {
        return connectionLaplacian_A * psi * massMatrix_B.transpose() + massMatrix_A * psi * connectionLaplacian_B.transpose();
    }

    double MinimalMatchingSolver::doubleWellPotential(geometrycentral::Vector<std::complex<double>>& psi)
    {
        double e = 0;

        if (options.doubleWellType == VERTEX)
        {
            for (Vertex v_A : mesh_A.vertices())
            {
                double m_A = geometry_A.vertexDualAreas[v_A];
                for (Vertex v_B : mesh_B.vertices())
                {
                    double m_B = geometry_B.vertexDualAreas[v_B];
                    double lumped_mass = m_A * m_B;

                    size_t v = productMesh.vertexIndex(v_A, v_B);
                    double modulus = std::abs(psi(v));

                    size_t v_AxB = productMesh.vertexIndex(v_A, v_B);
                    e += lumped_mass * pow(singularityPinningPotential(geometry_A.vertexIndices[v_A], geometry_B.vertexIndices[v_B]) - modulus * modulus, 2);
                }
            }
        }

        if (options.doubleWellType == EDGE)
        {
            for (Vertex v_A : mesh_A.vertices())
            {
                double m_A = geometry_A.vertexDualAreas[v_A];
                for (Edge e_B : mesh_B.edges())
                {
                    Halfedge ij_B = e_B.halfedge();
                    Halfedge ji_B = ij_B.twin();
                    double m_B = (geometry_B.faceAreas[ij_B.face()] + geometry_B.faceAreas[ji_B.face()]) / 3.;
                    double lumped_mass = m_A * m_B;

                    Vertex vi_B = ij_B.tailVertex();
                    Vertex vj_B = ij_B.tipVertex();

                    size_t vi = productMesh.vertexIndex(v_A, vi_B);
                    size_t vj = productMesh.vertexIndex(v_A, vj_B);

                    size_t ivi_A = geometry_A.vertexIndices[v_A];
                    size_t ivi_B = geometry_B.vertexIndices[vi_B];

                    size_t ivj_A = geometry_A.vertexIndices[v_A];
                    size_t ivj_B = geometry_B.vertexIndices[vj_B];

                    Vector2 psi_i = Vector2::fromComplex(psi(vi));
                    Vector2 psi_j = Vector2::fromComplex(psi(vj));

                    Vector2 psi_ij = 0.5 * psi_i + 0.5 * correspondenceConnection_B[ji_B] * psi_j;

                    double V_ij = 0.5 * (singularityPinningPotential(ivi_A,ivi_B) + singularityPinningPotential(ivj_A,ivj_B));
                    e += lumped_mass * pow(V_ij - psi_ij.norm2(), 2);
                }
            }

            for (Vertex v_B : mesh_B.vertices())
            {
                double m_B = geometry_B.vertexDualAreas[v_B];
                for (Edge e_A : mesh_A.edges())
                {
                    Halfedge ij_A = e_A.halfedge();
                    Halfedge ji_A = ij_A.twin();
                    double m_A = (geometry_A.faceAreas[ij_A.face()] + geometry_A.faceAreas[ji_A.face()]) / 3.;
                    double lumped_mass = m_A * m_B;

                    Vertex vi_A = ij_A.tailVertex();
                    Vertex vj_A = ij_A.tipVertex();

                    size_t vi = productMesh.vertexIndex(vi_A, v_B);
                    size_t vj = productMesh.vertexIndex(vj_A, v_B);

                    size_t ivi_A = geometry_A.vertexIndices[vi_A];
                    size_t ivi_B = geometry_B.vertexIndices[v_B];

                    size_t ivj_A = geometry_A.vertexIndices[vi_A];
                    size_t ivj_B = geometry_B.vertexIndices[v_B];


                    Vector2 psi_i = Vector2::fromComplex(psi(vi));
                    Vector2 psi_j = Vector2::fromComplex(psi(vj));

                    Vector2 psi_ij = 0.5 * psi_i + 0.5 * correspondenceConnection_A[ji_A] * psi_j;

                    double V_ij = 0.5 * (singularityPinningPotential(ivi_A,ivi_B) + singularityPinningPotential(ivj_A,ivj_B));
                    e += lumped_mass * pow(V_ij - psi_ij.norm2(), 2);
                }
            }
        }

        return e / 4.;
    }

    geometrycentral::Vector<std::complex<double>> MinimalMatchingSolver::doubleWellPotentialGradient(geometrycentral::Vector<std::complex<double>>& psi)
    {
        geometrycentral::Vector<std::complex<double>> gradient(psi.size());
        gradient.setZero();
        double e = 0;

        if (options.doubleWellType == VERTEX)
        {
            for (Vertex v_A : mesh_A.vertices())
            {
                double m_A = geometry_A.vertexDualAreas[v_A];
                for (Vertex v_B : mesh_B.vertices())
                {
                    double m_B = geometry_B.vertexDualAreas[v_B];
                    double lumped_mass = m_A * m_B;

                    size_t v = productMesh.vertexIndex(v_A, v_B);
                    double modulus = std::abs(psi(v));

                    double Vv = singularityPinningPotential(geometry_A.vertexIndices[v_A], geometry_B.vertexIndices[v_B]);

                    size_t v_AxB = productMesh.vertexIndex(v_A, v_B);
                    e += lumped_mass * pow(Vv - modulus * modulus, 2);

                    gradient(v) = -lumped_mass * (Vv - modulus * modulus) * psi(v);
                }
            }
        }

        if (options.doubleWellType == EDGE)
        {
            for (Vertex v_A : mesh_A.vertices())
            {
                double m_A = geometry_A.vertexDualAreas[v_A];
                for (Edge e_B : mesh_B.edges())
                {
                    Halfedge ij_B = e_B.halfedge();
                    Halfedge ji_B = ij_B.twin();
                    double m_B = (geometry_B.faceAreas[ij_B.face()] + geometry_B.faceAreas[ji_B.face()]) / 3.;
                    double lumped_mass = m_A * m_B;

                    Vertex vi_B = ij_B.tailVertex();
                    Vertex vj_B = ij_B.tipVertex();

                    size_t vi = productMesh.vertexIndex(v_A, vi_B);
                    size_t vj = productMesh.vertexIndex(v_A, vj_B);

                    size_t ivi_A = geometry_A.vertexIndices[v_A];
                    size_t ivi_B = geometry_B.vertexIndices[vi_B];

                    size_t ivj_A = geometry_A.vertexIndices[v_A];
                    size_t ivj_B = geometry_B.vertexIndices[vj_B];

                    Vector2 psi_i = Vector2::fromComplex(psi(vi));
                    Vector2 psi_j = Vector2::fromComplex(psi(vj));

                    Vector2 psi_ij = 0.5 * psi_i + 0.5 * correspondenceConnection_B[ji_B] * psi_j;
                    Vector2 psi_ji = correspondenceConnection_B[ij_B] * psi_ij;

                    double V_ij = 0.5 * (singularityPinningPotential(ivi_A,ivi_B) + singularityPinningPotential(ivj_A,ivj_B));

                    e += lumped_mass * pow(V_ij - psi_ij.norm2(), 2);

                    gradient(vi) -= lumped_mass * (V_ij - psi_ij.norm2()) * std::complex<double>(psi_ij) / 2.;
                    gradient(vj) -= lumped_mass * (V_ij - psi_ij.norm2()) * std::complex<double>(psi_ji) / 2.;
                }
            }

            for (Vertex v_B : mesh_B.vertices())
            {
                double m_B = geometry_B.vertexDualAreas[v_B];
                for (Edge e_A : mesh_A.edges())
                {
                    Halfedge ij_A = e_A.halfedge();
                    Halfedge ji_A = ij_A.twin();
                    double m_A = (geometry_A.faceAreas[ij_A.face()] + geometry_A.faceAreas[ji_A.face()]) / 3.;
                    double lumped_mass = m_A * m_B;

                    Vertex vi_A = ij_A.tailVertex();
                    Vertex vj_A = ij_A.tipVertex();

                    size_t vi = productMesh.vertexIndex(vi_A, v_B);
                    size_t vj = productMesh.vertexIndex(vj_A, v_B);

                    size_t ivi_A = geometry_A.vertexIndices[vi_A];
                    size_t ivi_B = geometry_B.vertexIndices[v_B];

                    size_t ivj_A = geometry_A.vertexIndices[vi_A];
                    size_t ivj_B = geometry_B.vertexIndices[v_B];

                    Vector2 psi_i = Vector2::fromComplex(psi(vi));
                    Vector2 psi_j = Vector2::fromComplex(psi(vj));

                    Vector2 psi_ij = 0.5 * psi_i + 0.5 * correspondenceConnection_A[ji_A] * psi_j;
                    Vector2 psi_ji = correspondenceConnection_A[ij_A] * psi_ij;

                    double V_ij = 0.5 * (singularityPinningPotential(ivi_A,ivi_B) + singularityPinningPotential(ivj_A,ivj_B));

                    e += lumped_mass * pow(V_ij - psi_ij.norm2(), 2);

                    gradient(vi) -= lumped_mass * (V_ij - psi_ij.norm2()) * std::complex<double>(psi_ij) / 2.;
                    gradient(vj) -= lumped_mass * (V_ij - psi_ij.norm2()) * std::complex<double>(psi_ji) / 2.;
                }
            }
        }

        return gradient;
    }

    double MinimalMatchingSolver::doubleWellPotential(DenseMatrix<std::complex<double>>& psi)
    {
        DenseMatrix<double> U = psi.cwiseAbs2() - singularityPinningPotential;
        double e = (U.transpose() * (massMatrixReal_A * U * massMatrixReal_B)).trace();
        return e / 4.;
    }

    DenseMatrix<std::complex<double>> MinimalMatchingSolver::doubleWellPotentialGradient(DenseMatrix<std::complex<double>>& psi)
    {
        DenseMatrix<double> U = psi.cwiseAbs2() - singularityPinningPotential;
        double e = (U.transpose() * (massMatrixReal_A * U * massMatrixReal_B)).trace();

        DenseMatrix<double> grad_R = massMatrixReal_A * U * massMatrixReal_B;
        return grad_R.cast <std::complex<double>> ().array() * psi.array();
    }


    double MinimalMatchingSolver::ginzburgLandauEnergy(double epsilon, geometrycentral::Vector<std::complex<double>>& psi)
    {
        return dirichletEnergy(psi) + epsilon * doubleWellPotential(psi);
    }

    geometrycentral::Vector<std::complex<double>> MinimalMatchingSolver::ginzburgLandauGradient(double epsilon, geometrycentral::Vector<std::complex<double>>& psi)
    {
        return dirichletEnergyGradient(psi) + epsilon * doubleWellPotentialGradient(psi);
    }

    double MinimalMatchingSolver::ginzburgLandauEnergy(double epsilon, DenseMatrix<std::complex<double>>& psi)
    {
        return dirichletEnergy(psi) + epsilon * doubleWellPotential(psi);
    }

    DenseMatrix<std::complex<double>> MinimalMatchingSolver::ginzburgLandauGradient(double epsilon, DenseMatrix<std::complex<double>>& psi)
    {
        return dirichletEnergyGradient(psi) + epsilon * doubleWellPotentialGradient(psi);
    }


    static int monitorProgress(void *instance,
                       const geometrycentral::Vector<double> &x,
                       const geometrycentral::Vector<double> &g,
                       const double fx,
                       const double step,
                       const int k,
                       const int ls)
    {
        MinimalMatchingSolver *self = (MinimalMatchingSolver *)instance;
        std::cout << std::setprecision(4)
                  << "================================" << std::endl
                  << "Iteration: " << k << std::endl
                  << "EPSILON: " << self->EPSILON << std::endl
                  << "Function Value: " << fx << std::endl
                  << "Gradient Inf Norm: " << g.cwiseAbs().maxCoeff() << std::endl;
        return 0;
    }


    // Optimize the section using L-BFGS
    void MinimalMatchingSolver::optimize(double epsilon)
    {
        auto &psi = currentSection;
        EPSILON = epsilon * EPSILON0;

        bool VERBOSE = true;

        double final_energy;
        geometrycentral::Vector<double> x = geometrycentral::complexToReal(psi);

        /* Set the minimization parameters */
        lbfgs::lbfgs_parameter_t params;
        params.g_epsilon = 1.0e-6;
        params.past = 3;
        params.delta = 1.0e-8;
        params.max_iterations = options.maxIterations;

        /* Start minimization */
        START_TIMING(LBFGS)
        int ret = lbfgs::lbfgs_optimize(x,
                                        final_energy,
                                        costFunction,
                                        nullptr,
                                        monitorProgress,
                                        this,
                                        params);
        FINISH_TIMING_PRINT(LBFGS)

        /* Report the result. */
        std::cout << std::setprecision (4)
                  << "================================" << std::endl
                  << "L-BFGS Optimization Returned: " << ret << std::endl
                  << "Minimized Cost: " << final_energy << std::endl;

        psi = geometrycentral::realToComplex(x);
    }

    double MinimalMatchingSolver::costFunction(void* instance, const geometrycentral::Vector<double>& x, geometrycentral::Vector<double>& g)
    {
        MinimalMatchingSolver *self = (MinimalMatchingSolver*)instance;

        if (self->options.vectorized)
        {
            geometrycentral::Vector<std::complex<double>> psi = geometrycentral::realToComplex(x);
            Vector<std::complex<double>> gM = self->ginzburgLandauGradient(self->EPSILON, psi);
            g = geometrycentral::complexToReal(gM);
            return self->ginzburgLandauEnergy(self->EPSILON, psi);
        } else
        {
            geometrycentral::Vector<std::complex<double>> psi_v = geometrycentral::realToComplex(x);
            DenseMatrix<std::complex<double>> psi = psi_v.reshaped(self->mesh_A.nVertices(), self->mesh_B.nVertices());
            Vector<std::complex<double>> gM = self->ginzburgLandauGradient(self->EPSILON, psi).reshaped();
            g = geometrycentral::complexToReal(gM);
            return self->ginzburgLandauEnergy(self->EPSILON, psi);
        }
    }


    // Optimize the section with epsilon scheduling
    void MinimalMatchingSolver::optimize(int eps_steps)
    {
        double epsilon = EPSILON0;

        for (int step = 0; step < eps_steps; ++step)
        {
            optimize(epsilon);
            epsilon *= 2;
        }
    }

    // Set Landmarks
    void MinimalMatchingSolver::setLandmarks(const std::vector<Landmark>& landmarks_)
    {
        landmarks = landmarks_;
        computeSingularityPinningPotential();
    }

    void MinimalMatchingSolver::computeSingularityPinningPotential()
    {
        auto &V = singularityPinningPotential;

        V = DenseMatrix<double>(mesh_A.nVertices(), mesh_B.nVertices());
        V.setConstant(std::numeric_limits<double>::infinity());

        if (landmarks.empty())
        {
            V.setConstant(1);
            return;
        }

        HeatMethodDistanceSolver distance_solver_A(geometry_A);
        HeatMethodDistanceSolver distance_solver_B(geometry_B);

        geometry_A.requireVertexIndices();
        geometry_B.requireVertexIndices();

            for (Landmark &l : landmarks)
            {
                VertexData<double> dist_a = distance_solver_A.computeDistance(l.pt_A);
                VertexData<double> dist_b = distance_solver_B.computeDistance(l.pt_B);

                for (Vertex v_A : mesh_A.vertices())
                    for (Vertex v_B : mesh_B.vertices())
                    {
                        size_t iV_A = geometry_A.vertexIndices[v_A];
                        size_t iV_B = geometry_B.vertexIndices[v_B];

                        double dist2 = dist_a[v_A] * dist_a[v_A] + dist_b[v_B] * dist_b[v_B];
                        dist2 *= options.doubleWellCoeff;
                        double V0 = 1. - exp(-dist2);
                        V(iV_A, iV_B) = fmin(V(iV_A, iV_B), V0);
                    }
            }

        V /= V.maxCoeff();
    }
}
