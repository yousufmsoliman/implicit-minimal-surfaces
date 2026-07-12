#pragma once

#include "topology/product_mesh.h"

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/surface_point.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"

#include "geometrycentral/surface/vertex_position_geometry.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;
using geometrycentral::Vector3;
using geometrycentral::Vector2;
using geometrycentral::DenseMatrix;
using geometrycentral::SparseMatrix;
using geometrycentral::SquareSolver;

namespace minimalmatching
{
    enum DiscretizationType
    {
        DEC, FEM
    };

    enum BundleType
    {
        SKYSCRAPER = 0,
        SPIN
    };

    enum DoubleWellDiscretizationType
    {
        VERTEX = 0,
        EDGE,
        FACE,
        CELL
    };

    enum CurvatureType
    {
        GAUSSIAN,
        AREA
    };

    struct Landmark {
        std::vector<SurfacePoint> pt_A;
        std::vector<SurfacePoint> pt_B;

        double weight = 1.0;
    };

    struct MinimalMatchingOptions {
        DiscretizationType laplacianType = FEM;
        DiscretizationType massMatrixType = FEM;

        CurvatureType curvatureAType = GAUSSIAN;
        CurvatureType curvatureBType = GAUSSIAN;

        BundleType bundleType = SPIN;

        DoubleWellDiscretizationType doubleWellType = EDGE;

        double doubleWellCoeff = 1;
        double trivialConnectionBlendingWeight = 1;

        int smoothingIterations = 0;
        int maxIterations = 1000;

        bool vectorized = false;
    };


    class MinimalMatchingSolver
    {
    public:
        MinimalMatchingSolver(ManifoldSurfaceMesh& mesh_A,
                              ManifoldSurfaceMesh& mesh_B,
                              IntrinsicGeometryInterface& geometry_A,
                              IntrinsicGeometryInterface& geometry_B,
                              VertexPositionGeometry& extrinsic_geometry_A,
                              VertexPositionGeometry& extrinsic_geometry_B);

        void optimize(int eps_steps);
        void optimize(double epsilon);

        void buildSpinConnections();
        void buildSkyscraperConnections();

        void closestPointInitialization(VertexData<Vector3> &pos_A, VertexData<Vector3> &pos_B);
        void setLandmarks(const std::vector<Landmark> &landmarks);

        std::pair<VertexData<SurfacePoint>, VertexData<SurfacePoint>> extractCorrespondence();
        std::pair<EdgeData<std::vector<SurfacePoint>>, EdgeData<std::vector<SurfacePoint>>> extractEdgeCorrespondence();

        MinimalMatchingOptions options;
        double EPSILON = 0.001;
        double EPSILON0 = 1;

        // Vectorized representation
        geometrycentral::Vector<std::complex<double>> initializationSection, currentSection;
        // Matrix representation
        DenseMatrix<std::complex<double>> initializationSectionMatrix, currentSectionMatrix;

        size_t vertexIndex(Vertex v_A, Vertex v_B)
        {
            return productMesh.vertexIndex(v_A, v_B);
        }

    protected:
        // Base two-dimensional geometries
        ManifoldSurfaceMesh &mesh_A, &mesh_B;
        IntrinsicGeometryInterface &geometry_A, &geometry_B;
        VertexPositionGeometry &extrinsic_geometry_A, &extrinsic_geometry_B;

        // 4D cell complex representing A x B
        ProductMesh productMesh;

        // Represent connections on the product mesh by connections on each slice of A and B separately
        // The closest point connection describes the connection whose singularities try to lie at the points specified by the vertex -> face map
        VertexData<HalfedgeData<Vector2>> closestPointConnection_A;
        VertexData<HalfedgeData<Vector2>> closestPointConnection_B;
        VertexData<FaceData<double>> closestPointCurvature_A;
        VertexData<FaceData<double>> closestPointCurvature_B;

        HalfedgeData<Vector2> correspondenceConnection_A;
        HalfedgeData<Vector2> correspondenceConnection_B;
        FaceData<double> correspondenceCurvature_A;
        FaceData<double> correspondenceCurvature_B;

        VertexData<HalfedgeData<Vector2>> correspondenceConnection_AxB_A;
        VertexData<HalfedgeData<Vector2>> correspondenceConnection_AxB_B;
        Vector<double> correspondenceCurvature_AxB;


        SparseMatrix<std::complex<double>> connectionLaplacian_A, connectionLaplacian_B;
        SparseMatrix<std::complex<double>> massMatrix_A, massMatrix_B;
        SparseMatrix<double> massMatrixReal_A, massMatrixReal_B;

        SparseMatrix<std::complex<double>> connectionLaplacian_AxB;
        SparseMatrix<std::complex<double>> massMatrix_AxB;

        SparseMatrix<double> dδ_A, dδ_B;
        std::unique_ptr<SquareSolver<double>> dδ_A_solver, dδ_B_solver;

        HalfedgeData<Vector2> leviCivita_A, leviCivita_B;
        HalfedgeData<Vector2> spinConnection_A, spinConnection_B;
        HalfedgeData<Vector2> skyscraperConnection_A, skyscraperConnection_B;
        FaceData<double> gaussCurvature_A, gaussCurvature_B;
        FaceData<double> spinCurvature_A, spinCurvature_B;
        FaceData<double> areaCurvature_A, areaCurvature_B;

        void buildProductConnectionLaplacian();

        // Ginzburg-Landau optimization
        double dirichletEnergy(geometrycentral::Vector<std::complex<double>> &psi);
        double doubleWellPotential(geometrycentral::Vector<std::complex<double>> &psi);
        double ginzburgLandauEnergy(double epsilon, geometrycentral::Vector<std::complex<double>> &psi);

        geometrycentral::Vector<std::complex<double>> dirichletEnergyGradient(geometrycentral::Vector<std::complex<double>> &psi);
        geometrycentral::Vector<std::complex<double>> doubleWellPotentialGradient(geometrycentral::Vector<std::complex<double>> &psi);
        geometrycentral::Vector<std::complex<double>> ginzburgLandauGradient(double epsilon, geometrycentral::Vector<std::complex<double>> &psi);

        double dirichletEnergy(DenseMatrix<std::complex<double>> &psi);
        double doubleWellPotential(DenseMatrix<std::complex<double>> &psi);
        double ginzburgLandauEnergy(double epsilon, DenseMatrix<std::complex<double>> &psi);

        DenseMatrix<std::complex<double>> dirichletEnergyGradient(DenseMatrix<std::complex<double>> &psi);
        DenseMatrix<std::complex<double>> doubleWellPotentialGradient(DenseMatrix<std::complex<double>> &psi);
        DenseMatrix<std::complex<double>> ginzburgLandauGradient(double epsilon, DenseMatrix<std::complex<double>> &psi);


        static double costFunction(void *instance,
                           const geometrycentral::Vector<double> &x,
                           geometrycentral::Vector<double> &g);

        // Landmarks via Singularity Pinning
        std::vector<Landmark> landmarks;
        DenseMatrix<double> singularityPinningPotential;

        void computeSingularityPinningPotential();
    };

    inline Vector3 point2triangle_projection(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        const Vector3 ab = b - a;
        const Vector3 ac = c - a;
        const Vector3 ap = p - a;

        const float d1 = dot(ab, ap);
        const float d2 = dot(ac, ap);
        if (d1 <= 0.f && d2 <= 0.f) return a; //#1

        const Vector3 bp = p - b;
        const float d3 = dot(ab, bp);
        const float d4 = dot(ac, bp);
        if (d3 >= 0.f && d4 <= d3) return b; //#2

        const Vector3 cp = p - c;
        const float d5 = dot(ab, cp);
        const float d6 = dot(ac, cp);
        if (d6 >= 0.f && d5 <= d6) return c; //#3

        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
        {
            const float v = d1 / (d1 - d3);
            return a + v * ab; //#4
        }

        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
        {
            const float v = d2 / (d2 - d6);
            return a + v * ac; //#5
        }

        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
        {
            const float v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + v * (c - b); //#6
        }

        const float denom = 1.f / (va + vb + vc);
        const float v = vb * denom;
        const float w = vc * denom;
        return a + v * ab + w * ac; //#0
    }

    inline double point2triangle_distance(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
    {
        return (p - point2triangle_projection(p, a,b,c)).norm();
    }

    // Pretty slow...
    inline std::tuple<Face,Vector3>
                closestPoint(Vector3 query_point,
                     ManifoldSurfaceMesh &target_mesh,
                     const VertexData<Vector3> &target_pos)
    {
        double min_dist = std::numeric_limits<double>::max();
        Face closest_face = target_mesh.face(0);
        Vector3 closest_point{0,0,0};

        for (Face f : target_mesh.faces())
        {
            Halfedge ij = f.halfedge(), jk = ij.next(), ki = jk.next();
            Vertex i = ij.vertex(), j = jk.vertex(), k = ki.vertex();

            Vector3 proj = point2triangle_projection(query_point, target_pos[i], target_pos[j], target_pos[k]);
            double dist = (proj - query_point).norm();

            if (dist < min_dist)
            {
                min_dist = dist;
                closest_face = f;
                closest_point = proj;
            }
        }

        return std::make_tuple(closest_face, closest_point);
    }

    inline Vector3 computeBarycentricCoords(const Vector3& A, const Vector3& B,
                                            const Vector3& C, const Vector3& P) {
        // Vectors representing the edges of the main triangle
        Vector3 v0 = B - A;
        Vector3 v1 = C - A;
        Vector3 v2 = P - A;

        // Compute dot products
        float dot00 = dot(v0, v0); // length squared of v0
        float dot01 = dot(v0, v1);
        float dot11 = dot(v1, v1); // length squared of v1
        float dot20 = dot(v2, v0);
        float dot21 = dot(v2, v1);

        // Compute denominator
        float denom = dot00 * dot11 - dot01 * dot01;

        // Check if the triangle is degenerate (denominator is zero or close to zero)
        if (std::abs(denom) < 1e-6) {
            // Handle degenerate triangle case (e.g., return invalid coordinates or an error)
            return Vector3(-1.0f, -1.0f, -1.0f); // Example error return
        }

        // Compute barycentric coordinates (u, v, w)
        // Here, u corresponds to beta, v corresponds to gamma, and w corresponds to alpha.
        float beta_u = (dot11 * dot20 - dot01 * dot21) / denom;
        float gamma_v = (dot00 * dot21 - dot01 * dot20) / denom;
        float alpha_w = 1.0f - beta_u - gamma_v;

        // The result is a vec3 with (alpha, beta, gamma)
        return Vector3(alpha_w, beta_u, gamma_v);
    }

    inline SurfacePoint closestSurfacePoint(Vector3 query_point, ManifoldSurfaceMesh &target_mesh, VertexData<Vector3> &embedding)
    {
        auto [f, b] = closestPoint(query_point, target_mesh, embedding);

        std::vector<Vertex> vc_A = {f.halfedge().vertex(), f.halfedge().next().vertex(), f.halfedge().next().next().vertex()};
        Vector3 bc = computeBarycentricCoords(embedding[vc_A[0]], embedding[vc_A[1]], embedding[vc_A[2]],
            b
        );

        return {f,bc};
    }

}
