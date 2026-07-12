#pragma once

#include "geometrycentral/utilities/vector2.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include "geometrycentral/numerical/linear_algebra_types.h"

using namespace geometrycentral::surface;
using geometrycentral::SparseMatrix;
using geometrycentral::Vector2;

namespace minimalmatching
{

SparseMatrix<std::complex<double>>
complexBundleMassMatrix(ManifoldSurfaceMesh& mesh,
                        IntrinsicGeometryInterface& geom,
                        const HalfedgeData<Vector2>& connection,
                        const FaceData<double>& curvature, bool transpose = false);

SparseMatrix<std::complex<double>>
complexBundleLaplacian(ManifoldSurfaceMesh& mesh,
                       IntrinsicGeometryInterface& geom,
                       const HalfedgeData<Vector2>& connection,
                       const FaceData<double>& curvature, bool transpose = false, double s = 0);

SparseMatrix<std::complex<double>>
complexBundleMassMatrixDEC(ManifoldSurfaceMesh& mesh,
                        IntrinsicGeometryInterface& geom,
                        const HalfedgeData<Vector2>& connection,
                        const FaceData<double>& curvature, bool transpose = false);

    SparseMatrix<std::complex<double>>
    complexBundleLaplacianDEC(ManifoldSurfaceMesh& mesh,
                           IntrinsicGeometryInterface& geom,
                           const HalfedgeData<Vector2>& connection,
                           const FaceData<double>& curvature, bool transpose = false);

}