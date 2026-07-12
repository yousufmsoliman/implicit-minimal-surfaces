#pragma once

#include "geometrycentral/utilities/vector3.h"
#include "geometrycentral/utilities/vector2.h"

using geometrycentral::Vector3;
using geometrycentral::Vector2;

namespace minimalmatching
{
    // Extracts the barycentric coordinates of a singularity in a face by
    // a homotopy continuation method
    Vector3
    locateSingularity(
        double omega_ij_0,
        double omega_jk_0,
        double omega_ki_0,

        double curvature_ijk_0,

        double z_i, double z_j, double z_k,

        int nSteps = 10
    );

    // Extracts the barycentric coordinates of a singularity in a (flat) edge-edge face by
    // solving a quadratic equation
    Vector2
    locateEdgeEdgeSingularity(
        Vector2 rij,
        Vector2 rxy,

        Vector2 zix, Vector2 zjx,
        Vector2 zjy, Vector2 ziy
    );

}