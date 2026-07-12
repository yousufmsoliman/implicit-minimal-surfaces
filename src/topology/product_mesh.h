#pragma once

#include "geometrycentral/surface/surface_mesh.h"

using namespace geometrycentral::surface;

namespace minimalmatching
{
    // A class to index variables in the product mesh
    class ProductMesh
    {
    public:
        ProductMesh(SurfaceMesh &mesh_A, SurfaceMesh &mesh_B);

        size_t nCells(int dimension);

        // Vertices
        size_t vertexIndex(Vertex v_A, Vertex v_B);
        std::tuple<Vertex,Vertex> vertex(size_t v_index);

        // Edges
        size_t edgeIndex(Edge e_A, Vertex v_B);
        size_t edgeIndex(Vertex v_A, Edge e_B);

        // Faces
        size_t faceIndex(Face f_A, Vertex v_B);
        size_t faceIndex(Edge e_A, Edge e_B);
        size_t faceIndex(Vertex v_A, Face f_B);

        VertexData<size_t> vInd_A, vInd_B;
        EdgeData<size_t> eInd_A, eInd_B;
        FaceData<size_t> fInd_A, fInd_B;

        size_t nV_A, nV_B;
        size_t nE_A, nE_B;
        size_t nF_A, nF_B;
    protected:
        SurfaceMesh &mesh_A, &mesh_B;
    };
}
