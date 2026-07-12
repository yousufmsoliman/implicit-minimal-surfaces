#include "product_mesh.h"

namespace minimalmatching
{
    ProductMesh::ProductMesh(SurfaceMesh& mesh_A, SurfaceMesh& mesh_B):
        mesh_A(mesh_A), mesh_B(mesh_B)
    {
        vInd_A = mesh_A.getVertexIndices();
        vInd_B = mesh_B.getVertexIndices();

        eInd_A = mesh_A.getEdgeIndices();
        eInd_B = mesh_B.getEdgeIndices();

        fInd_A = mesh_A.getFaceIndices();
        fInd_B = mesh_B.getFaceIndices();

        nV_A = mesh_A.nVertices();
        nV_B = mesh_B.nVertices();

        nE_A = mesh_A.nEdges();
        nE_B = mesh_B.nEdges();

        nF_A = mesh_A.nFaces();
        nF_B = mesh_B.nFaces();
    }

    size_t ProductMesh::nCells(int dimension)
    {
        switch (dimension)
        {
        case 0:
            return mesh_A.nVertices() * mesh_B.nVertices();
        case 1:
            return mesh_A.nVertices() * mesh_B.nEdges() + mesh_A.nEdges() * mesh_B.nVertices();
        case 2:
            return mesh_A.nVertices() * mesh_B.nFaces() + mesh_A.nEdges() * mesh_B.nEdges() + mesh_A.nFaces() * mesh_B.
                nVertices();
        case 3:
            return mesh_A.nEdges() * mesh_B.nFaces() + mesh_A.nFaces() * mesh_B.nEdges();
        case 4:
            return mesh_A.nFaces() * mesh_B.nFaces();
        default:
            return 0;
        }
    }

    size_t ProductMesh::vertexIndex(Vertex v_A, Vertex v_B)
    {
        return vInd_A[v_A] + vInd_B[v_B] * nV_A;
    }

    std::tuple<Vertex,Vertex> ProductMesh::vertex(size_t v_index)
    {
        size_t v_A = v_index % nV_A;
        size_t v_B = v_index / nV_A;

        return std::make_tuple(mesh_A.vertex(v_A), mesh_B.vertex(v_B));
    }

    size_t ProductMesh::edgeIndex(Vertex v_A, Edge e_B)
    {
        return vInd_A[v_A] + eInd_B[e_B] * nV_A;
    }

    size_t ProductMesh::edgeIndex(Edge e_A, Vertex v_B)
    {
        return nV_A * nE_B + eInd_A[e_A] + vInd_B[v_B] * nE_A;
    }

    size_t ProductMesh::faceIndex(Vertex v_A, Face f_B)
    {
        return vInd_A[v_A] + fInd_B[f_B] * nV_A;
    }

    size_t ProductMesh::faceIndex(Edge e_A, Edge e_B)
    {
        return nV_A * nF_B + eInd_A[e_A] + eInd_B[e_B] * nE_A;
    }

    size_t ProductMesh::faceIndex(Face f_A, Vertex v_B)
    {
        return nV_A * nF_B + nE_A * nE_B + fInd_A[f_A] + vInd_B[v_B] * nF_A;
    }




}
