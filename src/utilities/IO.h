#pragma once

#include "geometrycentral/surface/simple_polygon_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

inline std::string extractDirectory(const std::string &filepath)
{
    std::string dir(filepath.begin(), filepath.begin() + filepath.find_last_of("/") + 1);
    return dir;
}

inline void writeMeshWithNormals(const VertexPositionGeometry& geom,
    const VertexData<Vector3>& N,
    std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }

    // Write texture coords
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 N_uv = N[iV];
        out << "vt " << N_uv.x << " " << N_uv.y << " " << N_uv.z << std::endl;
    }

    // Write faces
    for (Face f : mesh.faces()) {
        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iV = h.tailVertex().getIndex();
            out << " " << (iV + 1) << "/" << (iV + 1);
        }
        out << std::endl;
    }
}

inline void writeMeshWithNormals(const VertexPositionGeometry& geom,
    const FaceData<Vector3>& N,
    std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }

    // Write texture coords
    for (size_t iF = 0; iF < mesh.nVertices(); iF++) {
        Vector3 N_uv = N[iF];
        out << "vt " << N_uv.x << " " << N_uv.y << " " << N_uv.z << std::endl;
    }

    // Write faces
    for (Face f : mesh.faces()) {
        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iF = h.face().getIndex();
            out << " " << (iF + 1) << "/" << (iF + 1);
        }
        out << std::endl;
    }
}

void writeMeshWithProjectiveTextureCoords(const VertexPositionGeometry& geom,
                                          const VertexData<Vector3>& pUV,
                                          std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }


    // Write texture coords
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        out << "vt " << pUV[iV].x << " " << pUV[iV].y << " " << pUV[iV].z
            << std::endl;
    }

    // Write faces
    for (Face f : mesh.faces()) {
        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iV = h.tailVertex().getIndex();
            out << " " << (iV + 1) << "/" << (iV + 1);
        }
        out << std::endl;
    }
}


void writeMeshWithProjectiveTextureCoords(const VertexPositionGeometry& geom,
                                          const VertexData<Vector3>& pUV,
                                          std::string filename) {

    std::cout << "Writing mesh to: " << filename << std::endl;

    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("failed to open output file " + filename);
    }
    writeMeshWithProjectiveTextureCoords(geom, pUV, outFile);
    outFile.close();
}

void writeMeshWithOrdinaryTextureCoords(const VertexPositionGeometry& geom,
                                        const VertexData<Vector3>& pUV,
                                        std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }


    // Write texture coords
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector2 uv{pUV[iV].x / pUV[iV].z, pUV[iV].y / pUV[iV].z};
        out << "vt " << uv.x << " " << uv.y << " " << 1
            << std::endl;
    }

    // Write faces
    for (Face f : mesh.faces()) {
        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iV = h.tailVertex().getIndex();
            out << " " << (iV + 1) << "/" << (iV + 1);
        }
        out << std::endl;
    }

}

void writeMeshWithOrdinaryTextureCoords(
    const VertexPositionGeometry& mesh,
    const VertexData<Vector3>& pUV, std::string filename) {

    std::cout << "Writing mesh to: " << filename << std::endl;

    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("failed to open output file " + filename);
    }

    writeMeshWithOrdinaryTextureCoords(mesh, pUV, outFile);
    outFile.close();
}

void writeMeshWithFaceScalar(const VertexPositionGeometry& geom,
                                        const FaceData<double>& func,
                                        std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
    }


    // Write faces
    for (Face f : mesh.faces()) {
        out << "vt " << func[f] << " " << 0 << " " << 0 << std::endl;

        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iV = h.tailVertex().getIndex();
            out << " " << (iV + 1) << "/" << (f.getIndex() + 1);
        }
        out << std::endl;
    }

}

void writeMeshWithFaceScalar(
    const VertexPositionGeometry& mesh,
    const FaceData<double>& func, std::string filename) {

    std::cout << "Writing mesh to: " << filename << std::endl;

    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("failed to open output file " + filename);
    }

    writeMeshWithFaceScalar(mesh, func, outFile);
    outFile.close();
}

void writeMeshWithVertexScalar(const VertexPositionGeometry& geom,
                                        const VertexData<double>& func,
                                        std::ostream& out) {

    // Write header
    auto &mesh = geom.mesh;

    out << "#  vertices: " << mesh.nVertices() << std::endl;
    out << "#     faces: " << mesh.nFaces() << std::endl;
    out << std::endl;

    // Write vertices
    for (size_t iV = 0; iV < mesh.nVertices(); iV++) {
        Vector3 p = geom.vertexPositions[iV];
        out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
        out << "vt " << func[iV] << " " << 0 << " " << 0 << std::endl;
    }


    // Write faces
    for (Face f : mesh.faces()) {
        out << "f";
        for (Halfedge h : f.adjacentHalfedges() ) {
            size_t iV = h.tailVertex().getIndex();
            out << " " << (iV + 1) << "/" << (iV + 1);
        }
        out << std::endl;
    }

}

void writeMeshWithVertexScalar(
    const VertexPositionGeometry& mesh,
    const VertexData<double>& func, std::string filename) {

    std::cout << "Writing mesh to: " << filename << std::endl;

    std::ofstream outFile(filename);
    if (!outFile) {
        throw std::runtime_error("failed to open output file " + filename);
    }

    writeMeshWithVertexScalar(mesh, func, outFile);
    outFile.close();
}