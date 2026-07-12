#include <filesystem>

#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"

#include "matching/minimal_matching.h"

#include <iterator>

#include "geometrycentral/surface/integer_coordinates_intrinsic_triangulation.h"
#include "geometrycentral/utilities/timing.h"

#include "utilities/hsl2rgb.h"
#include "utilities/IO.h"

template <typename T>
std::istream& operator>>(std::istream& is, std::vector<T>& v)
{
    std::copy(std::istream_iterator<T>(is), std::istream_iterator<T>(), std::back_inserter(v));
    return is;
}

#include "args/args.hxx"
#include "imgui.h"

#include <chrono>
#include <limits>
#include <memory>
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
std::chrono::time_point<high_resolution_clock> t1, t2;
std::chrono::duration<double, std::milli> ms_fp;

using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace minimalmatching;

// == geometry-central data
std::unique_ptr<ManifoldSurfaceMesh> mesh_A, mesh_B;
std::unique_ptr<VertexPositionGeometry> extrinsic_geometry_A, extrinsic_geometry_B;
std::unique_ptr<IntegerCoordinatesIntrinsicTriangulation> intrinsic_geometry_A, intrinsic_geometry_B;

VertexData<Vector3> embedding_A, embedding_B;
VertexData<Vector2> uvA, uvB;

VertexData<SurfacePoint> intrinsic_pos_A, intrinsic_pos_B;

// Polyscope data
polyscope::SurfaceMesh *psMesh_A, *psMesh_A_embedding;
polyscope::SurfaceMesh *psMesh_B, *psMesh_B_embedding;

bool ensureTargetMeshDisjoint = true;
Vector3 targetMeshTranslation{0, 0, 0};

// Solvers & parameters
float EPSCOEF = 100.0;
float DOUBLE_WELL_COEF = 1.0;
int BUNDLE_MODE = BundleType::SPIN;
int LAPLACIAN_DISCRETIZATION_MODE = DiscretizationType::FEM;
int MASS_DISCRETIZATION_MODE = DiscretizationType::FEM;
int DOUBLE_WELL_MODE = DoubleWellDiscretizationType::EDGE;
int CURVATURE_FORM_A = CurvatureType::GAUSSIAN;
int CURVATURE_FORM_B = CurvatureType::GAUSSIAN;

std::unique_ptr<MinimalMatchingSolver> minimalMatcher;

// Directory load variables
std::string INPUT_DIR = "";
std::string OUTPUT_DIR = "export";

std::string MESHNAME_A = "Source Mesh (A)";
std::string MESHNAME_B = "Target Mesh (B)";

std::string OUTPUT_FILENAME_A;
std::string OUTPUT_FILENAME_B;

std::string LANDMARKS_EXTENSION = ".pinned";
std::string LANDMARKS_FILENAME_A;
std::string LANDMARKS_FILENAME_B;

bool VERBOSE = true;
bool HEADLESS;
bool MULTIRESOLUTION = false;

// Manage a list of landmarks
std::vector<Landmark> landmarks;
bool useSingularityPinning = true;

VertexData<Vector3> domainColoring(VertexData<Vector3>& z)
{
    auto l = [&](double r)
    {
        return (2. / M_PI) * atan(r);
        return (r * r) / (r * r + 1);
    };

    VertexData<Vector3> c(*z.getMesh(), Vector3{0, 0, 0});
    for (Vertex v : z.getMesh()->vertices())
    {
        Vector2 zv{z[v].x, z[v].y};
        double arg = zv.arg() + 2. * M_PI / 3.;
        double len = z[v].norm();

        arg = fmod(arg + 2. * M_PI, 2. * M_PI);
        arg /= 2. * M_PI;

        auto cv = hsl2rgb(arg, 1., l(len));
        c[v] = Vector3{cv.r, cv.g, cv.b};
    }
    return c;
}

void visualizeSlice(Vertex v_A, Vertex v_B)
{
    auto& m_A = *mesh_A;
    auto& m_B = *mesh_B;

    VertexData<Vector3> z_vA(m_B, Vector3{0, 0, 0});
    VertexData<Vector3> z_vB(m_A, Vector3{0, 0, 0});

    VertexData<double> norm_vA(m_B, 0);
    VertexData<double> norm_vB(m_A, 0);

    for (Vertex v : m_A.vertices())
    {
        z_vB[v] = Vector3{
            minimalMatcher->currentSection(minimalMatcher->vertexIndex(v, v_B)).real(),
            minimalMatcher->currentSection(minimalMatcher->vertexIndex(v, v_B)).imag(),
            1
        };
        norm_vB[v] = z_vB[v].norm();
    }

    for (Vertex v : m_B.vertices())
    {
        z_vA[v] = Vector3{
            minimalMatcher->currentSection(minimalMatcher->vertexIndex(v_A, v)).real(),
            minimalMatcher->currentSection(minimalMatcher->vertexIndex(v_A, v)).imag(),
            1
        };
        norm_vA[v] = z_vA[v].norm();
    }

    auto c_A = domainColoring(z_vB);
    auto c_B = domainColoring(z_vA);

    psMesh_A->addVertexScalarQuantity("Section norm slice", norm_vB);
    psMesh_B->addVertexScalarQuantity("Section norm slice", norm_vA);
}

void solve()
{
    minimalMatcher->options.bundleType = static_cast<BundleType>(BUNDLE_MODE);
    minimalMatcher->options.laplacianType = static_cast<DiscretizationType>(LAPLACIAN_DISCRETIZATION_MODE);
    minimalMatcher->options.massMatrixType = static_cast<DiscretizationType>(MASS_DISCRETIZATION_MODE);
    minimalMatcher->options.doubleWellType = static_cast<DoubleWellDiscretizationType>(DOUBLE_WELL_MODE);
    minimalMatcher->optimize(EPSCOEF);
}

void refreshSolver()
{
    bool anyChanged = false;
    if (minimalMatcher->options.bundleType != BUNDLE_MODE)
        anyChanged = true;

    if (minimalMatcher->options.laplacianType != LAPLACIAN_DISCRETIZATION_MODE)
        anyChanged = true;

    if (minimalMatcher->options.massMatrixType != MASS_DISCRETIZATION_MODE)
        anyChanged = true;

    if (minimalMatcher->options.curvatureAType != CURVATURE_FORM_A)
        anyChanged = true;

    if (minimalMatcher->options.curvatureBType != CURVATURE_FORM_B)
        anyChanged = true;

    minimalMatcher->options.bundleType = static_cast<BundleType>(BUNDLE_MODE);
    minimalMatcher->options.laplacianType = static_cast<DiscretizationType>(LAPLACIAN_DISCRETIZATION_MODE);
    minimalMatcher->options.massMatrixType = static_cast<DiscretizationType>(MASS_DISCRETIZATION_MODE);
    minimalMatcher->options.curvatureAType = static_cast<CurvatureType>(CURVATURE_FORM_A);
    minimalMatcher->options.curvatureBType = static_cast<CurvatureType>(CURVATURE_FORM_B);

    if (anyChanged)
    {
        if (BUNDLE_MODE == SPIN)
            minimalMatcher->buildSpinConnections();
        else if (BUNDLE_MODE == SKYSCRAPER)
            minimalMatcher->buildSkyscraperConnections();
    }
}

void saveOBJ()
{
    std::error_code error;
    if (std::filesystem::exists(OUTPUT_DIR, error))
    {
        if (!std::filesystem::is_directory(OUTPUT_DIR))
        {
            std::cerr << OUTPUT_DIR << " is not a directory!" << std::endl;
            return;
        }
    } else if (std::filesystem::create_directory(OUTPUT_DIR)) {}
    else
    {
        std::cerr << "Failed to create the directory: " + OUTPUT_DIR +"!" << std::endl;
    }

    VertexPositionGeometry embedding_A_geom(*mesh_A, embedding_A);
    writeSurfaceMesh(*mesh_A, embedding_A_geom, OUTPUT_FILENAME_A);

    VertexPositionGeometry embedding_B_geom(*mesh_B, embedding_B);
    writeSurfaceMesh(*mesh_B, embedding_B_geom, OUTPUT_FILENAME_B);
}

bool vizFirstRun = true;

struct BoundingBox
{
    Vector3 min;
    Vector3 max;
};

BoundingBox computeBoundingBox(const VertexData<Vector3>& positions)
{
    double inf = std::numeric_limits<double>::infinity();
    BoundingBox box{{inf, inf, inf}, {-inf, -inf, -inf}};

    for (Vertex v : positions.getMesh()->vertices())
    {
        const Vector3& p = positions[v];
        box.min.x = std::min(box.min.x, p.x);
        box.min.y = std::min(box.min.y, p.y);
        box.min.z = std::min(box.min.z, p.z);
        box.max.x = std::max(box.max.x, p.x);
        box.max.y = std::max(box.max.y, p.y);
        box.max.z = std::max(box.max.z, p.z);
    }

    return box;
}

Vector3 boundingBoxSize(const BoundingBox& box)
{
    return box.max - box.min;
}

void computeTargetMeshBBoxShift()
{
    BoundingBox boxA = computeBoundingBox(extrinsic_geometry_A->vertexPositions);
    BoundingBox boxB = computeBoundingBox(extrinsic_geometry_B->vertexPositions);

    double scale = std::max(boundingBoxSize(boxA).norm(), boundingBoxSize(boxB).norm());
    double gap = 0.05 * scale;
    targetMeshTranslation = Vector3{boxA.max.x - boxB.min.x + gap, 0, 0};
}

Vector3 targetMeshBBoxShift()
{
    return ensureTargetMeshDisjoint ? targetMeshTranslation : Vector3{0, 0, 0};
}

VertexData<Vector3> targetMeshVertexPositions()
{
    Vector3 shift = targetMeshBBoxShift();
    VertexData<Vector3> positions(*mesh_B);
    for (Vertex v : mesh_B->vertices())
        positions[v] = extrinsic_geometry_B->vertexPositions[v] + shift;
    return positions;
}

Vector3 targetMeshInterpolatedPositions(const SurfacePoint& point)
{
    return point.interpolate(extrinsic_geometry_B->vertexPositions) + targetMeshBBoxShift();
}

void updateTargetMeshVisualization()
{
    if (psMesh_B != nullptr)
        psMesh_B->updateVertexPositions(targetMeshVertexPositions());
    if (psMesh_A_embedding != nullptr)
    {
        embedding_A = VertexData<Vector3>(*mesh_A);
        for (Vertex v_A : mesh_A->vertices())
        {
            Vertex v_A_intrinsic = intrinsic_geometry_A->intrinsicMesh->vertex(v_A.getIndex());
            SurfacePoint pos = intrinsic_pos_A[v_A_intrinsic];
            SurfacePoint pos_extrinsic = intrinsic_geometry_B->equivalentPointOnInput(pos);
            embedding_A[v_A] = pos_extrinsic.interpolate(extrinsic_geometry_B->vertexPositions) + targetMeshBBoxShift();
        }
        psMesh_A_embedding->updateVertexPositions(embedding_A);
    }
}

bool loadLandmarks()
{
    std::ifstream inputStreamA(LANDMARKS_FILENAME_A);
    if (!inputStreamA) return false;

    std::ifstream inputStreamB(LANDMARKS_FILENAME_B);
    if (!inputStreamB) return false;
    landmarks.clear();

    std::string line;

    while (getline(inputStreamA, line))
    {
        std::stringstream ss(line);

        std::vector<int> idx;
        ss >> idx;

        std::vector<SurfacePoint> p_A;
        for (auto i : idx)
        {
            Vertex v_A = mesh_A->vertex(std::rint(i));
            SurfacePoint pt_A(v_A);
            p_A.emplace_back(pt_A);
        }

        getline(inputStreamB, line);
        ss = std::stringstream(line);

        idx.clear();
        ss >> idx;

        std::vector<SurfacePoint> p_B;
        for (auto i : idx)
        {
            Vertex v_B = mesh_B->vertex(std::rint(i));
            SurfacePoint pt_B(v_B);
            p_B.emplace_back(pt_B);
        }

        landmarks.push_back({p_A, p_B, 1.});
    }

    minimalMatcher->setLandmarks(landmarks);
    return true;
}

void updateLandmarkViz()
{
    polyscope::removePointCloud("Landmark Points (A)", false);
    polyscope::removePointCloud("Landmark Points (B)", false);
    polyscope::removeCurveNetwork("Landmark Point Pairs", false);
    polyscope::removeCurveNetwork("Landmark Curves (A)", false);
    polyscope::removeCurveNetwork("Landmark Curves (B)", false);

    std::vector<Vector3> sourcePointPositions;
    std::vector<Vector3> targetPointPositions;
    std::vector<glm::vec3> pairEdgePositions;
    std::vector<std::array<size_t, 2>> pairEdgeIndices;
    std::vector<glm::vec3> sourceCurvePositions;
    std::vector<std::array<size_t, 2>> sourceCurveEdges;
    std::vector<glm::vec3> targetCurvePositions;
    std::vector<std::array<size_t, 2>> targetCurveEdges;

    auto appendCurve = [](const std::vector<SurfacePoint>& points,
                          const VertexData<Vector3>& vertexPositions,
                          std::vector<glm::vec3>& curvePositions,
                          std::vector<std::array<size_t, 2>>& curveEdges)
    {
        if (points.size() < 2) return;

        size_t offset = curvePositions.size();
        for (const SurfacePoint& p : points)
        {
            Vector3 pos = p.interpolate(vertexPositions);
            curvePositions.emplace_back(pos.x, pos.y, pos.z);
        }

        for (size_t i = 0; i + 1 < points.size(); ++i)
        {
            curveEdges.push_back({offset + i, offset + i + 1});
        }
    };

    for (Landmark& l : landmarks)
    {
        bool isPointConstraint = l.pt_A.size() == 1 && l.pt_B.size() == 1;
        if (isPointConstraint)
        {
            Vector3 pos_A = l.pt_A.front().interpolate(extrinsic_geometry_A->vertexPositions);
            Vector3 pos_B = targetMeshInterpolatedPositions(l.pt_B.front());

            sourcePointPositions.push_back(pos_A);
            targetPointPositions.push_back(pos_B);

            size_t edgeOffset = pairEdgePositions.size();
            pairEdgePositions.emplace_back(pos_A.x, pos_A.y, pos_A.z);
            pairEdgePositions.emplace_back(pos_B.x, pos_B.y, pos_B.z);
            pairEdgeIndices.push_back({edgeOffset, edgeOffset + 1});
        }
        else
        {
            appendCurve(l.pt_A, extrinsic_geometry_A->vertexPositions, sourceCurvePositions, sourceCurveEdges);
            appendCurve(l.pt_B, targetMeshVertexPositions(), targetCurvePositions, targetCurveEdges);
        }
    }

    if (!sourcePointPositions.empty())
    {
        polyscope::registerPointCloud("Landmark Points (A)", sourcePointPositions);
    }
    if (!targetPointPositions.empty())
    {
        polyscope::registerPointCloud("Landmark Points (B)", targetPointPositions);
    }
    if (!pairEdgeIndices.empty())
    {
        auto network = polyscope::registerCurveNetwork("Landmark Point Pairs", pairEdgePositions, pairEdgeIndices);
        network->setRadius(0.005, false);
    }
    if (!sourceCurveEdges.empty())
    {
        auto network = polyscope::registerCurveNetwork("Landmark Curves (A)", sourceCurvePositions, sourceCurveEdges);
        network->setRadius(0.005, false);
    }
    if (!targetCurveEdges.empty())
    {
        auto network = polyscope::registerCurveNetwork("Landmark Curves (B)", targetCurvePositions, targetCurveEdges);
        network->setRadius(0.005, false);
    }
}

void addLandmark(size_t ind_A, size_t ind_B)
{
    Vertex v_A = mesh_A->vertex(ind_A);
    Vertex v_B = mesh_B->vertex(ind_B);

    SurfacePoint pt_A(v_A);
    SurfacePoint pt_B(v_B);

    Landmark l = {{pt_A}, {pt_B}, 1.0};
    landmarks.push_back(l);
    updateLandmarkViz();
}


void landmarksMenu()
{
    bool anyChanged = false;

    ImGui::PushItemWidth(200);

    int id = -1;
    int eraseInd = -1;
    for (Landmark& l : landmarks)
    {
        std::stringstream ss;
        ss << "\tLandmark " << ++id;
        std::string vStr = ss.str();
        ImGui::PushID(vStr.c_str());

        ImGui::TextUnformatted(vStr.c_str());

        ImGui::SameLine();
        if (ImGui::Button("delete"))
        {
            eraseInd = id;
            anyChanged = true;
        }
        ImGui::Indent();

        ImGui::Unindent();
        ImGui::PopID();
    }
    ImGui::PopItemWidth();

    // actually do erase, if requested
    if (eraseInd != -1)
    {
        landmarks.erase(landmarks.begin() + eraseInd);
    }

    if (ImGui::Button("Add Landmark"))
    {
        long long int pickVert_A = psMesh_A->selectVertex();
        if (pickVert_A >= 0)
        {
            long long int pickVert_B = psMesh_B->selectVertex();
            if (pickVert_B >= 0)
            {
                addLandmark(pickVert_A, pickVert_B);
                anyChanged = true;
            }
        }
    }

    if (anyChanged)
    {
        updateLandmarkViz();
    }
}

FaceData<double> symmetricDirichletEnergyDensity(ManifoldSurfaceMesh& mesh, VertexData<Vector3>& pos_orig,
                                                 VertexData<Vector3>& pos_final)
{
    intrinsic_geometry_A->requireFaceAreas();
    FaceData<double> distortion(mesh, 0);
    double e = 0, area = 0;
    for (Face f : mesh.faces())
    {
        Halfedge ij = f.halfedge();
        Halfedge jk = ij.next();
        Halfedge ki = jk.next();

        Vertex i = ij.tailVertex();
        Vertex j = jk.tailVertex();
        Vertex k = ki.tailVertex();

        // compute edge vectors
        Vector3 u1 = pos_orig[j] - pos_orig[i];
        Vector3 u2 = pos_orig[k] - pos_orig[i];

        Vector3 v1 = pos_final[j] - pos_final[i];
        Vector3 v2 = pos_final[k] - pos_final[i];

        // compute orthonormal bases
        Vector3 e1 = u1;
        e1 = unit(e1);
        Vector3 e2 = (u2 - dot(u2, e1) * e1);
        e2 = unit(e2);

        Vector3 f1 = v1;
        f1 = unit(f1);
        Vector3 f2 = (v2 - dot(v2, f1) * f1);
        f2 = unit(f2);

        // project onto bases
        Vector3 p0 = Vector3(0, 0, 0);
        Vector3 p1 = Vector3(dot(u1, e1), dot(u1, e2), 0);
        Vector3 p2 = Vector3(dot(u2, e1), dot(u2, e2), 0);

        Vector3 q0 = Vector3(0, 0, 0);
        Vector3 q1 = Vector3(dot(v1, f1), dot(v1, f2), 0);
        Vector3 q2 = Vector3(dot(v2, f1), dot(v2, f2), 0);

        double A = 0.5 * cross(u1, u2).norm();
        area += A;

        Vector3 Ss = (q0 * (p1.y - p2.y) + q1 * (p2.y - p0.y) + q2 * (p0.y - p1.y)) / (2. * A);
        Vector3 St = (q0 * (p2.x - p1.x) + q1 * (p0.x - p2.x) + q2 * (p1.x - p0.x)) / (2. * A);
        double a = dot(Ss, Ss);
        double b = dot(Ss, St);
        double c = dot(St, St);
        double det = std::sqrt(std::pow(a - c, 2) + 4.0 * b * b);

        double Gamma = std::sqrt(0.5 * (a + c + det));
        double gamma = std::sqrt(0.5 * (a + c - det));
        distortion[f] = 0.5 * (Gamma * Gamma + gamma * gamma + (1. / Gamma * 1. / Gamma) + (1. / gamma * 1. / gamma));

        e += distortion[f] * A;
    }

    e /= area;

    return distortion;
}

double symmetricDirichletEnergy(VertexData<Vector3>& embA, VertexData<Vector3>& embB)
{
    FaceData<double> dist_A = symmetricDirichletEnergyDensity(*mesh_A, extrinsic_geometry_A->vertexPositions, embA);
    FaceData<double> dist_B = symmetricDirichletEnergyDensity(*mesh_B, extrinsic_geometry_B->vertexPositions, embB);

    extrinsic_geometry_A->requireFaceAreas();
    extrinsic_geometry_B->requireFaceAreas();

    double E_A = (dist_A.raw().array() * extrinsic_geometry_A->faceAreas.raw().array()).sum();
    double E_B = (dist_B.raw().array() * extrinsic_geometry_B->faceAreas.raw().array()).sum();

    psMesh_A->addFaceScalarQuantity("Distortion", dist_A);
    psMesh_B->addFaceScalarQuantity("Distortion", dist_B);

    return 0.5 * (E_A + E_B);
}

void visualizeVertexCorrespondence()
{
    psMesh_A_embedding = polyscope::registerSurfaceMesh(MESHNAME_A + " inside " + MESHNAME_B, embedding_A,
                                                    mesh_A->getFaceVertexList());
    psMesh_B_embedding = polyscope::registerSurfaceMesh(MESHNAME_B + " inside " + MESHNAME_A, embedding_B,
                                                        mesh_B->getFaceVertexList());

    auto uvA_viz = psMesh_A_embedding->addVertexParameterizationQuantity("Texture Transfer", uvA);
    auto uvB_viz = psMesh_B_embedding->addVertexParameterizationQuantity("Texture Transfer", uvB);

    uvA_viz->setEnabled(true);
    uvA_viz->setStyle(polyscope::ParamVizStyle::LOCAL_RAD);
    uvA_viz->setCheckerSize(0.1);

    uvB_viz->setEnabled(true);
    uvB_viz->setStyle(polyscope::ParamVizStyle::LOCAL_CHECK);
    uvB_viz->setCheckerSize(0.1);

}

void extractVertexCorrespondence()
{
    std::tie(intrinsic_pos_A, intrinsic_pos_B) = minimalMatcher->extractCorrespondence();

    embedding_A = VertexData<Vector3>(*mesh_A);
    for (Vertex v_A : mesh_A->vertices())
    {
        Vertex v_A_intrinsic = intrinsic_geometry_A->intrinsicMesh->vertex(v_A.getIndex());
        SurfacePoint pos = intrinsic_pos_A[v_A_intrinsic];
        SurfacePoint pos_extrinsic = intrinsic_geometry_B->equivalentPointOnInput(pos);
        embedding_A[v_A] = pos_extrinsic.interpolate(extrinsic_geometry_B->vertexPositions) + targetMeshBBoxShift();
    }

    embedding_B = VertexData<Vector3>(*mesh_B);
    for (Vertex v_B : mesh_B->vertices())
    {
        Vertex v_B_intrinsic = intrinsic_geometry_B->intrinsicMesh->vertex(v_B.getIndex());
        SurfacePoint pos = intrinsic_pos_B[v_B_intrinsic];
        if (pos == SurfacePoint()) continue;
        SurfacePoint pos_extrinsic = intrinsic_geometry_A->equivalentPointOnInput(pos);
        embedding_B[v_B] = pos_extrinsic.interpolate(extrinsic_geometry_A->vertexPositions);
    }

    if (!HEADLESS)
        visualizeVertexCorrespondence();
}


void callback()
{
    if (ImGui::Button("Compute Correspondence"))
    {
        START_TIMING(SOLVE)
        minimalMatcher->closestPointInitialization(extrinsic_geometry_A->vertexPositions,
                                                   extrinsic_geometry_B->vertexPositions);
        solve();
        FINISH_TIMING_PRINT(SOLVE)
        extractVertexCorrespondence();
        saveOBJ();
    }

    ImGui::Separator();
    if (ImGui::Button("Initialize"))
    {
        START_TIMING(INITIALIZE)
        minimalMatcher->closestPointInitialization(extrinsic_geometry_A->vertexPositions,
                                                   extrinsic_geometry_B->vertexPositions);
        FINISH_TIMING_PRINT(INITIALIZE)
    }
    ImGui::SameLine();
    if (ImGui::Button("Refine"))
    {
        START_TIMING(SOLVE)
        solve();
        FINISH_TIMING_PRINT(SOLVE)
    }

    ImGui::SameLine();

    if (ImGui::Button("Reset"))
    {
        minimalMatcher->currentSection = minimalMatcher->initializationSection;
    }

    ImGui::Spacing();

    if (ImGui::Button("Extract Vertex Correspondence"))
    {
        extractVertexCorrespondence();
    }
    ImGui::SameLine();
    if (ImGui::Button("Export Vertex Correspondence"))
    {
        saveOBJ();
    }

    if (ImGui::Button("Extract Mesh Correspondence"))
    {
        auto [paths_A, paths_B] = minimalMatcher->extractEdgeCorrespondence();

        std::vector<glm::vec3> edge_paths_pos;
        std::vector<std::array<size_t, 2>> edge_paths;
        std::vector<std::vector<size_t>> edge_path_full;
        for (Edge e : intrinsic_geometry_A->intrinsicMesh->edges())
        {
            auto gamma = paths_A[e];

            if (gamma.size() < 2) continue;

            for (size_t i = 0; i < gamma.size() - 1; i++)
            {
                SurfacePoint p0 = gamma[i];
                SurfacePoint p1 = gamma[i + 1];

                Vector3 v0{0, 0, 0}, v1{0, 0, 0};
                if (p0.type == SurfacePointType::Face)
                {
                    SurfacePoint p0_ext = intrinsic_geometry_B->equivalentPointOnInput(p0);
                    v0 = targetMeshInterpolatedPositions(p0_ext);
                }
                else if (p0.type == SurfacePointType::Edge)
                {
                    v0 = targetMeshInterpolatedPositions(p0);
                }

                if (p1.type == SurfacePointType::Face)
                {
                    SurfacePoint p1_ext = intrinsic_geometry_B->equivalentPointOnInput(p1);
                    v1 = targetMeshInterpolatedPositions(p1_ext);
                }
                else if (p1.type == SurfacePointType::Edge)
                {
                    v1 = targetMeshInterpolatedPositions(p1);
                }

                edge_paths.push_back({edge_paths_pos.size(), edge_paths_pos.size() + 1});
                edge_paths_pos.push_back(
                    glm::vec3(v0.x, v0.y, v0.z)
                );
                edge_paths_pos.push_back(
                    glm::vec3(v1.x, v1.y, v1.z)
                );
            }
        }

        polyscope::registerCurveNetwork("E_A in B", edge_paths_pos, edge_paths);
    }


    if (ImGui::TreeNode("Section I/O"))
    {
        if (ImGui::Button("Export Section"))
        {
                std::string section_filename = OUTPUT_DIR + "/" + MESHNAME_A + "x" + MESHNAME_B + ".section";
                std::ofstream outStream(section_filename);
                if (!outStream) throw std::runtime_error("couldn't open output file " + section_filename);
                outStream << minimalMatcher->currentSection;
                outStream.close();
        }
        ImGui::SameLine();
        if (ImGui::Button("Import Section"))
        {
            // Warning: There are no safety checks
            std::string section_filename = OUTPUT_DIR + "/" + MESHNAME_A + "x" + MESHNAME_B + ".section";
            std::ifstream inStream(section_filename);
            if (!inStream) throw std::runtime_error("couldn't open output file " + section_filename);

            std::string line;

            minimalMatcher->initializationSection.resize(mesh_A->nVertices() * mesh_B->nVertices());

            size_t iV = 0;
            while (getline(inStream, line))
            {
                std::stringstream ss(line);

                std::complex<double> zi;
                ss >> zi;

                minimalMatcher->initializationSection(iV++) = zi;
            }

            minimalMatcher->currentSection = minimalMatcher->initializationSection;
            inStream.close();
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    ImGui::Text("Solve options");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::InputFloat("Unit Norm Penalty", &EPSCOEF);
    ImGui::InputInt("Max Iterations", &minimalMatcher->options.maxIterations);
    ImGui::InputInt("Curvature Smoothing Iterations", &minimalMatcher->options.smoothingIterations);

    if (ImGui::RadioButton("Spin Bundle", &BUNDLE_MODE, BundleType::SPIN))
    {
        refreshSolver();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Skyscraper Bundle", &BUNDLE_MODE, BundleType::SKYSCRAPER))
    {
        refreshSolver();
    }
    if (ImGui::RadioButton("FEM Laplacian", &LAPLACIAN_DISCRETIZATION_MODE, DiscretizationType::FEM))
    {
        refreshSolver();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("DEC Laplacian", &LAPLACIAN_DISCRETIZATION_MODE, DiscretizationType::DEC))
    {
        refreshSolver();
    }
    if (ImGui::RadioButton("FEM Mass", &MASS_DISCRETIZATION_MODE, DiscretizationType::FEM))
    {
        refreshSolver();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("DEC Mass", &MASS_DISCRETIZATION_MODE, DiscretizationType::DEC))
    {
        refreshSolver();
    }

    if (ImGui::RadioButton("K (A)", &CURVATURE_FORM_A, CurvatureType::GAUSSIAN))
    {
        refreshSolver();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("dA (A)", &CURVATURE_FORM_A, CurvatureType::AREA))
    {
        refreshSolver();
    }
    if (ImGui::RadioButton("K (B)", &CURVATURE_FORM_B, CurvatureType::GAUSSIAN))
    {
        refreshSolver();
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("dA (B)", &CURVATURE_FORM_B, CurvatureType::AREA))
    {
        refreshSolver();
    }

    if (ImGui::InputFloat("Double Well Coefficient", &DOUBLE_WELL_COEF))
    {
        minimalMatcher->options.doubleWellCoeff = DOUBLE_WELL_COEF;
    }

    ImGui::Text("Double Well Discretization");
    ImGui::SameLine();
    ImGui::RadioButton("Vertex", &DOUBLE_WELL_MODE, DoubleWellDiscretizationType::VERTEX);
    ImGui::SameLine();
    ImGui::RadioButton("Edge", &DOUBLE_WELL_MODE, DoubleWellDiscretizationType::EDGE);

    ImGui::Separator();
    if (ImGui::TreeNode("Landmarks"))
    {
        landmarksMenu();

        ImGui::Checkbox("Singularity Pinning", &useSingularityPinning);
        ImGui::SameLine();
        if (ImGui::Button("Setup Landmarks"))
        {
            if (!useSingularityPinning)
                minimalMatcher->setLandmarks({});
            else
                minimalMatcher->setLandmarks(landmarks);
        }

        if (ImGui::Button("Save Landmarks"))
        {
            std::string landmarks_filename = OUTPUT_DIR + +"/" + MESHNAME_A + ".landmarks";
            std::ofstream outStream(landmarks_filename);
            if (!outStream) throw std::runtime_error("couldn't open output file " + landmarks_filename);
            for (Landmark& l : landmarks)
            {
                for (auto& p_A : l.pt_A)
                    outStream << p_A.vertex.getIndex() << " ";
                outStream << std::endl;
                for (auto& p_B : l.pt_B)
                    outStream << p_B.vertex.getIndex() << " ";
                outStream << std::endl;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load Landmarks"))
        {
            if (loadLandmarks())
                updateLandmarkViz();
        }
        ImGui::TreePop();
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Experiments"))
    {
        if (ImGui::Button("Optimal Epsilon Test"))
        {
            int maxIterations = minimalMatcher->options.maxIterations;
            minimalMatcher->options.maxIterations = 100;

            std::string epsilon_test_filename = OUTPUT_DIR + +"/" + MESHNAME_A + "_" + MESHNAME_B + ".epsTest";
            std::ofstream outStream(epsilon_test_filename);
            if (!outStream) throw std::runtime_error("couldn't open output file " + epsilon_test_filename);

            int maxEpsIters = 100;
            float minEpsCoefficient = 10;
            float maxEpsCoefficient = 200;
            float epsCoefficient = 1;
            for (size_t i = 0; i < maxEpsIters; i++)
            {
                epsCoefficient = minEpsCoefficient + (maxEpsCoefficient - minEpsCoefficient) * ((float)i) / ((float)(
                    maxEpsIters - 1));
                // minimalMatcher->currentSection = minimalMatcher->initializationSection;
                minimalMatcher->optimize(epsCoefficient);
                auto [intrinsic_pos_A, intrinsic_pos_B] = minimalMatcher->extractCorrespondence();

                embedding_A = VertexData<Vector3>(*mesh_A);
                for (Vertex v_A : mesh_A->vertices())
                {
                    SurfacePoint pos = intrinsic_pos_A[v_A];
                    SurfacePoint pos_extrinsic = intrinsic_geometry_B->equivalentPointOnInput(pos);
                    embedding_A[v_A] = pos_extrinsic.interpolate(extrinsic_geometry_B->vertexPositions);
                }

                embedding_B = VertexData<Vector3>(*mesh_B);
                for (Vertex v_B : mesh_B->vertices())
                {
                    SurfacePoint pos = intrinsic_pos_B[v_B];
                    SurfacePoint pos_extrinsic = intrinsic_geometry_A->equivalentPointOnInput(pos);
                    embedding_B[v_B] = pos_extrinsic.interpolate(extrinsic_geometry_A->vertexPositions);
                }

                double distortion = symmetricDirichletEnergy(embedding_A, embedding_B);
                outStream << epsCoefficient << ", " << distortion << std::endl;
            }
            outStream.close();
            minimalMatcher->options.maxIterations = maxIterations;
        }

        ImGui::TreePop();
    }

    ImGui::Separator();
    ImGui::Text("Visualization options");
    ImGui::Separator();

    if (ImGui::Checkbox("Make Embeddings Disjoint", &ensureTargetMeshDisjoint))
    {
        updateTargetMeshVisualization();
        updateLandmarkViz();
    }

    if (ImGui::Button("Visualize Slice"))
    {
        long long int pickVert_A = psMesh_A->selectVertex();
        long long int pickVert_B = psMesh_B->selectVertex();

        visualizeSlice(mesh_A->vertex(pickVert_A), mesh_B->vertex(pickVert_B));
    }
}

int main(int argc, char** argv)
{
    // Configure the argument parser
    args::ArgumentParser parser("Implicit Minimal Surfaces for Bijective Correspondences.");
    args::HelpFlag help(parser, "help", "Display this help menu", {"help"});

    args::Positional<std::string> directory(parser, "directory", "The working directory.");
    args::Positional<std::string> sourceMeshFilename(parser, "source", "A triangulated surface to be matched.");
    args::Positional<std::string> targetMeshFilename(parser, "target", "A triangulated surface to be matched.");

    args::Group group(parser);
    args::Flag verbose(group, "verbose", "Verbose output", {"V", "verbose"});
    args::Flag headless(group, "headless", "Don't use the GUI.", {"l", "headless"});

    // Parse args
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help&)
    {
        std::cout << parser;
        return 0;
    } catch (args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    if (!directory)
    {
        std::cerr << "Please specify a working directory as an argument." << std::endl;
        return EXIT_FAILURE;
    }

    if (!sourceMeshFilename)
    {
        std::cerr << "Please specify a source mesh file as argument." << std::endl;
        return EXIT_FAILURE;
    }
    if (!targetMeshFilename)
    {
        std::cerr << "Please specify a target mesh file as argument." << std::endl;
        return EXIT_FAILURE;
    }

    // Load mesh
    INPUT_DIR = args::get(directory);
    OUTPUT_DIR = INPUT_DIR + "/" + OUTPUT_DIR;

    std::string sourceMeshFilePath = INPUT_DIR + "/" + args::get(sourceMeshFilename);
    std::string targetMeshFilePath = INPUT_DIR + "/" + args::get(targetMeshFilename);

    MESHNAME_A = polyscope::guessNiceNameFromPath(sourceMeshFilePath);
    OUTPUT_FILENAME_A = OUTPUT_DIR + "/" + MESHNAME_A + "_mapped.obj";
    MESHNAME_B = polyscope::guessNiceNameFromPath(targetMeshFilePath);
    OUTPUT_FILENAME_B = OUTPUT_DIR + "/" + MESHNAME_B + "_mapped.obj";

    LANDMARKS_FILENAME_A = INPUT_DIR + "/" + MESHNAME_A + LANDMARKS_EXTENSION;
    LANDMARKS_FILENAME_B = INPUT_DIR + "/" + MESHNAME_B + LANDMARKS_EXTENSION;

    HEADLESS = headless;
    VERBOSE = verbose;

    // Read the meshes.
    std::tie(mesh_A, extrinsic_geometry_A) = readManifoldSurfaceMesh(sourceMeshFilePath);
    std::tie(mesh_B, extrinsic_geometry_B) = readManifoldSurfaceMesh(targetMeshFilePath);

    intrinsic_geometry_A.reset(new IntegerCoordinatesIntrinsicTriangulation(*mesh_A, *extrinsic_geometry_A, 1e-3));
    intrinsic_geometry_B.reset(new IntegerCoordinatesIntrinsicTriangulation(*mesh_B, *extrinsic_geometry_B, 1e-3));

    intrinsic_geometry_A->requireFaceAreas();
    intrinsic_geometry_B->requireFaceAreas();

    intrinsic_geometry_A->inputEdgeLengths *= std::sqrt(1. / intrinsic_geometry_A->faceAreas.raw().sum());
    intrinsic_geometry_B->inputEdgeLengths *= std::sqrt(1. / intrinsic_geometry_B->faceAreas.raw().sum());

    intrinsic_geometry_A->flipToDelaunay();
    intrinsic_geometry_B->flipToDelaunay();

    intrinsic_geometry_A->refreshQuantities();
    intrinsic_geometry_B->refreshQuantities();

    intrinsic_geometry_A->requireMeshLengthScale();
    intrinsic_geometry_B->requireMeshLengthScale();

    minimalMatcher = std::make_unique<MinimalMatchingSolver>(
        *intrinsic_geometry_A->intrinsicMesh, *intrinsic_geometry_B->intrinsicMesh,
        *intrinsic_geometry_A, *intrinsic_geometry_B, *extrinsic_geometry_A, *extrinsic_geometry_B);

    refreshSolver();
    bool haveLandmarks = loadLandmarks();

    if (!HEADLESS)
    {
        polyscope::init();
        polyscope::state::userCallback = callback;
        psMesh_A = polyscope::registerSurfaceMesh(MESHNAME_A, extrinsic_geometry_A->vertexPositions,
                                                  mesh_A->getFaceVertexList());
        psMesh_B = polyscope::registerSurfaceMesh(MESHNAME_B, extrinsic_geometry_B->vertexPositions,
                                                  mesh_B->getFaceVertexList());
        computeTargetMeshBBoxShift();
        updateTargetMeshVisualization();

        uvA = VertexData<Vector2>(*mesh_A);
        uvB = VertexData<Vector2>(*mesh_B);

        for (Vertex v : mesh_A->vertices())
            uvA[v] = Vector2{
                extrinsic_geometry_A->vertexPositions[v].x + extrinsic_geometry_A->vertexPositions[v].z,
                extrinsic_geometry_A->vertexPositions[v].y};
        for (Vertex v : mesh_B->vertices())
            uvB[v] = Vector2{
                extrinsic_geometry_B->vertexPositions[v].x + extrinsic_geometry_B->vertexPositions[v].z,
                extrinsic_geometry_B->vertexPositions[v].y};

        auto uvA_viz = psMesh_A->addVertexParameterizationQuantity("Source Texture", uvA);
        auto uvB_viz = psMesh_B->addVertexParameterizationQuantity("Source Texture", uvB);

        uvA_viz->setEnabled(true);
        uvA_viz->setStyle(polyscope::ParamVizStyle::LOCAL_RAD);
        uvA_viz->setCheckerSize(0.1);

        uvB_viz->setEnabled(true);
        uvB_viz->setStyle(polyscope::ParamVizStyle::LOCAL_CHECK);
        uvB_viz->setCheckerSize(0.1);

        if (mesh_A->isTriangular()) psMesh_A->setAllPermutations(polyscopePermutations(*mesh_A));
        if (mesh_B->isTriangular()) psMesh_B->setAllPermutations(polyscopePermutations(*mesh_B));

        if (haveLandmarks)
            updateLandmarkViz();

        polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
        polyscope::view::setUpDir(polyscope::UpDir::YUp);
        polyscope::view::setFrontDir(polyscope::FrontDir::ZFront);
        polyscope::show();
    }
    else
    {
        minimalMatcher->closestPointInitialization(extrinsic_geometry_A->vertexPositions,
                                                   extrinsic_geometry_B->vertexPositions);
        solve();
        extractVertexCorrespondence();
        saveOBJ();
    }

    return EXIT_SUCCESS;
}
