#include "boundary_conditions.hpp"
#include "mesh.hpp"
#include "comm.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

namespace {

std::string trimAndUpper(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    std::string out = s.substr(start, end - start);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    }
    return out;
}

} // namespace

BcType BoundaryConditions::parseBcTag(const std::string& rawTag, int neighborElementId) {
    std::string tag = trimAndUpper(rawTag);

    // 1. Check explicit multi-character tags first (avoids 'P' prefix ambiguity between PEC/PMC/PML/PERIODIC)
    if (tag.find("MPI") != std::string::npos) {
        return BcType::MPI_CUT;
    }
    if (tag.find("PML") != std::string::npos) {
        return BcType::PML;
    }
    if (tag.find("PMC") != std::string::npos || tag.find("SYM") != std::string::npos) {
        return BcType::PMC;
    }
    if (tag.find("PEC") != std::string::npos) {
        return BcType::PEC;
    }
    if (tag.find("PER") != std::string::npos) {
        return BcType::PERIODIC;
    }

    // 2. Check standard single-character Nek5000 / NekCEM tags
    if (tag == "P") {
        return BcType::PERIODIC;
    }
    if (tag == "E") {
        return (neighborElementId >= 0) ? BcType::INTERIOR : BcType::PEC;
    }
    if (tag == "W" || tag == "V") {
        return BcType::PEC;
    }
    if (tag == "S" || tag == "M") {
        return BcType::PMC;
    }

    // 3. Fallback based on topological connectivity
    if (neighborElementId >= 0) {
        return BcType::INTERIOR;
    }
    return BcType::PEC;
}

std::string BoundaryConditions::toString(BcType bc) {
    switch (bc) {
        case BcType::INTERIOR: return "E";
        case BcType::PEC:      return "PEC";
        case BcType::PMC:      return "PMC";
        case BcType::PML:      return "PML";
        case BcType::PERIODIC: return "PERIODIC";
        case BcType::MPI_CUT:  return "MPI";
        default:               return "PEC";
    }
}

void BoundaryConditions::buildFacePointBcTypes(const Mesh& mesh, std::vector<int>& h_bcType) {
    const auto& faceData = mesh.getFaceData();
    size_t totalFacePoints = 0;
    for (const auto& fd : faceData) {
        totalFacePoints += fd.points.size();
    }

    h_bcType.resize(totalFacePoints);
    size_t ptIdx = 0;
    for (const auto& fd : faceData) {
        int samplePlus = fd.points.empty() ? -1 : fd.points[0].volIdxPlus;
        BcType faceBc = parseBcTag(fd.bcType, samplePlus);

        for (const auto& pt : fd.points) {
            BcType ptBc = faceBc;
            // If a face point has no exterior neighbor (volIdxPlus < 0), it cannot be INTERIOR/PERIODIC/MPI
            if (pt.volIdxPlus < 0 &&
                (ptBc == BcType::INTERIOR || ptBc == BcType::PERIODIC || ptBc == BcType::MPI_CUT)) {
                ptBc = BcType::PEC;
            }
            h_bcType[ptIdx++] = static_cast<int>(ptBc);
        }
    }
}

bool BoundaryConditions::hasPmlFaces(const Mesh& mesh) {
    for (const auto& f : mesh.getFaces()) {
        if (parseBcTag(f.bcType, f.neighborElementId) == BcType::PML) {
            return true;
        }
    }
    for (const auto& fd : mesh.getFaceData()) {
        int samplePlus = fd.points.empty() ? -1 : fd.points[0].volIdxPlus;
        if (parseBcTag(fd.bcType, samplePlus) == BcType::PML) {
            return true;
        }
    }
    return false;
}

bool BoundaryConditions::validatePmlConfig(const PmlConfig& cfg) {
    // Matches pml_errchk in contrib/NekCEM/src/cem_maxwell_pml.F
    if (cfg.thickness < 1 || cfg.thickness > 10) {
        if (Comm::isRoot()) {
            std::cerr << "[PML ERROR] Invalid pml_thickness = " << cfg.thickness
                      << ". Must be between 1 and 10." << std::endl;
        }
        return false;
    }
    if (cfg.order < 1.0 || cfg.order > 10.0) {
        if (Comm::isRoot()) {
            std::cerr << "[PML ERROR] Invalid pml_order = " << cfg.order
                      << ". Must be between 1 and 10." << std::endl;
        }
        return false;
    }
    if (cfg.reflectErr <= 0.0 || cfg.reflectErr > 1.0) {
        if (Comm::isRoot()) {
            std::cerr << "[PML ERROR] Invalid pml_reflect_err = " << cfg.reflectErr
                      << ". Must be in (0, 1]." << std::endl;
        }
        return false;
    }
    return true;
}

void BoundaryConditions::marchFaces(const Mesh& mesh, std::vector<double>& faceary) {
    // Matches march_faces in contrib/NekCEM/src/cem_maxwell_pml.F
    // NekWave faceId (0..5) corresponds to Ed's 1..6 numbering:
    //   axis 0 (r): negFace = 3 (r = -1), posFace = 1 (r = +1)
    //   axis 1 (s): negFace = 0 (s = -1), posFace = 2 (s = +1)
    //   axis 2 (t): negFace = 4 (t = -1), posFace = 5 (t = +1)
    const int negFaces[3] = {3, 0, 4};
    const int posFaces[3] = {1, 2, 5};

    const int nelt = mesh.getNumElements();
    const int N = mesh.getN();
    const int N2 = N * N;

    // 1. Increment opposing faces inside each element if at least one is non-zero
    for (int elt = 0; elt < nelt; ++elt) {
        for (int axis = 0; axis < 3; ++axis) {
            int posFace = posFaces[axis];
            int negFace = negFaces[axis];
            for (int idx = 0; idx < N2; ++idx) {
                double& posVal = faceary[(elt * 6 + posFace) * N2 + idx];
                double& negVal = faceary[(elt * 6 + negFace) * N2 + idx];
                if (posVal + negVal != 0.0) {
                    posVal += 1.0;
                    negVal += 1.0;
                }
            }
        }
    }

    // 2. Maximum across shared interior neighboring element faces (gs_op max)
    // Note: Do not march across periodic boundaries (matches NekCEM comment in march_faces).
    const auto& faces = mesh.getFaces();
    std::vector<double> synced = faceary;
    for (const auto& f : faces) {
        int eltA = f.elementId;
        int faceA = f.faceId;
        int eltB = f.neighborElementId;
        int faceB = f.neighborFaceId;

        if (eltB >= 0 && faceB >= 0 && parseBcTag(f.bcType, eltB) == BcType::INTERIOR) {
            if (N >= 3) {
                for (int iz = 1; iz < N - 1; ++iz) {
                    for (int ix = 1; ix < N - 1; ++ix) {
                        int idx = iz * N + ix;
                        double vA = faceary[(eltA * 6 + faceA) * N2 + idx];
                        double vB = faceary[(eltB * 6 + faceB) * N2 + idx];
                        double m = std::max(vA, vB);
                        synced[(eltA * 6 + faceA) * N2 + idx] = m;
                        synced[(eltB * 6 + faceB) * N2 + idx] = m;
                    }
                }
            } else {
                for (int idx = 0; idx < N2; ++idx) {
                    double vA = faceary[(eltA * 6 + faceA) * N2 + idx];
                    double vB = faceary[(eltB * 6 + faceB) * N2 + idx];
                    double m = std::max(vA, vB);
                    synced[(eltA * 6 + faceA) * N2 + idx] = m;
                    synced[(eltB * 6 + faceB) * N2 + idx] = m;
                }
            }
        }
    }
    faceary.swap(synced);
}

void BoundaryConditions::pmlFillFaceArray(const Mesh& mesh, std::vector<double>& faceary, int thick) {
    // Matches pml_fill_faceary in contrib/NekCEM/src/cem_maxwell_pml.F
    const int nelt = mesh.getNumElements();
    const int N = mesh.getN();
    const int N2 = N * N;

    faceary.assign(static_cast<size_t>(nelt) * 6 * N2, 0.0);

    const auto& faces = mesh.getFaces();
    for (const auto& f : faces) {
        if (parseBcTag(f.bcType, f.neighborElementId) == BcType::PML) {
            int elt = f.elementId;
            int face = f.faceId;
            if (N >= 3) {
                // Seed interior nodes of PML faces (avoiding edges so values do not bleed sideways)
                for (int iz = 1; iz < N - 1; ++iz) {
                    for (int ix = 1; ix < N - 1; ++ix) {
                        faceary[(elt * 6 + face) * N2 + iz * N + ix] = 1.0;
                    }
                }
            } else {
                for (int idx = 0; idx < N2; ++idx) {
                    faceary[(elt * 6 + face) * N2 + idx] = 1.0;
                }
            }
        }
    }

    // March inward through opposing hex faces `thick` times
    for (int step = 0; step < thick; ++step) {
        marchFaces(mesh, faceary);
    }
}

int BoundaryConditions::dirLocalToGlobal(const Mesh& mesh, int elt, int symDir) {
    // Matches dir_local_to_global in contrib/NekCEM/src/cem_maxwell_pml.F
    // symDir is in 0..5 symmetric convention: 0=-r, 1=+r, 2=-s, 3=+s, 4=-t, 5=+t
    double locvec[3] = {0.0, 0.0, 0.0};
    int axis = symDir / 2;
    locvec[axis] = (symDir % 2 == 0) ? -1.0 : 1.0;

    int p0 = elt * mesh.getNumPointsPerElement();
    double globvec[3];
    globvec[0] = mesh.getRx()[p0] * locvec[0]
               + mesh.getSx()[p0] * locvec[1]
               + mesh.getTx()[p0] * locvec[2];
    globvec[1] = mesh.getRy()[p0] * locvec[0]
               + mesh.getSy()[p0] * locvec[1]
               + mesh.getTy()[p0] * locvec[2];
    globvec[2] = mesh.getRz()[p0] * locvec[0]
               + mesh.getSz()[p0] * locvec[1]
               + mesh.getTz()[p0] * locvec[2];

    int argmax = 0;
    double biggest = 0.0;
    for (int i = 0; i < 3; ++i) {
        if (std::abs(globvec[i]) >= biggest) {
            biggest = std::abs(globvec[i]);
            argmax = i;
        }
    }

    int globdir = argmax * 2; // Default to negative global axis direction (0=-X, 2=-Y, 4=-Z)
    if (globvec[argmax] >= 0.0) {
        globdir += 1;         // Positive global axis direction (1=+X, 3=+Y, 5=+Z)
    }
    return globdir;
}

void BoundaryConditions::pmlExtentAndTags(const Mesh& mesh, const std::vector<double>& faceary, PmlData& pml) {
    // Matches pml_extent_and_tags in contrib/NekCEM/src/cem_maxwell_pml.F
    const int nelt = mesh.getNumElements();
    const int N = mesh.getN();
    const int N2 = N * N;
    const int nxyz = mesh.getNumPointsPerElement();

    // Mapping from symmetric face index (0=-r, 1=+r, 2=-s, 3=+s, 4=-t, 5=+t)
    // to NekWave / Ed's faceId (0..5)
    const int eface0[6] = {3, 1, 0, 2, 4, 5};
    const int oppSymFace[6] = {1, 0, 3, 2, 5, 4};

    const int indPtIdx = (N >= 3) ? (1 * N + 1) : 0;
    const double pmlInf = 1.0e20;

    for (int axis = 0; axis < 3; ++axis) {
        pml.pmlOuter[axis * 2 + 0] =  pmlInf;
        pml.pmlInner[axis * 2 + 0] = -pmlInf;
        pml.pmlInner[axis * 2 + 1] =  pmlInf;
        pml.pmlOuter[axis * 2 + 1] = -pmlInf;
    }

    pml.pmlTag.assign(nelt, 0);

    const auto& xm1 = mesh.getCoordX();
    const auto& ym1 = mesh.getCoordY();
    const auto& zm1 = mesh.getCoordZ();

    for (int elt = 0; elt < nelt; ++elt) {
        for (int axis = 0; axis < 3; ++axis) {
            int symFace = axis * 2; // Negative symmetric face along local axis
            double farHere = faceary[(elt * 6 + eface0[symFace]) * N2 + indPtIdx];
            double farOpp  = faceary[(elt * 6 + eface0[oppSymFace[symFace]]) * N2 + indPtIdx];

            if (farHere != 0.0 && farOpp != 0.0) {
                if (farHere == farOpp) {
                    if (Comm::isRoot()) {
                        std::cerr << "[PML ERROR] No gradient in PML indicators in element " << elt
                                  << ". PML layers from opposite boundaries may be colliding." << std::endl;
                    }
                    continue;
                }

                int globFace = dirLocalToGlobal(mesh, elt, symFace);
                int globAxis = globFace / 2;
                int globSign = (globFace % 2) * 2 - 1;

                // Force negGlobFace to negative side (0=-X, 2=-Y, 4=-Z) and posGlobFace to positive side (1=+X, 3=+Y, 5=+Z)
                int negGlobFace = globAxis * 2;
                int posGlobFace = globAxis * 2 + 1;

                const std::vector<double>& coordArr =
                    (globAxis == 0) ? xm1 : ((globAxis == 1) ? ym1 : zm1);

                double minCoord =  pmlInf;
                double maxCoord = -pmlInf;
                int base = elt * nxyz;
                for (int i = 0; i < nxyz; ++i) {
                    double c = coordArr[base + i];
                    minCoord = std::min(minCoord, c);
                    maxCoord = std::max(maxCoord, c);
                }

                if (globSign * (farHere - farOpp) > 0.0) {
                    // Positive-side PML slab (+X, +Y, or +Z)
                    pml.pmlInner[posGlobFace] = std::min(pml.pmlInner[posGlobFace], minCoord);
                    pml.pmlOuter[posGlobFace] = std::max(pml.pmlOuter[posGlobFace], maxCoord);
                    pml.pmlTag[elt] |= (1 << posGlobFace);
                } else if (globSign * (farHere - farOpp) < 0.0) {
                    // Negative-side PML slab (-X, -Y, or -Z)
                    pml.pmlInner[negGlobFace] = std::max(pml.pmlInner[negGlobFace], maxCoord);
                    pml.pmlOuter[negGlobFace] = std::min(pml.pmlOuter[negGlobFace], minCoord);
                    pml.pmlTag[elt] |= (1 << negGlobFace);
                }
            }
        }
    }

    // Global MPI reductions across ranks (matching NekCEM's gop calls)
    for (int axis = 0; axis < 3; ++axis) {
        pml.pmlInner[axis * 2 + 0] = Comm::allreduceMax(pml.pmlInner[axis * 2 + 0]);
        pml.pmlInner[axis * 2 + 1] = Comm::allreduceMin(pml.pmlInner[axis * 2 + 1]);
        pml.pmlOuter[axis * 2 + 0] = Comm::allreduceMin(pml.pmlOuter[axis * 2 + 0]);
        pml.pmlOuter[axis * 2 + 1] = Comm::allreduceMax(pml.pmlOuter[axis * 2 + 1]);
    }

    // For multi-block mitered O-grid meshes (where adjacent PML blocks meet along
    // 45-degree miter planes), ensure any marched PML element whose nodes extend into
    // an orthogonal PML slab [pmlInner, pmlOuter] also enables conductivity along that axis.
    for (int elt = 0; elt < nelt; ++elt) {
        if (pml.pmlTag[elt] == 0) continue;
        int base = elt * nxyz;
        for (int axis = 0; axis < 3; ++axis) {
            const std::vector<double>& coordArr =
                (axis == 0) ? xm1 : ((axis == 1) ? ym1 : zm1);
            int negFace = axis * 2 + 0;
            int posFace = axis * 2 + 1;
            double minC =  pmlInf;
            double maxC = -pmlInf;
            for (int i = 0; i < nxyz; ++i) {
                double c = coordArr[base + i];
                minC = std::min(minC, c);
                maxC = std::max(maxC, c);
            }
            if (pml.pmlInner[negFace] > -0.5 * pmlInf &&
                minC < pml.pmlInner[negFace] - 1.0e-12) {
                pml.pmlTag[elt] |= (1 << negFace);
            }
            if (pml.pmlInner[posFace] < 0.5 * pmlInf &&
                maxC > pml.pmlInner[posFace] + 1.0e-12) {
                pml.pmlTag[elt] |= (1 << posFace);
            }
        }
    }
}

void BoundaryConditions::pmlCalcSigma(const Mesh& mesh, const PmlConfig& cfg, PmlData& pml) {
    // Matches pml_calc_sigma in contrib/NekCEM/src/cem_maxwell_pml.F (Taflove Eq. 7.55, 7.57)
    const int nelt = mesh.getNumElements();
    const int nxyz = mesh.getNumPointsPerElement();
    const int npts = mesh.getTotalPoints();

    pml.pmlSigma.assign(static_cast<size_t>(3) * npts, 0.0);
    pml.pmlPtr.clear();

    for (int ie = 0; ie < nelt; ++ie) {
        if (pml.pmlTag[ie] != 0) {
            pml.pmlPtr.push_back(ie);
        }
    }

    pml.maxPml = static_cast<int>(pml.pmlPtr.size());
    pml.maxPmlGlobal = Comm::allreduceSum(pml.maxPml);
    pml.enabled = (pml.maxPmlGlobal > 0);

    if (Comm::isRoot() && pml.enabled) {
        std::cout << "[PML] Initialized UPML absorbing layers (local/global elements: "
                  << pml.maxPml << " / " << pml.maxPmlGlobal
                  << ", thickness = " << cfg.thickness
                  << ", order = " << cfg.order
                  << ", R(0) = " << cfg.reflectErr << ")" << std::endl;
    }

    const auto& xm1 = mesh.getCoordX();
    const auto& ym1 = mesh.getCoordY();
    const auto& zm1 = mesh.getCoordZ();

    // Vacuum wave impedance eta = sqrt(mu / eps) = 1.0 in normalized units
    const double eta = 1.0;

    for (int e = 0; e < pml.maxPml; ++e) {
        int ie = pml.pmlPtr[e];
        for (int face = 0; face < 6; ++face) {
            int axis = face / 2;
            double width = std::abs(pml.pmlOuter[face] - pml.pmlInner[face]);

            if ((pml.pmlTag[ie] & (1 << face)) != 0 && width > 1.0e-14) {
                double sigmaMax = -(cfg.order + 1.0) * std::log(cfg.reflectErr) / (2.0 * eta * width);

                for (int i = 0; i < nxyz; ++i) {
                    int j = ie * nxyz + i;
                    double coord = (axis == 0) ? xm1[j] : ((axis == 1) ? ym1[j] : zm1[j]);
                    double zero2one = (coord - pml.pmlInner[face]) / (pml.pmlOuter[face] - pml.pmlInner[face]);
                    zero2one = std::max(0.0, std::min(1.0, zero2one));

                    double sigmaVal = sigmaMax * std::pow(zero2one, cfg.order);
                    pml.pmlSigma[axis * npts + j] = sigmaVal;
                }
            }
        }
    }
}

PmlData BoundaryConditions::initializePml(const Mesh& mesh, const PmlConfig& cfg) {
    PmlData pml;
    if (!hasPmlFaces(mesh)) {
        return pml;
    }
    if (!validatePmlConfig(cfg)) {
        return pml;
    }

    std::vector<double> faceary;
    pmlFillFaceArray(mesh, faceary, cfg.thickness);
    pmlExtentAndTags(mesh, faceary, pml);
    pmlCalcSigma(mesh, cfg, pml);

    return pml;
}
