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

bool BoundaryConditions::hasPmlFaces(const Mesh& mesh) {
    for (const auto& f : mesh.getFaces()) {
        if (parseBcTag(f.bcType, f.neighborElementId) == BcType::PML) {
            return true;
        }
    }
    return false;
}

bool BoundaryConditions::validatePmlConfig(const PmlConfig& cfg) {
    // Matches pml_errchk in contrib/NekCEM/src/cem_maxwell_pml.F
    if (cfg.thickness < 1 || cfg.thickness > 100) {
        if (Comm::isRoot()) {
            std::cerr << "[PML ERROR] Invalid pml_thickness = " << cfg.thickness
                      << ". Must be between 1 and 100." << std::endl;
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

void BoundaryConditions::pmlFillFaceArray(const Mesh& mesh, std::vector<int>& faceLevel, int thick) {
    // Matches pml_fill_faceary in contrib/NekCEM/src/cem_maxwell_pml.F
    const int nelt = mesh.getNumElements();
    const int negFaces[3] = {3, 0, 4};
    const int posFaces[3] = {1, 2, 5};

    // March 1 integer level per face (6 * nelt ints)
    faceLevel.assign(static_cast<size_t>(nelt) * 6, 0);

    const auto& faces = mesh.getFaces();
    // Pre-collect interior shared face pairs (with eltA < eltB to touch each pair once per step)
    std::vector<std::pair<size_t, size_t>> interiorPairs;
    interiorPairs.reserve(faces.size() / 2);

    for (const auto& f : faces) {
        const BcType bc = parseBcTag(f.bcType, f.neighborElementId);
        if (bc == BcType::PML) {
            faceLevel[static_cast<size_t>(f.elementId) * 6 + f.faceId] = 1;
        } else if (bc == BcType::INTERIOR && f.neighborElementId >= 0 && f.neighborFaceId >= 0) {
            if (f.elementId < f.neighborElementId ||
                (f.elementId == f.neighborElementId && f.faceId < f.neighborFaceId)) {
                size_t idxA = static_cast<size_t>(f.elementId) * 6 + f.faceId;
                size_t idxB = static_cast<size_t>(f.neighborElementId) * 6 + f.neighborFaceId;
                interiorPairs.push_back({idxA, idxB});
            }
        }
    }

    // March inward through opposing hex faces `thick` times in O(nelt) using compact ints
    for (int step = 0; step < thick; ++step) {
        for (int elt = 0; elt < nelt; ++elt) {
            const size_t base = static_cast<size_t>(elt) * 6;
            for (int axis = 0; axis < 3; ++axis) {
                int& pVal = faceLevel[base + posFaces[axis]];
                int& nVal = faceLevel[base + negFaces[axis]];
                if (pVal + nVal != 0) {
                    pVal += 1;
                    nVal += 1;
                }
            }
        }
        for (const auto& pr : interiorPairs) {
            int m = std::max(faceLevel[pr.first], faceLevel[pr.second]);
            faceLevel[pr.first] = m;
            faceLevel[pr.second] = m;
        }
    }
}

int BoundaryConditions::dirLocalToGlobal(const Mesh& mesh, int elt, int symDir) {
    // Matches dir_local_to_global in contrib/NekCEM/src/cem_maxwell_pml.F
    // symDir is in 0..5 symmetric convention: 0=-r, 1=+r, 2=-s, 3=+s, 4=-t, 5=+t
    double locvec[3] = {0.0, 0.0, 0.0};
    int axis = symDir / 2;
    locvec[axis] = (symDir % 2 == 0) ? -1.0 : 1.0;

    // Evaluate trilinear Jacobian at corner 0 (r = -1, s = -1, t = -1) directly from corners
    const auto& c = mesh.getElementCorners()[elt];
    double xr = 0.5 * (c[1][0] - c[0][0]), yr = 0.5 * (c[1][1] - c[0][1]), zr = 0.5 * (c[1][2] - c[0][2]);
    double xs = 0.5 * (c[3][0] - c[0][0]), ys = 0.5 * (c[3][1] - c[0][1]), zs = 0.5 * (c[3][2] - c[0][2]);
    double xt = 0.5 * (c[4][0] - c[0][0]), yt = 0.5 * (c[4][1] - c[0][1]), zt = 0.5 * (c[4][2] - c[0][2]);
    double detJ = xr * (ys * zt - yt * zs) - xs * (yr * zt - yt * zr) + xt * (yr * zs - ys * zr);
    double invJ = (std::abs(detJ) > 1.0e-20) ? (1.0 / detJ) : 1.0;
    double rx0 = invJ * (ys * zt - yt * zs), sx0 = invJ * (yt * zr - yr * zt), tx0 = invJ * (yr * zs - ys * zr);
    double ry0 = invJ * (xt * zs - xs * zt), sy0 = invJ * (xr * zt - xt * zr), ty0 = invJ * (xs * zr - xr * zs);
    double rz0 = invJ * (xs * yt - xt * ys), sz0 = invJ * (xt * yr - xr * yt), tz0 = invJ * (xr * ys - xs * yr);

    double globvec[3];
    globvec[0] = rx0 * locvec[0] + sx0 * locvec[1] + tx0 * locvec[2];
    globvec[1] = ry0 * locvec[0] + sy0 * locvec[1] + ty0 * locvec[2];
    globvec[2] = rz0 * locvec[0] + sz0 * locvec[1] + tz0 * locvec[2];

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

void BoundaryConditions::pmlExtentAndTags(const Mesh& mesh, const std::vector<int>& faceLevel, PmlData& pml) {
    // Matches pml_extent_and_tags in contrib/NekCEM/src/cem_maxwell_pml.F
    const int nelt = mesh.getNumElements();

    // Mapping from symmetric face index (0=-r, 1=+r, 2=-s, 3=+s, 4=-t, 5=+t)
    // to NekWave / Ed's faceId (0..5)
    const int eface0[6] = {3, 1, 0, 2, 4, 5};
    const int oppSymFace[6] = {1, 0, 3, 2, 5, 4};

    const double pmlInf = 1.0e20;

    for (int axis = 0; axis < 3; ++axis) {
        pml.pmlOuter[axis * 2 + 0] =  pmlInf;
        pml.pmlInner[axis * 2 + 0] = -pmlInf;
        pml.pmlInner[axis * 2 + 1] =  pmlInf;
        pml.pmlOuter[axis * 2 + 1] = -pmlInf;
    }

    pml.pmlTag.assign(nelt, 0);

    const auto& allCorners = mesh.getElementCorners();

    for (int elt = 0; elt < nelt; ++elt) {
        for (int axis = 0; axis < 3; ++axis) {
            int symFace = axis * 2; // Negative symmetric face along local axis
            int farHere = faceLevel[static_cast<size_t>(elt) * 6 + eface0[symFace]];
            int farOpp  = faceLevel[static_cast<size_t>(elt) * 6 + eface0[oppSymFace[symFace]]];

            if (farHere != 0 && farOpp != 0) {
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

                double minCoord =  pmlInf;
                double maxCoord = -pmlInf;
                const auto& c = allCorners[elt];
                for (int a = 0; a < 8; ++a) {
                    double val = c[a][globAxis];
                    minCoord = std::min(minCoord, val);
                    maxCoord = std::max(maxCoord, val);
                }

                if (globSign * (farHere - farOpp) > 0) {
                    // Positive-side PML slab (+X, +Y, or +Z)
                    pml.pmlInner[posGlobFace] = std::min(pml.pmlInner[posGlobFace], minCoord);
                    pml.pmlOuter[posGlobFace] = std::max(pml.pmlOuter[posGlobFace], maxCoord);
                    pml.pmlTag[elt] |= (1 << posGlobFace);
                } else if (globSign * (farHere - farOpp) < 0) {
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
        const auto& c = allCorners[elt];
        for (int axis = 0; axis < 3; ++axis) {
            int negFace = axis * 2 + 0;
            int posFace = axis * 2 + 1;
            double minC =  pmlInf;
            double maxC = -pmlInf;
            for (int a = 0; a < 8; ++a) {
                double val = c[a][axis];
                minC = std::min(minC, val);
                maxC = std::max(maxC, val);
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
    // Builds compact PML element index list (pmlPtr); point-wise conductivity
    // profile d_pmlSigma is evaluated directly on the GPU via gpu_compute_pml_sigma_kernel.
    const int nelt = mesh.getNumElements();

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
}

PmlData BoundaryConditions::initializePml(const Mesh& mesh, const PmlConfig& cfg) {
    PmlData pml;
    if (!hasPmlFaces(mesh)) {
        return pml;
    }
    if (!validatePmlConfig(cfg)) {
        return pml;
    }

    std::vector<int> faceLevel;
    pmlFillFaceArray(mesh, faceLevel, cfg.thickness);
    pmlExtentAndTags(mesh, faceLevel, pml);
    pmlCalcSigma(mesh, cfg, pml);

    return pml;
}
