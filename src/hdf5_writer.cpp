#include "hdf5_writer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef NEKWAVE_HAVE_HDF5
#include <hdf5.h>
#endif

struct Hdf5Writer::Impl {
    std::string outDir;
    std::string fileStem;
    std::string h5Path;
    std::string xmfPath;
    std::string pvdPath;
    std::string binPath;
    int npts = 0;
    int numElements = 0;
    int orderN = 0;
    int totalCells = 0;
    bool useHexCells = false;
    bool enableXdmf = true;
    bool exportContinuous = false;
    bool bInitialized = false;

    int numCgPoints = 0;
    std::vector<int> dgToCg;
    std::vector<double> cgMultiplicityInv;
    std::vector<double> cgCoords;
    std::vector<int> cellConnectivity;

    std::vector<std::pair<int, double>> recordedSteps;
    std::vector<double> meshCoords;

#ifdef NEKWAVE_HAVE_HDF5
    hid_t fileId = -1;
    hid_t timeSeriesGroup = -1;
#else
    std::ofstream binFile;
#endif

    ~Impl() {
        close();
    }

    void close() {
        if (!bInitialized) return;

#ifdef NEKWAVE_HAVE_HDF5
        if (timeSeriesGroup >= 0) {
            H5Gclose(timeSeriesGroup);
            timeSeriesGroup = -1;
        }
        if (fileId >= 0) {
            H5Fclose(fileId);
            fileId = -1;
        }
        if (enableXdmf && !xmfPath.empty()) {
            writeXdmfDescriptor(xmfPath, fileStem + ".h5", exportContinuous ? numCgPoints : npts, recordedSteps, totalCells, useHexCells);
        }
#else
        if (binFile.is_open()) {
            binFile.close();
        }
        writePvdDescriptor();
#endif

        bInitialized = false;
    }

    void writePvdDescriptor() {
        std::ofstream pvd(pvdPath);
        if (!pvd.is_open()) return;

        pvd << "<?xml version=\"1.0\"?>\n";
        pvd << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        pvd << "  <Collection>\n";
        for (const auto& st : recordedSteps) {
            int step = st.first;
            double time = st.second;
            std::string vtuName = fileStem + "_step_" + std::to_string(step) + ".vtu";
            pvd << "    <DataSet timestep=\"" << std::scientific << std::setprecision(8) << time 
                << "\" group=\"\" part=\"0\" file=\"" << vtuName << "\"/>\n";
        }
        pvd << "  </Collection>\n";
        pvd << "</VTKFile>\n";
        pvd.close();
    }

    void writeVtuStep(int step, const double* state, const DerivedFields* derived) {
        std::string vtuPath = outDir + "/" + fileStem + "_step_" + std::to_string(step) + ".vtu";
        std::ofstream vtu(vtuPath);
        if (!vtu.is_open()) return;

        int p = orderN - 1;
        bool useHexCells = (p >= 1 && numElements > 0);
        int cellsPerElem = p * p * p;
        int totalCells = useHexCells ? (numElements * cellsPerElem) : (exportContinuous ? numCgPoints : npts);
        int outPts = exportContinuous ? numCgPoints : npts;

        vtu << "<?xml version=\"1.0\"?>\n";
        vtu << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        vtu << "  <UnstructuredGrid>\n";
        vtu << "    <Piece NumberOfPoints=\"" << outPts << "\" NumberOfCells=\"" << totalCells << "\">\n";

        // Points
        vtu << "      <Points>\n";
        vtu << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        const std::vector<double>& ptCoords = exportContinuous ? cgCoords : meshCoords;
        for (int i = 0; i < outPts; ++i) {
            vtu << ptCoords[3 * i + 0] << " " 
                << ptCoords[3 * i + 1] << " " 
                << ptCoords[3 * i + 2] << "\n";
        }
        vtu << "        </DataArray>\n";
        vtu << "      </Points>\n";

        // PointData: Vectors and Scalars
        vtu << "      <PointData Vectors=\"E\" Scalars=\"magnitude_E\">\n";

        if (!exportContinuous) {
            // Raw Discontinuous Galerkin point fields
            // 1. E
            vtu << "        <DataArray type=\"Float64\" Name=\"E\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int i = 0; i < npts; ++i) {
                vtu << state[0 * npts + i] << " " 
                    << state[1 * npts + i] << " " 
                    << state[2 * npts + i] << "\n";
            }
            vtu << "        </DataArray>\n";

            // 2. H
            vtu << "        <DataArray type=\"Float64\" Name=\"H\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int i = 0; i < npts; ++i) {
                vtu << state[3 * npts + i] << " " 
                    << state[4 * npts + i] << " " 
                    << state[5 * npts + i] << "\n";
            }
            vtu << "        </DataArray>\n";

            // 3. curl_E
            if (derived && derived->curlE) {
                vtu << "        <DataArray type=\"Float64\" Name=\"curl_E\" NumberOfComponents=\"3\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    vtu << derived->curlE[0 * npts + i] << " " 
                        << derived->curlE[1 * npts + i] << " " 
                        << derived->curlE[2 * npts + i] << "\n";
                }
                vtu << "        </DataArray>\n";

                vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_curl_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    double cx = derived->curlE[0 * npts + i], cy = derived->curlE[1 * npts + i], cz = derived->curlE[2 * npts + i];
                    vtu << std::sqrt(cx * cx + cy * cy + cz * cz) << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 4. curl_H
            if (derived && derived->curlH) {
                vtu << "        <DataArray type=\"Float64\" Name=\"curl_H\" NumberOfComponents=\"3\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    vtu << derived->curlH[0 * npts + i] << " " 
                        << derived->curlH[1 * npts + i] << " " 
                        << derived->curlH[2 * npts + i] << "\n";
                }
                vtu << "        </DataArray>\n";

                vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_curl_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    double cx = derived->curlH[0 * npts + i], cy = derived->curlH[1 * npts + i], cz = derived->curlH[2 * npts + i];
                    vtu << std::sqrt(cx * cx + cy * cy + cz * cz) << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 5. div_E
            if (derived && derived->divE) {
                vtu << "        <DataArray type=\"Float64\" Name=\"div_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    vtu << derived->divE[i] << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 6. div_H
            if (derived && derived->divH) {
                vtu << "        <DataArray type=\"Float64\" Name=\"div_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int i = 0; i < npts; ++i) {
                    vtu << derived->divH[i] << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 7. magnitude_E
            vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int i = 0; i < npts; ++i) {
                double ex = state[0 * npts + i], ey = state[1 * npts + i], ez = state[2 * npts + i];
                vtu << std::sqrt(ex * ex + ey * ey + ez * ez) << "\n";
            }
            vtu << "        </DataArray>\n";

            // 8. magnitude_H
            vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int i = 0; i < npts; ++i) {
                double hx = state[3 * npts + i], hy = state[4 * npts + i], hz = state[5 * npts + i];
                vtu << std::sqrt(hx * hx + hy * hy + hz * hz) << "\n";
            }
            vtu << "        </DataArray>\n";

            // 9. energy_density
            vtu << "        <DataArray type=\"Float64\" Name=\"energy_density\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int i = 0; i < npts; ++i) {
                double ex = state[0 * npts + i], ey = state[1 * npts + i], ez = state[2 * npts + i];
                double hx = state[3 * npts + i], hy = state[4 * npts + i], hz = state[5 * npts + i];
                vtu << 0.5 * (ex * ex + ey * ey + ez * ez + hx * hx + hy * hy + hz * hz) << "\n";
            }
            vtu << "        </DataArray>\n";

        } else {
            // Continuous CG nodal field averaging across coincident element boundaries
            std::vector<double> cg_Ex(numCgPoints, 0.0), cg_Ey(numCgPoints, 0.0), cg_Ez(numCgPoints, 0.0);
            std::vector<double> cg_Hx(numCgPoints, 0.0), cg_Hy(numCgPoints, 0.0), cg_Hz(numCgPoints, 0.0);
            std::vector<double> cg_cEx(numCgPoints, 0.0), cg_cEy(numCgPoints, 0.0), cg_cEz(numCgPoints, 0.0);
            std::vector<double> cg_cHx(numCgPoints, 0.0), cg_cHy(numCgPoints, 0.0), cg_cHz(numCgPoints, 0.0);
            std::vector<double> cg_divE(numCgPoints, 0.0), cg_divH(numCgPoints, 0.0);

            for (int i = 0; i < npts; ++i) {
                int cg = dgToCg[i];
                cg_Ex[cg] += state[0 * npts + i];
                cg_Ey[cg] += state[1 * npts + i];
                cg_Ez[cg] += state[2 * npts + i];
                cg_Hx[cg] += state[3 * npts + i];
                cg_Hy[cg] += state[4 * npts + i];
                cg_Hz[cg] += state[5 * npts + i];

                if (derived) {
                    if (derived->divE) cg_divE[cg] += derived->divE[i];
                    if (derived->divH) cg_divH[cg] += derived->divH[i];
                    if (derived->curlE) {
                        cg_cEx[cg] += derived->curlE[0 * npts + i];
                        cg_cEy[cg] += derived->curlE[1 * npts + i];
                        cg_cEz[cg] += derived->curlE[2 * npts + i];
                    }
                    if (derived->curlH) {
                        cg_cHx[cg] += derived->curlH[0 * npts + i];
                        cg_cHy[cg] += derived->curlH[1 * npts + i];
                        cg_cHz[cg] += derived->curlH[2 * npts + i];
                    }
                }
            }

            for (int k = 0; k < numCgPoints; ++k) {
                double invW = cgMultiplicityInv[k];
                cg_Ex[k] *= invW; cg_Ey[k] *= invW; cg_Ez[k] *= invW;
                cg_Hx[k] *= invW; cg_Hy[k] *= invW; cg_Hz[k] *= invW;
                cg_divE[k] *= invW; cg_divH[k] *= invW;
                cg_cEx[k] *= invW; cg_cEy[k] *= invW; cg_cEz[k] *= invW;
                cg_cHx[k] *= invW; cg_cHy[k] *= invW; cg_cHz[k] *= invW;
            }

            // 1. E
            vtu << "        <DataArray type=\"Float64\" Name=\"E\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int k = 0; k < numCgPoints; ++k) {
                vtu << cg_Ex[k] << " " << cg_Ey[k] << " " << cg_Ez[k] << "\n";
            }
            vtu << "        </DataArray>\n";

            // 2. H
            vtu << "        <DataArray type=\"Float64\" Name=\"H\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for (int k = 0; k < numCgPoints; ++k) {
                vtu << cg_Hx[k] << " " << cg_Hy[k] << " " << cg_Hz[k] << "\n";
            }
            vtu << "        </DataArray>\n";

            // 3. curl_E
            if (derived && derived->curlE) {
                vtu << "        <DataArray type=\"Float64\" Name=\"curl_E\" NumberOfComponents=\"3\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    vtu << cg_cEx[k] << " " << cg_cEy[k] << " " << cg_cEz[k] << "\n";
                }
                vtu << "        </DataArray>\n";

                vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_curl_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    double cx = cg_cEx[k], cy = cg_cEy[k], cz = cg_cEz[k];
                    vtu << std::sqrt(cx * cx + cy * cy + cz * cz) << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 4. curl_H
            if (derived && derived->curlH) {
                vtu << "        <DataArray type=\"Float64\" Name=\"curl_H\" NumberOfComponents=\"3\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    vtu << cg_cHx[k] << " " << cg_cHy[k] << " " << cg_cHz[k] << "\n";
                }
                vtu << "        </DataArray>\n";

                vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_curl_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    double cx = cg_cHx[k], cy = cg_cHy[k], cz = cg_cHz[k];
                    vtu << std::sqrt(cx * cx + cy * cy + cz * cz) << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 5. div_E
            if (derived && derived->divE) {
                vtu << "        <DataArray type=\"Float64\" Name=\"div_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    vtu << cg_divE[k] << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 6. div_H
            if (derived && derived->divH) {
                vtu << "        <DataArray type=\"Float64\" Name=\"div_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
                for (int k = 0; k < numCgPoints; ++k) {
                    vtu << cg_divH[k] << "\n";
                }
                vtu << "        </DataArray>\n";
            }

            // 7. magnitude_E
            vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_E\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int k = 0; k < numCgPoints; ++k) {
                double ex = cg_Ex[k], ey = cg_Ey[k], ez = cg_Ez[k];
                vtu << std::sqrt(ex * ex + ey * ey + ez * ez) << "\n";
            }
            vtu << "        </DataArray>\n";

            // 8. magnitude_H
            vtu << "        <DataArray type=\"Float64\" Name=\"magnitude_H\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int k = 0; k < numCgPoints; ++k) {
                double hx = cg_Hx[k], hy = cg_Hy[k], hz = cg_Hz[k];
                vtu << std::sqrt(hx * hx + hy * hy + hz * hz) << "\n";
            }
            vtu << "        </DataArray>\n";

            // 9. energy_density
            vtu << "        <DataArray type=\"Float64\" Name=\"energy_density\" NumberOfComponents=\"1\" format=\"ascii\">\n";
            for (int k = 0; k < numCgPoints; ++k) {
                double ex = cg_Ex[k], ey = cg_Ey[k], ez = cg_Ez[k];
                double hx = cg_Hx[k], hy = cg_Hy[k], hz = cg_Hz[k];
                vtu << 0.5 * (ex * ex + ey * ey + ez * ez + hx * hx + hy * hy + hz * hz) << "\n";
            }
            vtu << "        </DataArray>\n";
        }
        vtu << "      </PointData>\n";

        // Cells: 3D Volumetric Micro-Hexahedra connecting GLL nodes
        vtu << "      <Cells>\n";
        vtu << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
        if (useHexCells) {
            for (int c = 0; c < totalCells; ++c) {
                int base = 8 * c;
                vtu << cellConnectivity[base + 0] << " " << cellConnectivity[base + 1] << " "
                    << cellConnectivity[base + 2] << " " << cellConnectivity[base + 3] << " "
                    << cellConnectivity[base + 4] << " " << cellConnectivity[base + 5] << " "
                    << cellConnectivity[base + 6] << " " << cellConnectivity[base + 7] << "\n";
            }
        } else {
            for (int i = 0; i < outPts; ++i) vtu << i << "\n";
        }
        vtu << "        </DataArray>\n";

        vtu << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
        if (useHexCells) {
            int curOffset = 0;
            for (int c = 0; c < totalCells; ++c) {
                curOffset += 8;
                vtu << curOffset << (c + 1 == totalCells ? "" : " ");
            }
        } else {
            for (int i = 1; i <= outPts; ++i) vtu << i << " ";
        }
        vtu << "\n        </DataArray>\n";

        vtu << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
        if (useHexCells) {
            for (int c = 0; c < totalCells; ++c) {
                vtu << 12 << (c + 1 == totalCells ? "" : " "); // 12 = VTK_HEXAHEDRON
            }
        } else {
            for (int i = 0; i < outPts; ++i) vtu << 1 << " ";
        }
        vtu << "\n        </DataArray>\n";
        vtu << "      </Cells>\n";

        vtu << "    </Piece>\n";
        vtu << "  </UnstructuredGrid>\n";
        vtu << "</VTKFile>\n";
        vtu.close();
    }
};


Hdf5Writer::Hdf5Writer() : impl_(new Impl()) {}

Hdf5Writer::~Hdf5Writer() {
    delete impl_;
}

bool Hdf5Writer::isSupported() {
#ifdef NEKWAVE_HAVE_HDF5
    return true;
#else
    return false;
#endif
}

bool Hdf5Writer::initialize(const std::string& h5Path, const Mesh& mesh, bool enableXdmf, bool exportContinuous) {
    impl_->close();

    impl_->npts = mesh.getTotalPoints();
    impl_->numElements = mesh.getNumElements();
    impl_->orderN = mesh.getN();
    impl_->enableXdmf = enableXdmf;
    impl_->exportContinuous = exportContinuous;
    impl_->recordedSteps.clear();

    // Determine directory and file stem
    size_t slash = h5Path.find_last_of("/\\");
    if (slash != std::string::npos) {
        impl_->outDir = h5Path.substr(0, slash);
        impl_->fileStem = h5Path.substr(slash + 1);
    } else {
        impl_->outDir = ".";
        impl_->fileStem = h5Path;
    }
    size_t dot = impl_->fileStem.find_last_of('.');
    if (dot != std::string::npos) {
        impl_->fileStem = impl_->fileStem.substr(0, dot);
    }

    impl_->h5Path = impl_->outDir + "/" + impl_->fileStem + ".h5";
    impl_->xmfPath = impl_->outDir + "/" + impl_->fileStem + ".xmf";
    impl_->binPath = impl_->outDir + "/" + impl_->fileStem + ".bin";
    impl_->pvdPath = impl_->outDir + "/" + impl_->fileStem + ".pvd";

    const int npts = impl_->npts;
    const auto& x = mesh.getCoordX();
    const auto& y = mesh.getCoordY();
    const auto& z = mesh.getCoordZ();

    impl_->meshCoords.resize(npts * 3);
    for (int i = 0; i < npts; ++i) {
        impl_->meshCoords[3 * i + 0] = x[i];
        impl_->meshCoords[3 * i + 1] = y[i];
        impl_->meshCoords[3 * i + 2] = z[i];
    }

    int p = impl_->orderN - 1;
    bool useHexCells = (p >= 1 && impl_->numElements > 0);
    int cellsPerElem = p * p * p;
    int totalCells = useHexCells ? (impl_->numElements * cellsPerElem) : impl_->npts;
    impl_->useHexCells = useHexCells;
    impl_->totalCells = totalCells;
    int nPtsPerElem = impl_->orderN * impl_->orderN * impl_->orderN;

    if (impl_->exportContinuous) {
        // Group coincident nodes across element interfaces into unique CG points
        struct PointRef {
            int idx;
            double x, y, z;
        };
        std::vector<PointRef> pts(impl_->npts);
        for (int i = 0; i < impl_->npts; ++i) {
            pts[i] = {i, impl_->meshCoords[3 * i + 0], impl_->meshCoords[3 * i + 1], impl_->meshCoords[3 * i + 2]};
        }
        std::sort(pts.begin(), pts.end(), [](const PointRef& a, const PointRef& b) {
            return a.x < b.x;
        });

        impl_->dgToCg.assign(impl_->npts, -1);
        impl_->cgCoords.clear();
        std::vector<int> cgCounts;
        const double eps = 1e-6;
        const double eps2 = eps * eps;

        for (size_t s = 0; s < pts.size(); ++s) {
            int origIdx = pts[s].idx;
            if (impl_->dgToCg[origIdx] != -1) continue;

            int newCg = static_cast<int>(cgCounts.size());
            impl_->dgToCg[origIdx] = newCg;
            impl_->cgCoords.push_back(pts[s].x);
            impl_->cgCoords.push_back(pts[s].y);
            impl_->cgCoords.push_back(pts[s].z);
            cgCounts.push_back(1);

            for (size_t f = s + 1; f < pts.size(); ++f) {
                if (pts[f].x - pts[s].x > eps) break;
                int candIdx = pts[f].idx;
                if (impl_->dgToCg[candIdx] != -1) continue;
                double dy = pts[f].y - pts[s].y;
                double dz = pts[f].z - pts[s].z;
                if (std::abs(dy) <= eps && std::abs(dz) <= eps) {
                    double dx = pts[f].x - pts[s].x;
                    if (dx * dx + dy * dy + dz * dz <= eps2) {
                        impl_->dgToCg[candIdx] = newCg;
                        cgCounts[newCg]++;
                    }
                }
            }
        }
        impl_->numCgPoints = static_cast<int>(cgCounts.size());
        impl_->cgMultiplicityInv.resize(impl_->numCgPoints);
        for (int k = 0; k < impl_->numCgPoints; ++k) {
            impl_->cgMultiplicityInv[k] = 1.0 / static_cast<double>(cgCounts[k]);
        }

        std::cout << "[VTK] Continuous CG mesh export enabled: " << impl_->npts
                  << " DG nodes -> " << impl_->numCgPoints << " unique conforming CG nodes." << std::endl;
    } else {
        impl_->numCgPoints = impl_->npts;
    }

    // Build precomputed cell connectivity
    impl_->cellConnectivity.clear();
    if (useHexCells) {
        impl_->cellConnectivity.reserve(totalCells * 8);
        for (int e = 0; e < impl_->numElements; ++e) {
            int elemOffset = e * nPtsPerElem;
            for (int ck = 0; ck < p; ++ck) {
                for (int cj = 0; cj < p; ++cj) {
                    for (int ci = 0; ci < p; ++ci) {
                        int p0 = elemOffset + ci     + impl_->orderN * (cj     + impl_->orderN * ck);
                        int p1 = elemOffset + (ci+1) + impl_->orderN * (cj     + impl_->orderN * ck);
                        int p2 = elemOffset + (ci+1) + impl_->orderN * (cj+1   + impl_->orderN * ck);
                        int p3 = elemOffset + ci     + impl_->orderN * (cj+1   + impl_->orderN * ck);
                        int p4 = elemOffset + ci     + impl_->orderN * (cj     + impl_->orderN * (ck+1));
                        int p5 = elemOffset + (ci+1) + impl_->orderN * (cj     + impl_->orderN * (ck+1));
                        int p6 = elemOffset + (ci+1) + impl_->orderN * (cj+1   + impl_->orderN * (ck+1));
                        int p7 = elemOffset + ci     + impl_->orderN * (cj+1   + impl_->orderN * (ck+1));

                        if (impl_->exportContinuous) {
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p0]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p1]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p2]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p3]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p4]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p5]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p6]);
                            impl_->cellConnectivity.push_back(impl_->dgToCg[p7]);
                        } else {
                            impl_->cellConnectivity.push_back(p0);
                            impl_->cellConnectivity.push_back(p1);
                            impl_->cellConnectivity.push_back(p2);
                            impl_->cellConnectivity.push_back(p3);
                            impl_->cellConnectivity.push_back(p4);
                            impl_->cellConnectivity.push_back(p5);
                            impl_->cellConnectivity.push_back(p6);
                            impl_->cellConnectivity.push_back(p7);
                        }
                    }
                }
            }
        }
    } else {
        impl_->cellConnectivity.reserve(totalCells);
        for (int i = 0; i < totalCells; ++i) {
            impl_->cellConnectivity.push_back(i);
        }
    }

#ifdef NEKWAVE_HAVE_HDF5
    impl_->fileId = H5Fcreate(impl_->h5Path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (impl_->fileId < 0) {
        std::cerr << "[HDF5 ERROR] Could not create HDF5 file: " << impl_->h5Path << std::endl;
        return false;
    }

    hid_t scalarSpace = H5Screate(H5S_SCALAR);
    int verMajor = 1, verMinor = 0;
    hid_t attrMajor = H5Acreate2(impl_->fileId, "nekwave_version_major", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrMajor, H5T_NATIVE_INT, &verMajor);
    H5Aclose(attrMajor);

    hid_t attrMinor = H5Acreate2(impl_->fileId, "nekwave_version_minor", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrMinor, H5T_NATIVE_INT, &verMinor);
    H5Aclose(attrMinor);

    int orderVal = impl_->orderN;
    hid_t attrOrder = H5Acreate2(impl_->fileId, "polynomial_order_N", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrOrder, H5T_NATIVE_INT, &orderVal);
    H5Aclose(attrOrder);

    int neltVal = impl_->numElements;
    hid_t attrNelt = H5Acreate2(impl_->fileId, "num_elements", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrNelt, H5T_NATIVE_INT, &neltVal);
    H5Aclose(attrNelt);

    int totalPtsVal = impl_->npts;
    hid_t attrPts = H5Acreate2(impl_->fileId, "total_points", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrPts, H5T_NATIVE_INT, &totalPtsVal);
    H5Aclose(attrPts);

    H5Sclose(scalarSpace);

    hid_t meshGroup = H5Gcreate2(impl_->fileId, "/mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t coordDims[2] = {static_cast<hsize_t>(npts), 3};
    hid_t coordSpace = H5Screate_simple(2, coordDims, NULL);
    hid_t coordDset = H5Dcreate2(meshGroup, "coordinates", H5T_IEEE_F64LE, coordSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(coordDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, impl_->meshCoords.data());
    H5Dclose(coordDset);
    H5Sclose(coordSpace);

    hsize_t jacDims[1] = {static_cast<hsize_t>(npts)};
    hid_t jacSpace = H5Screate_simple(1, jacDims, NULL);
    hid_t jacDset = H5Dcreate2(meshGroup, "jacobian", H5T_IEEE_F64LE, jacSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(jacDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mesh.getJac().data());
    H5Dclose(jacDset);
    H5Sclose(jacSpace);

    if (impl_->useHexCells && impl_->totalCells > 0) {
        hsize_t connDims[2] = {static_cast<hsize_t>(impl_->totalCells), 8};
        hid_t connSpace = H5Screate_simple(2, connDims, NULL);
        hid_t connDset = H5Dcreate2(meshGroup, "connectivity", H5T_STD_I32LE, connSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(connDset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, impl_->cellConnectivity.data());
        H5Dclose(connDset);
        H5Sclose(connSpace);
    }

    H5Gclose(meshGroup);

    impl_->timeSeriesGroup = H5Gcreate2(impl_->fileId, "/time_series", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::cout << "[HDF5] Initialized binary HDF5 archive: " << impl_->h5Path 
              << " (" << npts << " collocation nodes)" << std::endl;
#else
    impl_->binFile.open(impl_->binPath, std::ios::binary | std::ios::trunc);
    if (!impl_->binFile.is_open()) {
        std::cerr << "[IO ERROR] Could not create binary archive: " << impl_->binPath << std::endl;
        return false;
    }

    const char magic[16] = "NEKWAVE_BIN_V2";
    impl_->binFile.write(magic, 16);
    int32_t header[4] = {impl_->orderN, impl_->numElements, impl_->npts, 0};
    impl_->binFile.write(reinterpret_cast<const char*>(header), sizeof(header));
    impl_->binFile.write(reinterpret_cast<const char*>(impl_->meshCoords.data()), npts * 3 * sizeof(double));
    impl_->binFile.flush();

    std::cout << "[IO NOTICE] Native HDF5 C library not linked at compile time.\n"
              << "            Writing ParaView VTK collection: " << impl_->pvdPath << "\n"
              << "            Writing IEEE-754 binary archive: " << impl_->binPath << "\n"
              << "            (Open " << impl_->pvdPath << " directly in ParaView without any plugins)" << std::endl;
#endif

    impl_->bInitialized = true;
    return true;
}

bool Hdf5Writer::writeStep(int step, double time, const double* state, int npts, const DerivedFields* derived) {
    if (!impl_->bInitialized) return false;
    if (npts != impl_->npts) return false;

    impl_->recordedSteps.push_back({step, time});

#ifdef NEKWAVE_HAVE_HDF5
    std::string stepName = "step_" + std::to_string(step);
    hid_t stepGroup = H5Gcreate2(impl_->timeSeriesGroup, stepName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (stepGroup < 0) return false;

    hid_t scalarSpace = H5Screate(H5S_SCALAR);
    hid_t attrTime = H5Acreate2(stepGroup, "time", H5T_IEEE_F64LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrTime, H5T_NATIVE_DOUBLE, &time);
    H5Aclose(attrTime);

    hid_t attrStep = H5Acreate2(stepGroup, "step", H5T_STD_I32LE, scalarSpace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attrStep, H5T_NATIVE_INT, &step);
    H5Aclose(attrStep);
    H5Sclose(scalarSpace);

    hsize_t fieldDims[2] = {static_cast<hsize_t>(npts), 3};
    hid_t fieldSpace = H5Screate_simple(2, fieldDims, NULL);

    hid_t dsetE = H5Dcreate2(stepGroup, "E", H5T_IEEE_F64LE, fieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> bufE(npts * 3);
    for (int i = 0; i < npts; ++i) {
        bufE[3 * i + 0] = state[0 * npts + i];
        bufE[3 * i + 1] = state[1 * npts + i];
        bufE[3 * i + 2] = state[2 * npts + i];
    }
    H5Dwrite(dsetE, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bufE.data());
    H5Dclose(dsetE);

    hid_t dsetH = H5Dcreate2(stepGroup, "H", H5T_IEEE_F64LE, fieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> bufH(npts * 3);
    for (int i = 0; i < npts; ++i) {
        bufH[3 * i + 0] = state[3 * npts + i];
        bufH[3 * i + 1] = state[4 * npts + i];
        bufH[3 * i + 2] = state[5 * npts + i];
    }
    H5Dwrite(dsetH, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bufH.data());
    H5Dclose(dsetH);

    if (derived) {
        if (derived->curlE) {
            hid_t dsetCE = H5Dcreate2(stepGroup, "curl_E", H5T_IEEE_F64LE, fieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            std::vector<double> bufCE(npts * 3);
            for (int i = 0; i < npts; ++i) {
                bufCE[3 * i + 0] = derived->curlE[0 * npts + i];
                bufCE[3 * i + 1] = derived->curlE[1 * npts + i];
                bufCE[3 * i + 2] = derived->curlE[2 * npts + i];
            }
            H5Dwrite(dsetCE, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bufCE.data());
            H5Dclose(dsetCE);
        }
        if (derived->curlH) {
            hid_t dsetCH = H5Dcreate2(stepGroup, "curl_H", H5T_IEEE_F64LE, fieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            std::vector<double> bufCH(npts * 3);
            for (int i = 0; i < npts; ++i) {
                bufCH[3 * i + 0] = derived->curlH[0 * npts + i];
                bufCH[3 * i + 1] = derived->curlH[1 * npts + i];
                bufCH[3 * i + 2] = derived->curlH[2 * npts + i];
            }
            H5Dwrite(dsetCH, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, bufCH.data());
            H5Dclose(dsetCH);
        }

        hsize_t scalarDims[1] = {static_cast<hsize_t>(npts)};
        hid_t scalarFieldSpace = H5Screate_simple(1, scalarDims, NULL);
        if (derived->divE) {
            hid_t dsetDivE = H5Dcreate2(stepGroup, "div_E", H5T_IEEE_F64LE, scalarFieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetDivE, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, derived->divE);
            H5Dclose(dsetDivE);
        }
        if (derived->divH) {
            hid_t dsetDivH = H5Dcreate2(stepGroup, "div_H", H5T_IEEE_F64LE, scalarFieldSpace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dsetDivH, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, derived->divH);
            H5Dclose(dsetDivH);
        }
        H5Sclose(scalarFieldSpace);
    }

    H5Sclose(fieldSpace);
    H5Gclose(stepGroup);

    H5Fflush(impl_->fileId, H5F_SCOPE_LOCAL);
#else
    int32_t stepHeader[2] = {step, 0};
    impl_->binFile.write(reinterpret_cast<const char*>(stepHeader), sizeof(stepHeader));
    impl_->binFile.write(reinterpret_cast<const char*>(&time), sizeof(double));

    std::vector<double> bufE(npts * 3);
    std::vector<double> bufH(npts * 3);
    for (int i = 0; i < npts; ++i) {
        bufE[3 * i + 0] = state[0 * npts + i];
        bufE[3 * i + 1] = state[1 * npts + i];
        bufE[3 * i + 2] = state[2 * npts + i];
        bufH[3 * i + 0] = state[3 * npts + i];
        bufH[3 * i + 1] = state[4 * npts + i];
        bufH[3 * i + 2] = state[5 * npts + i];
    }
    impl_->binFile.write(reinterpret_cast<const char*>(bufE.data()), npts * 3 * sizeof(double));
    impl_->binFile.write(reinterpret_cast<const char*>(bufH.data()), npts * 3 * sizeof(double));

    int32_t hasDerived = (derived && derived->divE && derived->divH && derived->curlE && derived->curlH) ? 1 : 0;
    impl_->binFile.write(reinterpret_cast<const char*>(&hasDerived), sizeof(int32_t));
    if (hasDerived) {
        impl_->binFile.write(reinterpret_cast<const char*>(derived->divE), npts * sizeof(double));
        impl_->binFile.write(reinterpret_cast<const char*>(derived->divH), npts * sizeof(double));
        impl_->binFile.write(reinterpret_cast<const char*>(derived->curlE), 3 * npts * sizeof(double));
        impl_->binFile.write(reinterpret_cast<const char*>(derived->curlH), 3 * npts * sizeof(double));
    }
    impl_->binFile.flush();

    impl_->writeVtuStep(step, state, derived);
    impl_->writePvdDescriptor();
#endif

    return true;
}

void Hdf5Writer::close() {
    impl_->close();
}

bool Hdf5Writer::writeXdmfDescriptor(
    const std::string& xmfPath,
    const std::string& h5BaseName,
    int npts,
    const std::vector<std::pair<int, double>>& stepTimes,
    int totalCells,
    bool useHexCells)
{
    std::ofstream xmf(xmfPath);
    if (!xmf.is_open()) return false;

    xmf << "<?xml version=\"1.0\" ?>\n";
    xmf << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    xmf << "<Xdmf Version=\"3.0\">\n";
    xmf << "  <Domain>\n";
    xmf << "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (const auto& st : stepTimes) {
        int step = st.first;
        double time = st.second;

        xmf << "      <Grid Name=\"step_" << step << "\" GridType=\"Uniform\">\n";
        xmf << "        <Time Value=\"" << std::scientific << std::setprecision(8) << time << "\"/>\n";
        if (useHexCells && totalCells > 0) {
            xmf << "        <Topology TopologyType=\"Hexahedron\" NumberOfElements=\"" << totalCells << "\">\n";
            xmf << "          <DataItem Dimensions=\"" << totalCells << " 8\" NumberType=\"Int\" Precision=\"4\" Format=\"HDF\">\n";
            xmf << "            " << h5BaseName << ":/mesh/connectivity\n";
            xmf << "          </DataItem>\n";
            xmf << "        </Topology>\n";
        } else {
            xmf << "        <Topology TopologyType=\"Polyvertex\" NumberOfElements=\"" << npts << "\"/>\n";
        }
        xmf << "        <Geometry GeometryType=\"XYZ\">\n";
        xmf << "          <DataItem Dimensions=\"" << npts << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
        xmf << "            " << h5BaseName << ":/mesh/coordinates\n";
        xmf << "          </DataItem>\n";
        xmf << "        </Geometry>\n";

        xmf << "        <Attribute Name=\"E\" AttributeType=\"Vector\" Center=\"Node\">\n";
        xmf << "          <DataItem Dimensions=\"" << npts << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
        xmf << "            " << h5BaseName << ":/time_series/step_" << step << "/E\n";
        xmf << "          </DataItem>\n";
        xmf << "        </Attribute>\n";

        xmf << "        <Attribute Name=\"H\" AttributeType=\"Vector\" Center=\"Node\">\n";
        xmf << "          <DataItem Dimensions=\"" << npts << " 3\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n";
        xmf << "            " << h5BaseName << ":/time_series/step_" << step << "/H\n";
        xmf << "          </DataItem>\n";
        xmf << "        </Attribute>\n";

        xmf << "      </Grid>\n";
    }

    xmf << "    </Grid>\n";
    xmf << "  </Domain>\n";
    xmf << "</Xdmf>\n";

    xmf.close();
    return true;
}
