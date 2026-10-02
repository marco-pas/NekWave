#ifndef NW_SRC_HDF5_WRITER_HPP
#define NW_SRC_HDF5_WRITER_HPP

#include "mesh.hpp"
#include <string>
#include <vector>

/**
 * @file hdf5_writer.hpp
 * @brief High-performance binary HDF5 / XDMF time-series field writer for NekWave.
 *
 * Provides structured scientific dataset output:
 *   - /mesh/coordinates: [Npts, 3] physical collocation node positions (x, y, z)
 *   - /mesh/connectivity: [Nelt, 8] element-to-corner topological indices
 *   - /time_series/step_{N}/E: [Npts, 3] electric vector field (Ex, Ey, Ez)
 *   - /time_series/step_{N}/H: [Npts, 3] magnetic vector field (Hx, Hy, Hz)
 *   - Accompanied by a companion XDMF XML descriptor (.xmf) for zero-copy
 *     visualization directly in ParaView, VisIt, and PyVista.
 */
class Hdf5Writer {
public:
    Hdf5Writer();
    ~Hdf5Writer();

    // Prevent copy
    Hdf5Writer(const Hdf5Writer&) = delete;
    Hdf5Writer& operator=(const Hdf5Writer&) = delete;

    /**
     * @brief Precomputed physical spatial derivative fields (divergence and curl).
     */
    struct DerivedFields {
        const double* divE = nullptr;    // [npts] scalar divergence of E
        const double* divH = nullptr;    // [npts] scalar divergence of H
        const double* curlE = nullptr;   // [3 * npts] SoA vector curl of E
        const double* curlH = nullptr;   // [3 * npts] SoA vector curl of H

        DerivedFields() = default;
        DerivedFields(const double* dE, const double* dH, const double* cE, const double* cH)
            : divE(dE), divH(dH), curlE(cE), curlH(cH) {}
    };

    /**
     * @brief Initializes output files (HDF5 and companion XDMF, or VTU/PVD fallback).
     * @param h5Path Path to target .h5 file.
     * @param mesh Reference to the discretized spatial mesh.
     * @param enableXdmf Whether to generate companion XDMF (.xmf) descriptor.
     * @param exportContinuous Whether to average DG interface nodes into a continuous CG mesh.
     * @return true on success.
     */
    bool initialize(const std::string& h5Path, const Mesh& mesh, bool enableXdmf = true, bool exportContinuous = false);

    /**
     * @brief Selective field export options.
     */
    struct FieldSaveOptions {
        bool saveE = true;
        bool saveH = true;
        bool saveCurlE = false;
        bool saveCurlH = false;
        bool saveDivE = false;
        bool saveDivH = false;
        bool saveMagnitudeE = false;
        bool saveMagnitudeH = false;
        bool saveMagnitudeCurlE = false;
        bool saveMagnitudeCurlH = false;
        bool saveEnergyDensity = false;

        void enableAllExtras() {
            saveCurlE = true;
            saveCurlH = true;
            saveDivE = true;
            saveDivH = true;
            saveMagnitudeE = true;
            saveMagnitudeH = true;
            saveMagnitudeCurlE = true;
            saveMagnitudeCurlH = true;
            saveEnergyDensity = true;
        }
    };

    void setFieldSaveOptions(const FieldSaveOptions& opts);
    const FieldSaveOptions& getFieldSaveOptions() const;

    /**
     * @brief Appends field snapshot at a given time step.
     * @param step Time step iteration number.
     * @param time Physical simulation time.
     * @param state Contiguous state array [6 * npts] in SoA layout.
     * @param npts Number of spatial collocation points.
     * @param derived Optional pre-computed divergence and curl arrays.
     * @return true on success.
     */
    bool writeStep(int step, double time, const double* state, int npts, const DerivedFields* derived = nullptr);

    /**
     * @brief Finalizes and closes the HDF5 file and XML descriptor.
     */
    void close();

    /**
     * @brief Queries whether native HDF5 C library was compiled in.
     */
    static bool isSupported();

    /**
     * @brief Writes a standalone companion XDMF XML file for a given HDF5 dataset.
     */
    static bool writeXdmfDescriptor(
        const std::string& xmfPath,
        const std::string& h5BaseName,
        int npts,
        const std::vector<std::pair<int, double>>& stepTimes,
        int totalCells = 0,
        bool useHexCells = false,
        const FieldSaveOptions* options = nullptr
    );

    /**
     * @brief Writes a master companion XDMF XML file combining all MPI rank partitions.
     */
    static bool writeMasterXdmfDescriptor(
        const std::string& xmfPath,
        const std::string& baseStem,
        int numRanks,
        const std::vector<int>& allNpts,
        const std::vector<int>& allTotalCells,
        const std::vector<std::pair<int, double>>& stepTimes,
        bool useHexCells = false,
        const FieldSaveOptions* options = nullptr
    );

private:
    struct Impl;
    Impl* impl_;
};

#endif // NW_SRC_HDF5_WRITER_HPP
