module load openmpi/4.1.6-gcc-12.2.0-kvteryp 2>/dev/null || module load openmpi/4.1.4-gcc-12.2.0-77fifwj 2>/dev/null || true
module load hdf5/1.14.3-openmpi-4.1.6-gcc-12.2.0-2z7sqsi 2>/dev/null || module load hdf5/1.12.2-openmpi-4.1.4-gcc-11.3.0-2m5andk 2>/dev/null || true
export LD_LIBRARY_PATH="/lib64:${LD_LIBRARY_PATH}"