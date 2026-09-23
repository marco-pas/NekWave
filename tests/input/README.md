# Test Input Fixtures

This directory contains read-only input parameter files (`.par`) used by the automated test suite (`nekwave-test`) and CTest regression runs (`CavityGaussianRun`, `CavityEigenmodeRun`).

## Guidelines
- **Read-Only**: Test fixtures in this directory must not be modified during test execution.
- **No Output Pollution**: All test configurations specify `output_dir = none` and `export_fields = false` to guarantee that test runs do not generate any disk artifacts or mutate existing data.

