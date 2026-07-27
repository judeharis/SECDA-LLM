# SECDA-LLM
[WiP] SECDA-LLM platform, which integrated llama.cpp with the SECDA Design Methodology


```
git clone https://github.com/judeharis/SECDA-LLM.git
cd SECDA-LLM
git submodule init
git submodule update
./setup.sh
```

## CMake presets

The repository root is the CMake entrypoint for SECDA-LLM. It wraps the vendored `llama.cpp` project and exposes top-level configure, build, and test presets through `CMakePresets.json`.

### Configure

```bash
cmake --preset SECDA-sim-x64
cmake --preset SECDA-nosim-x64
cmake --preset x64-linux-gcc-debug
```

### Build

```bash
cmake --build --preset SECDA-sim-x64 -j
cmake --build --preset SECDA-nosim-x64 -j
cmake --build --preset x64-linux-gcc-debug -j
```

### Test

Native x86-64 presets also expose CTest presets:

```bash
ctest --preset SECDA-sim-x64
ctest --preset SECDA-nosim-x64
ctest --preset x64-linux-gcc-debug
```

### Available root presets

- `SECDA-sim-x64`: x86-64 SECDA simulation build with SECDA quantization options enabled.
- `SECDA-nosim-x64`: x86-64 build with the SECDA backend disabled.
- `SECDA-armv7-debug`: cross-compiled ARMv7 SECDA build.
- `SECDA-aarch64-debug`: cross-compiled AArch64 SECDA build.
- `x64-linux-gcc-debug`: native GCC debug build through the wrapper project.
- `x64-linux-gcc-release`: native GCC release build through the wrapper project.

The ARM presets are intended for configure and build steps. They do not have root `ctest` presets because those binaries are not expected to run on the local x86-64 host.