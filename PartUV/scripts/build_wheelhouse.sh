#!/usr/bin/env bash
set -euo pipefail

# ---- Config you can tweak ----------------------------------------------------
# Which Python ABIs to build for (auto-skips if not present in /opt/python).
PY_TAGS=("cp39" "cp310" "cp311" "cp312" "cp313")   # add/remove as you like

# Torch (CUDA 12.8) spec for each build. Leave empty if your project doesn't need it.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
TORCH_SPEC="torch==2.7.1"

# manylinux platform tag for this image; change to manylinux2014_x86_64 if using that image.
PLAT="manylinux_2_28_x86_64"

# Where the final, repaired wheels will go.
WHEELHOUSE="./wheelhouse"

# Optional: also build a dependency wheelhouse (for offline installation).
# BUILD_DEPS_WHEELHOUSE=true
# -----------------------------------------------------------------------------

mkdir -p "${WHEELHOUSE}"
dnf install -y microdnf

microdnf install -y curl ca-certificates tar gzip git which findutils && microdnf clean all

curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo \
  -o /etc/yum.repos.d/cuda.repo


  # refresh metadata (repo already added for rhel8)
microdnf makecache
microdnf install -y cuda-toolkit-12-8

# point /usr/local/cuda at 12.8
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda

# sanity check
/usr/local/cuda/bin/nvcc --version

/opt/python/cp310-cp310/bin/pip  install --upgrade pybind11[global] scikit-build-core ninja numpy

dnf -y install dnf-plugins-core epel-release
# dnf config-manager --set-enabled crb
dnf -y install CGAL-devel yaml-cpp-devel tbb-devel gmp-devel mpfr-devel boost-devel

# CUDA env (you already set this in your snippet; keep it here so each build sees it)
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH

# ---- minimal knobs -----------------------------------------------------------


# We are *not* building a dependency wheelhouse; just your wheel.
BUILD_DEPS_WHEELHOUSE=false
# -----------------------------------------------------------------------------

mkdir -p "${WHEELHOUSE}"

# CUDA env (if you’re compiling CUDA code)
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH

cmake_prefix_for_py() {
  local PYBIN="$1"; local PY="$PYBIN/python"
  "$PY" -m pybind11 --cmakedir 2>/dev/null || true
}

build_one() {
  local tag="$1"
  local PYBIN="/opt/python/${tag}-${tag}/bin"
  [[ -x "${PYBIN}/python" ]] || { echo "[skip] ${tag} missing"; return 0; }

  echo "=== ${tag} ==="
  rm -rf build dist _skbuild .ninja_log .ninja_deps

  # Only the build-time essentials.
  "${PYBIN}/pip" install -U --no-cache-dir pip wheel build auditwheel scikit-build-core ninja pybind11[global] numpy

  # Install torch only if you truly need its headers for compilation.
  if [[ -n "${TORCH_SPEC}" ]]; then
    "${PYBIN}/pip" install -U --no-cache-dir --index-url "${TORCH_INDEX}" "${TORCH_SPEC}"
  fi

  # Make pybind11’s CMake config visible to CMake.
  export CMAKE_PREFIX_PATH="$(cmake_prefix_for_py "${PYBIN}"):${CMAKE_PREFIX_PATH:-}"

  # Build the wheel without pulling in runtime deps or creating an isolated env.
  "${PYBIN}/python" -m build -w --no-isolation

  # Repair to produce manylinux-tagged wheels (bundles CGAL/TBB/GMP/MPFR/yaml-cpp, etc.).
  shopt -s nullglob
  for whl in dist/*.whl; do
    auditwheel repair --plat "${PLAT}" -w "${WHEELHOUSE}" "${whl}"
  done
  shopt -u nullglob

  echo "=== done ${tag} ==="
}

for t in "${PY_TAGS[@]}"; do build_one "$t"; done

echo "Repaired wheels:"
ls -1 "${WHEELHOUSE}" || true