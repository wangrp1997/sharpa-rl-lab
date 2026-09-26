#!/bin/bash
# Clone IDTO at the pinned commit, drop this example in beside allegro_hand,
# and bring the Sharpa meshes in next to the URDF. Does not modify the solver.
set -euo pipefail

SHA=de0629c7811aa9b330e56c4385629005b09495f0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/build/idto"
MESH_SRC="${DEST}/../sharpa-urdf"

rm -rf "${DEST}"
git clone --depth 1 https://github.com/ToyotaResearchInstitute/idto.git "${DEST}"
cd "${DEST}"
got="$(git rev-parse HEAD)"
if [[ "${got}" != "${SHA}" ]]; then
  echo "IDTO HEAD ${got} is not the pinned ${SHA}" >&2
  exit 1
fi

mkdir -p "${DEST}/examples/sharpa_wave"
cp "${ROOT}/idto_sharpa/sharpa_wave.cc" "${DEST}/examples/sharpa_wave/"
cp "${ROOT}/idto_sharpa/sharpa_wave.yaml" "${DEST}/examples/sharpa_wave/"
cp "${ROOT}/idto_sharpa/CMakeLists.txt" "${DEST}/examples/sharpa_wave/CMakeLists.txt"
if ! grep -q "add_subdirectory(sharpa_wave)" "${DEST}/examples/CMakeLists.txt"; then
  printf '\nadd_subdirectory(sharpa_wave)\n' >> "${DEST}/examples/CMakeLists.txt"
fi

rm -rf "${MESH_SRC}"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/sharpa-robotics/sharpa-urdf-usd-xml.git "${MESH_SRC}"
cd "${MESH_SRC}"
git sparse-checkout set wave_01/right_sharpa_wave
mkdir -p "${DEST}/models/right_sharpa_wave"
cp "${MESH_SRC}/wave_01/right_sharpa_wave/package.xml" \
  "${MESH_SRC}/wave_01/right_sharpa_wave/right_sharpa_wave.urdf" \
  "${DEST}/models/right_sharpa_wave/"
cp -a "${MESH_SRC}/wave_01/right_sharpa_wave/meshes" "${DEST}/models/right_sharpa_wave/"
# Keep the URDF that was checked against the simulator.
cp "${ROOT}/assets/SharpaWave/right_sharpa_wave.urdf" \
  "${DEST}/models/right_sharpa_wave/right_sharpa_wave.urdf"

echo "prepared ${DEST}"
