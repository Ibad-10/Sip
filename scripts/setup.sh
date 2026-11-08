#!/usr/bin/env bash
# One-time setup on a fresh clone (Linux, macOS, or Git Bash on Windows). Needs: git, uv.
set -euo pipefail
cd "$(dirname "$0")/.."
MENAGERIE_COMMIT=98a70c569c4c1b6da467a20012f766ee1f8dc521   # pinned robot model version

uv sync
if [ ! -f third_party/mujoco_menagerie/hello_robot_stretch_3/stretch.xml ]; then
  rm -rf third_party/mujoco_menagerie
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
  git -C third_party/mujoco_menagerie sparse-checkout set hello_robot_stretch_3
  git -C third_party/mujoco_menagerie checkout -q "$MENAGERIE_COMMIT"
fi
[ -f .env ] || cp .env.example .env
uv run python -c "from sip.scene.build_house import build_house; import mujoco; mujoco.MjModel.from_xml_path(build_house()); print('scene OK')"
echo "Setup done. Optional: put GEMINI_API_KEY / ELEVENLABS_API_KEY in .env. Run: uv run python -m sip.app --request water"
