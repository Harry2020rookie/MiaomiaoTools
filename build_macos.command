#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYINSTALLER_CONFIG_DIR="$PWD/work/pyinstaller-cache"
if [[ "$(uname -s)" != Darwin ]]; then
    echo "请在 macOS 上构建。"
    exit 1
fi
.venv/bin/python -m PyInstaller --noconfirm --clean \
    --distpath work/macos-release --workpath work/pyinstaller \
    MiaomiaoTools-GitHub/build_macos.spec
mkdir -p outputs
ditto "work/macos-release/妙妙工具.app" "outputs/妙妙工具-实托邦.app"
codesign --verify --deep --strict "outputs/妙妙工具-实托邦.app"
echo "构建完成：outputs/妙妙工具-实托邦.app"
