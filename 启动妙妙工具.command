#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x "outputs/妙妙工具-实托邦.app/Contents/MacOS/MiaomiaoTools" ]]; then
    exec "outputs/妙妙工具-实托邦.app/Contents/MacOS/MiaomiaoTools"
fi
if [[ -x "outputs/妙妙工具-自动层数.app/Contents/MacOS/MiaomiaoTools" ]]; then
    exec "outputs/妙妙工具-自动层数.app/Contents/MacOS/MiaomiaoTools"
fi
if [[ -x "outputs/妙妙工具.app/Contents/MacOS/MiaomiaoTools" ]]; then
    exec "outputs/妙妙工具.app/Contents/MacOS/MiaomiaoTools"
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "请先按照 MACOS.md 安装 Python 3.11+ 并创建 .venv。"
    read -r -p "按回车关闭窗口…" _
    exit 1
fi
exec .venv/bin/python MiaomiaoTools-GitHub/app.py
