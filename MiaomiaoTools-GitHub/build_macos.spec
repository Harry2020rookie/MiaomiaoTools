from pathlib import Path

project_root = Path(SPECPATH)
datas = [(str(project_root / "data" / name), "data/" + name)
         for name in ("icons", "templates", "template", "rules", "generated")]
datas.append((str(project_root / "NOTICE.md"), "."))
a = Analysis(
    [str(project_root / "app.py")], pathex=[str(project_root)],
    binaries=[], datas=datas, hiddenimports=[],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["pytest", "mss"], noarchive=False, optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="MiaomiaoTools",
    debug=False, strip=False, upx=False, console=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MiaomiaoTools")
app = BUNDLE(
    coll, name="妙妙工具.app", icon=str(project_root / "data/icons/miaomiao.icns"),
    bundle_identifier="io.github.mashiropro.miaomiaotools",
    info_plist={
        "CFBundleShortVersionString": "1.3.1",
        "CFBundleVersion": "5",
        "LSMinimumSystemVersion": "14.0",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "See bundled NOTICE.md",
    },
)
