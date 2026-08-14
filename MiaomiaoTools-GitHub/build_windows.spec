from pathlib import Path


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "data/icons"), "data/icons"),
    (str(project_root / "data/templates"), "data/templates"),
    (str(project_root / "data/template"), "data/template"),
    (str(project_root / "data/rules"), "data/rules"),
    (
        str(project_root / "data/generated/unknown_mystery_text.png"),
        "data/generated",
    ),
]

a = Analysis(
    [str(project_root / "app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=["mss"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="妙妙工具",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "data/icons/wrong-turn.ico"),
)
