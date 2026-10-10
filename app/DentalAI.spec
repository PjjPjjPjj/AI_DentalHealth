# PyInstaller spec for Dental AI Assistant (Windows target).
#
# Build on a Windows machine (Python 3.10):
#   pip install -r app/requirements_app.txt pyinstaller
#   pyinstaller app/DentalAI.spec --noconfirm
# Output: dist/DentalAI/DentalAI.exe  (folder distribution; zip and ship)
#
# The build machine needs NO CUDA/PyTorch - the app runs on onnxruntime.
# END USERS need nothing: the dist folder is self-contained.

import sys
from pathlib import Path

APP = Path(SPECPATH)          # app/
ROOT = APP.parent

block_cipher = None

# keep only the Qt binding we actually use on this platform
_GUI_EXCLUDE = "PyQt5" if sys.platform == "win32" else "PySide6"

datas = [
    (str(APP / "models"), "app/models"),
    (str(APP / "fonts"), "app/fonts"),
    (str(APP / "demo"), "app/demo"),
]

hiddenimports = [
    "app.qt_compat",
]

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # nothing from the training stack should sneak in
        "torch", "torchvision", "ultralytics", "matplotlib", "pandas",
        "IPython", "jupyter", "PyQt6", _GUI_EXCLUDE,
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DentalAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # GUI app, no console window
    icon=str(APP / "assets" / "icon.ico") if (APP / "assets" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="DentalAI",
)
