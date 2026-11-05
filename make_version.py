"""Erzeugt version_info.txt (Windows-Dateieigenschaften der EXE) aus APP_VERSION/APP_NAME."""
import re, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "S3D_Viewer.pyw"), encoding="utf-8").read()
ver = re.search(r'APP_VERSION = "([^"]+)"', src).group(1)
name = re.search(r'APP_NAME = "([^"]+)"', src).group(1)
nums = [int(x) for x in re.findall(r"\d+", ver)][:4]
nums += [0] * (4 - len(nums))
t = tuple(nums)
txt = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040704b0', [
      StringStruct('CompanyName', 'Christian Diezinger'),
      StringStruct('FileDescription', '{name} - Viewer und Messwerkzeug fuer .s3d-Dateien'),
      StringStruct('FileVersion', '{ver}'),
      StringStruct('InternalName', '3D_Viewer'),
      StringStruct('LegalCopyright', '(c) 2026 Christian Diezinger - MIT-Lizenz'),
      StringStruct('OriginalFilename', '3D_Viewer.exe'),
      StringStruct('ProductName', '{name}'),
      StringStruct('ProductVersion', '{ver}')])]),
    VarFileInfo([VarStruct('Translation', [0x0407, 1200])])
  ]
)
"""
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "version_info.txt"), "w", encoding="utf-8").write(txt)
print(f"version_info.txt: {name} {ver}")
