@echo off
setlocal
cd /d "%~dp0"
set NAME=3D_Viewer
set VENV=.venv_build

echo === Saubere Build-Umgebung (nur die noetigen Pakete) ===
if not exist "%VENV%\Scripts\python.exe" (
  python -m venv "%VENV%" || goto :err
)
set PY=%VENV%\Scripts\python.exe
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt pyinstaller || goto :err

echo === Versionsinfo erzeugen ===
"%PY%" make_version.py || goto :err

echo === EXE bauen ===
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name %NAME% ^
  --icon S3D_Viewer.ico --add-data "S3D_Viewer.ico;." --version-file version_info.txt ^
  --hidden-import s3d_core --collect-submodules pyqtgraph.opengl ^
  --hidden-import OpenGL.platform.win32 --hidden-import OpenGL.arrays.numpymodule ^
  --exclude-module tkinter --exclude-module matplotlib --exclude-module scipy ^
  --exclude-module cupy --exclude-module numba --exclude-module torch --exclude-module bitsandbytes ^
  S3D_Viewer.pyw || goto :err

echo === Demo-Datei dazulegen ===
if exist "demo\Demo_Welle.s3d" copy /y "demo\Demo_Welle.s3d" "dist\Demo_Welle.s3d" >nul

echo === Signieren (Certum SimplySign) ===
set SIGNTOOL=
for /f "delims=" %%i in ('where signtool 2^>nul') do if not defined SIGNTOOL set "SIGNTOOL=%%i"
if not defined SIGNTOOL (
  for /f "delims=" %%i in ('dir /b /s /o-n "%ProgramFiles(x86)%\Windows Kits\10\bin\*signtool.exe" 2^>nul ^| findstr /i "\\x64\\"') do if not defined SIGNTOOL set "SIGNTOOL=%%i"
)
if not defined SIGNTOOL (
  echo signtool nicht gefunden - EXE bleibt unsigniert.
  goto :done
)
echo signtool: %SIGNTOOL%
"%SIGNTOOL%" sign /a /fd sha256 /tr http://time.certum.pl /td sha256 "dist\%NAME%.exe"
if errorlevel 1 echo Signieren fehlgeschlagen - SimplySign angemeldet?

:done
echo.
for %%A in ("dist\%NAME%.exe") do echo Fertig: dist\%NAME%.exe  (%%~zA Bytes)  + dist\Demo_Welle.s3d
pause
exit /b 0

:err
echo FEHLER beim Build.
pause
exit /b 1
