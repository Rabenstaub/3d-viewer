@echo off
cd /d "%~dp0"
python S3D_Viewer.pyw %*
if errorlevel 1 pause
