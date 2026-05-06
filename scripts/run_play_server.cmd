@echo off
REM AWBW Flask play / replay UI — from repo root
cd /d "%~dp0.."
python -m server.app
if errorlevel 1 pause
