@echo off
call "C:\Users\Sean\anaconda3\Scripts\activate.bat" ncp_robustness_proj
if errorlevel 1 (
    echo Error: Failed to activate environment.
    exit /b 1
)
echo Environment activated successfully.
pause