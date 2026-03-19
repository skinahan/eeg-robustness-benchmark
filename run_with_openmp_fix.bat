@echo off
REM Batch script to run experiments with OpenMP fix
set KMP_DUPLICATE_LIB_OK=TRUE
python %*

