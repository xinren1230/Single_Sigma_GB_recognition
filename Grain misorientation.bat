@echo off

CALL C:\Users\x.chen\Anaconda3\Scripts\activate.bat C:\Users\x.chen\Anaconda3

CALL conda activate base

CD /D "C:\Users\x.chen\Single_Sigma_GB_recognition"

python "Grain_misorientation.py"

pause
