@echo off
cd /d "%~dp0"
if not exist .venv (py -3.12 -m venv .venv)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py --full-individual
python -m streamlit run app.py
pause
