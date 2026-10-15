"""Reproduce Paper 08 end to end (run from the package folder):
   1) experiment + figures  2) defence visualisation  3) the Word paper"""
import subprocess, sys, os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
for cmd in (["python3", "ideas.py", "8"], ["python3", "defend_visuals.py", "8"], ["python3", "paper08.py"]):
    subprocess.run(cmd, check=True)
