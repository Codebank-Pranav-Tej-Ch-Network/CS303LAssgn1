"""
Convenience script: runs C1, C2, C3 in sequence.

Run from the project root:
    python run_all.py
"""

import subprocess
import sys


def run(module):
    print(f"\n{'='*70}\nRUNNING {module}\n{'='*70}")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        print(f"\n!! {module} exited with an error (see above). Continuing... !!")


if __name__ == "__main__":
    run("src.c1_linear_regression")
    run("src.c2_polynomial_lasso")
    run("src.c3_logistic_regression")
    print("\nDone. Check the outputs/ directory for plots and report.txt files.")
