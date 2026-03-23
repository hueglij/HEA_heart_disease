"""
Build a single report.ipynb from all chapter notebooks.

Usage:
    python build_report.py

This creates report.ipynb in the project root by concatenating:
    - notebooks/00_introduction.ipynb
    - notebooks/01_eda.ipynb
    - notebooks/02_preprocessing.ipynb
    - notebooks/03_log_regression.ipynb
    - notebooks/04_grad_boost.ipynb
    - notebooks/05_model_comparison.ipynb

To export as HTML:
    jupyter nbconvert --to html report.ipynb
"""

import nbformat
import copy
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHAPTER_FILES = [
    "notebooks/00_introduction.ipynb",
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_log_regression.ipynb",
    "notebooks/04_grad_boost.ipynb",
    "notebooks/05_model_comparison.ipynb",
]


def fix_relative_paths(cell):
    """Adjust relative paths from notebooks/ subdirectory to project root."""
    new_cell = copy.deepcopy(cell)
    src = new_cell["source"]
    if isinstance(src, list):
        src = "".join(src)
    src = src.replace('"../results/', '"results/')
    src = src.replace("'../results/", "'results/")
    src = src.replace('"../data/', '"data/')
    src = src.replace("'../data/", "'data/")
    src = src.replace('os.path.abspath("..")', 'os.path.abspath(".")')
    new_cell["source"] = src
    return new_cell


def build_report():
    report = nbformat.v4.new_notebook()
    report.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    for chapter_path in CHAPTER_FILES:
        full_path = os.path.join(PROJECT_ROOT, chapter_path)
        print(f"  Adding {chapter_path} ...")
        nb = nbformat.read(full_path, as_version=4)

        for cell in nb.cells:
            fixed = fix_relative_paths(cell)

            if fixed["cell_type"] == "markdown":
                new_cell = nbformat.v4.new_markdown_cell(fixed["source"])
            else:
                new_cell = nbformat.v4.new_code_cell(fixed["source"])

            report.cells.append(new_cell)

    out_path = os.path.join(PROJECT_ROOT, "report.ipynb")
    nbformat.write(report, out_path)
    print(f"\nReport written to: {out_path}")
    print(f"Total cells: {len(report.cells)}")
    print(f"\nTo export as HTML:")
    print(f"  jupyter nbconvert --to html report.ipynb")


if __name__ == "__main__":
    print("Building report.ipynb from chapter notebooks...\n")
    build_report()
