import nbformat as nbf

import numpy as np

import sys
import os


def build_check():
    well_executed = True
    file = ""

    files = search_files()
    for f in files:
        if not file_check(f):
            well_executed = False
            file = f
            break

    return well_executed, file


def file_check(file):
    ntbk = nbf.read(file, nbf.NO_CONVERT)

    is_executed = True
    has_failed = False
    execution_order = np.array([], dtype=np.int32)

    for cell in ntbk.cells:
        if cell.cell_type == "code":
            if cell.execution_count is None:
                is_executed = False
                break
            elif len(cell.outputs) > 0:
                for output in cell.outputs:
                    if output.output_type == "error":
                        has_failed = True
                        break
                if has_failed:
                    break
            execution_order = np.append(execution_order, cell.execution_count)

    is_ordered = np.array_equal(execution_order, np.arange(1, len(execution_order) + 1))

    return is_executed & (not has_failed) & is_ordered


def search_files():
    res = []

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".ipynb"):
                res.append(os.path.join(root, file))

    return res


if __name__ == "__main__":
    print("=" * 88)
    print("Checking if notebook is executed...")
    well_executed, file = build_check()
    if well_executed:
        print("Success!")
        print("=" * 88)
    else:
        print(f"FAILED: {file} is not executed as expected.")
        print("=" * 88)
        sys.exit(1)
