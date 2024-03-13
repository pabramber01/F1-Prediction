import nbformat as nbf

import numpy as np

import sys


def build_check():
    ntbk = nbf.read("./src/main.ipynb", nbf.NO_CONVERT)

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


if __name__ == "__main__":
    print("=" * 88)
    print("Checking if notebook is executed...")
    if not build_check():
        print("FAILED: notebook is not executed as expected.")
        print("=" * 88)
        sys.exit(1)
    else:
        print("Success!")
        print("=" * 88)
