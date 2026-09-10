"""Build a .ipynb from a plain-text cell file.

Cell file format: cells separated by lines starting with
    #%% md      -> markdown cell
    #%% code    -> code cell
"""
import json, sys, re

def build(src_path, out_path, kernel="python3"):
    text = open(src_path, encoding="utf-8").read()
    parts = re.split(r'^#%%[ \t]+(md|code)[ \t]*$', text, flags=re.M)
    # parts[0] is preamble (ignored); then alternating kind, body
    cells = []
    for kind, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if not body.strip():
            continue
        lines = body.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [], "source": source})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": kernel},
                       "language_info": {"name": "python", "version": "3.10.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out_path}: {len(cells)} cells")

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
