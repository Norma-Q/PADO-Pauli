# Korean notebooks (*_ko.ipynb) must run the same code as their English originals: compare the code
# cells of every pair as Python ASTs (comments and markdown may differ). Run: python tutorial/check_ko_pairs.py
import ast, glob, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
bad = 0
for ko in sorted(glob.glob(os.path.join(here, "*_ko.ipynb"))):
    en = ko.replace("_ko.ipynb", ".ipynb")
    a = [c for c in json.load(open(en, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    b = [c for c in json.load(open(ko, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    same = len(a) == len(b) and all(ast.dump(ast.parse("".join(x["source"]))) == ast.dump(ast.parse("".join(y["source"]))) for x, y in zip(a, b))
    print(f"{os.path.basename(en)} vs {os.path.basename(ko)}: {len(a)} code cells, {'same code' if same else 'CODE DIFFERS'}")
    bad += not same
sys.exit(1 if bad else 0)
