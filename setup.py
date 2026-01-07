from setuptools import setup
from setuptools.command.build_py import build_py
from pathlib import Path
import re


def defines_to_python(src, dst):
    src = Path(src)
    dst = Path(dst)

    INT_DEFINE = re.compile(
        r"#define\s+(\w+)\s+\(?\s*([0-9]+)\s*\)?"
    )

    REAL_CAST_DEFINE = re.compile(
        r"#define\s+(\w+)\s+\(\(REAL\)\s*([0-9eE\.\+\-]+)\s*\)"
    )

    defines = {}

    for line in src.read_text().splitlines():
        if m := INT_DEFINE.match(line):
            defines[m.group(1)] = int(m.group(2))
        elif m := REAL_CAST_DEFINE.match(line):
            defines[m.group(1)] = float(m.group(2))

    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w") as f:
        f.write("# AUTO-GENERATED FROM globals.h — DO NOT EDIT\n\n")
        for k, v in sorted(defines.items()):
            f.write(f"{k} = {v}\n")

class build_py_with_cuda(build_py):
    def run(self):
        root = Path(__file__).parent

        globals_src = str(root / "cuslines" / "globals.h")
        globals_dst = str(root / "cuslines" / "cuda_python" / "_globals.py")
        defines_to_python(globals_src, globals_dst)

        super().run()

setup(
    cmdclass={"build_py": build_py_with_cuda},
)
