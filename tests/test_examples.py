import sys, os
from pathlib import Path
from importlib import import_module


def test_examples():
    """Walks through the examples folder and tests each example by
    dynamically importing them which will also run their code.
    """
    # The training and analysis examples are not included since they require
    # installing quite large libraries and can take a significant amount of
    # time.
    exclude = {
        "./forces_and_energies/training_pytorch.py",
        "./forces_and_energies/training_tensorflow.py",
        "./forces_and_energies/analysis.py",
        "./clustering/training.py",
    }
    old_cwd = os.getcwd()
    os.chdir(Path(__file__).parent.parent / "examples")
    example_root = os.getcwd()
    paths = set()
    for root, dirs, files in list(os.walk("./")):
        files.sort()
        for f in files:
            filename = os.path.join(root, f)
            if filename.endswith(".py") and filename not in exclude:
                parts = filename[2:-3]
                parts = parts.rsplit("/", 1)
                if len(parts) != 1:
                    folder = parts[0]
                    cwd = "{}/{}".format(example_root, folder)
                else:
                    cwd = example_root
                os.chdir(cwd)
                modulename = parts[-1]
                if cwd not in paths:
                    sys.path.insert(0, cwd)
                    paths.add(cwd)
                import_module(modulename)
    os.chdir(old_cwd)
