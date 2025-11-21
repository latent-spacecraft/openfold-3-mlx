import os
from pathlib import Path

PLACEHOLDER_PATH = "placeholder"

def prep_deepspeed():
    # deepspeed requires the envvar set, but doesn't care about value
    if not os.environ["CUTLASS_PATH"]:
        os.environ["CUTLASS_PATH"] = os.environ.get("CUTLASS_PATH", PLACEHOLDER_PATH) 


def prep_cutlass():
    # Skip this step if cutlass package itself is available 
    if os.environ["CUTLASS_PATH"] != PLACEHOLDER_PATH:
        return

    # otherwise, apparently need to set the headers for cutlass
    import cutlass_library

    headers_dir = Path(cutlass_library.__file__).parent / "source/include"
    cpath = os.environ.get("CPATH", "")
    # TODO: technically, this test should be a little fancier
    if str(headers_dir.resolve()) not in cpath:
        if cpath:
            cpath += ":"

        os.environ["CPATH"] = cpath + str(headers_dir.resolve())
