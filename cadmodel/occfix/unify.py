import os
import sys
import signal
import base64
import subprocess
from OCC.Core.BRepTools import breptools
from OCC.Core.TopoDS import TopoDS_Shape


def safe_unify_same_domain(shape: TopoDS_Shape, timeout: float = 60.0) -> TopoDS_Shape:
    encoded_data = base64.b64encode(breptools.WriteToString(shape).encode("utf-8"))

    try:
        proc = subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(input=encoded_data, timeout=timeout)

        if proc.returncode == 0 and stdout:
            return breptools.ReadFromString(base64.b64decode(stdout).decode("utf-8"))
    except subprocess.TimeoutExpired:
        proc.kill()

    return None


def _run_worker():
    def timeout_handler(signum, frame):
        sys.stderr.write("[Worker] Timeout: Force killing process after 10 minutes\n")
        sys.stderr.flush()
        os.kill(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)

    try:
        input_data = sys.stdin.buffer.read()
        shape = breptools.ReadFromString(base64.b64decode(input_data).decode("utf-8"))

        from OCC.Core.ShapeFix import ShapeFix_Shape
        from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        fixer = ShapeFix_Shape(shape)
        fixer.Perform()

        unifier = ShapeUpgrade_UnifySameDomain(fixer.Shape(), True, True, True)
        unifier.Build()

        result = breptools.WriteToString(unifier.Shape())
        sys.stdout.buffer.write(base64.b64encode(result.encode("utf-8")))
    except Exception as e:
        sys.stderr.write(f"[Worker] Failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _run_worker()
