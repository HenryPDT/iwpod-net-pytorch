"""`iwpod export-trt`: ONNX -> TensorRT engine via trtexec (run on target).

Profiles derive from --size (min EXPORT_MIN_SIZE / opt <size> / max EXPORT_MAX_SIZE,
batch 1..--batch) and are emitted only for dynamic ONNX.
FP16 default on; INT8 needs a calibration cache/table (see --int8 --calib).
The builder log is captured next to the engine. Requires trtexec on PATH
(i.e. run this on the x86/DeepStream or Jetson target, not here).

Example:
  iwpod export-trt --onnx out/train/exp1/model.onnx --size 416 --fp16
  iwpod export-trt --onnx model.onnx --size 416 --batch 8 --int8 --calib calib.cache
"""
import os
import shutil
import subprocess

from iwpod.constants import EXPORT_MAX_SIZE, EXPORT_MIN_SIZE


def register(p):
    p.add_argument("--onnx", required=True, help="Input .onnx (from `iwpod export`)")
    p.add_argument("--size", nargs="+", type=int, default=[416],
                   help="Opt square [H,W]: one number = square, two = H W")
    p.add_argument("--batch", type=int, default=8, help="Max batch for profiles")
    p.add_argument("--fp16", action="store_true", default=True,
                   help="FP16 build (default on; pass --no-fp16 to disable)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--int8", action="store_true", help="INT8 build (needs --calib)")
    p.add_argument("--calib", default=None, help="INT8 calibration cache/table")
    p.add_argument("--workspace", type=int, default=2048, help="Workspace MiB")
    p.add_argument("-o", "--output", default=None, help="Engine path (default: <onnx>.engine)")
    p.add_argument("--trtexec", default="trtexec", help="trtexec binary")
    p.set_defaults(func=run)


def onnx_is_dynamic(path):
    try:
        import onnx
        m = onnx.load(path)
        for inp in m.graph.input:
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_param:
                    return True
    except Exception:
        return False
    return False


def _workspace_flag(trtexec, workspace_mib):
    """TRT 10 uses memPoolSize; TRT 8.5 uses --workspace=."""
    try:
        r = subprocess.run([trtexec, "--help"], capture_output=True, text=True, timeout=10)
        help_txt = (r.stdout or "") + (r.stderr or "")
    except Exception:
        help_txt = ""
    if "memPoolSize" in help_txt:
        return f"--memPoolSize=workspace:{workspace_mib}"
    return f"--workspace={workspace_mib}"


def build_command(args):
    from iwpod.cli.export import parse_size
    if shutil.which(args.trtexec) is None:
        raise RuntimeError(
            f"'{args.trtexec}' not found: run export-trt on a TensorRT machine "
            f"(x86 DeepStream container or Jetson), not here.")
    if not os.path.isfile(args.onnx):
        raise RuntimeError(f"Invalid onnx file: {args.onnx}")
    if args.int8 and not args.calib:
        raise RuntimeError("--int8 requires --calib <cache/table>")
    h, w = parse_size(args.size)
    engine = args.output or (os.path.splitext(args.onnx)[0] + ".engine")
    cmd = [args.trtexec, f"--onnx={args.onnx}", f"--saveEngine={engine}",
           _workspace_flag(args.trtexec, args.workspace)]
    if onnx_is_dynamic(args.onnx):
        min_h = min(h, EXPORT_MIN_SIZE)
        min_w = min(w, EXPORT_MIN_SIZE)
        max_h = max(h, EXPORT_MAX_SIZE)
        max_w = max(w, EXPORT_MAX_SIZE)
        cmd += [
            f"--minShapes=input:1x3x{min_h}x{min_w}",
            f"--optShapes=input:{args.batch}x3x{h}x{w}",
            f"--maxShapes=input:{args.batch}x3x{max_h}x{max_w}",
        ]
    elif args.batch != 1:
        print("Note: ONNX is static; ignoring TRT min/opt/max profiles "
              "(re-export with --dynamic to vary batch at runtime)")
    if args.fp16 and not args.int8:
        cmd.append("--fp16")
    if args.int8:
        cmd += ["--int8", f"--calib={args.calib}"]
    return cmd, engine


def run(args):
    cmd, engine = build_command(args)
    print("Running:", " ".join(cmd))
    log_path = os.path.splitext(engine)[0] + ".build.log"
    with open(log_path, "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"trtexec failed (rc={r.returncode}); see {log_path}")
    print(f"Done: {engine}\n  build log: {log_path}")
