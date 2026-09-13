"""Entry point: `iwpod <train|eval|infer|export> [options]`."""
import argparse

from iwpod.cli import evaluate as _evaluate
from iwpod.cli import export as _export
from iwpod.cli import export_trt as _export_trt
from iwpod.cli import infer as _infer
from iwpod.cli import train as _train


def main(argv=None):
    ap = argparse.ArgumentParser(prog="iwpod", description="IWPOD-NET v2 quad LP detector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _train.register(sub.add_parser("train", help="Train (v2 loss, AMP, cosine, EMA)"))
    _evaluate.register(sub.add_parser("eval", help="Quad-IoU eval on annotated crops"))
    _infer.register(sub.add_parser("infer", help="Headless folder inference to quads + plates"))
    _export.register(sub.add_parser("export", help="Export ONNX (DeepStream-ready)"))
    _export_trt.register(sub.add_parser("export-trt", help="Build TensorRT engine via trtexec (on target)"))
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
