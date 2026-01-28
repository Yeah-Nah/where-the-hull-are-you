#!/usr/bin/env python3
"""
Convert YOLOv8 .pt model to .blob format for OAK-D cameras.

This script:
1. Loads a YOLOv8 .pt model
2. Exports it to ONNX format
3. Converts ONNX to OpenVINO IR format
4. Compiles to .blob format for Myriad X (OAK-D)
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def convert_pt_to_blob(
    pt_model_path: str,
    output_dir: str = "src/models",
    img_size: tuple | list = (512, 384),
    shaves: int = 6,
    openvino_version: str = "2022.1",
):
    """
    Convert YOLOv8 .pt model to .blob format.

    Args:
        pt_model_path: Path to the .pt model file
        output_dir: Directory to save output files (default: src/models)
        img_size: Input image size as (W, H) tuple (default: (512, 384))
        shaves: Number of SHAVE cores for MyriadX (default: 6)
        openvino_version: OpenVINO version to use (default: 2022.1)
    """
    pt_path = Path(pt_model_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_name = pt_path.stem

    print(f"[1/3] Loading YOLOv8 model: {pt_path}")
    model = YOLO(str(pt_path))

    # Step 1: Export to ONNX
    # Note: img_size is in (W, H) format as specified by user requirements
    print(f"[2/3] Exporting to ONNX format (img_size={img_size})...")
    onnx_path = output_path / f"{model_name}.onnx"
    model.export(format="onnx", imgsz=img_size, simplify=True, opset=12)

    # The export creates the file in the same directory as the .pt file
    # Move it to output directory if needed
    default_onnx = pt_path.parent / f"{model_name}.onnx"
    if default_onnx.exists() and default_onnx != onnx_path:
        default_onnx.rename(onnx_path)

    print(f"   ✓ ONNX model saved: {onnx_path}")

    # Step 2: Convert ONNX to OpenVINO IR
    print("[3/3] Converting ONNX to .blob format...")
    print("   Note: This requires blobconverter package")

    try:
        import blobconverter

        blob_path = output_path / f"{model_name}.blob"

        # Convert using blobconverter
        blobconverter.from_onnx(
            model=str(onnx_path),
            output_dir=str(output_path),
            shaves=shaves,
            version=openvino_version,
            use_cache=False,
        )

        print(f"   ✓ Blob model saved: {blob_path}")
        print("\n✅ Conversion complete!")
        print("\nOutput files:")
        print(f"   - ONNX: {onnx_path}")
        print(f"   - BLOB: {blob_path}")

    except ImportError:
        print("\n⚠️  blobconverter not installed.")
        print("\nTo complete the conversion, install blobconverter:")
        print("   pip install blobconverter")
        print("\nThen run:")
        print(
            f"   python -c \"import blobconverter; blobconverter.from_onnx('{onnx_path}', output_dir='{output_path}', shaves={shaves})\""
        )
        print(f"\nONNX model has been created at: {onnx_path}")

    except Exception as e:
        print(f"\n❌ Error during blob conversion: {e}")
        print(f"\nONNX model has been created at: {onnx_path}")
        print("You can try converting manually using the DepthAI tools:")
        print("https://tools.luxonis.com/")
        sys.exit(1)


def main():
    """Run pipeline to convert .pt model to blob format."""
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 .pt model to .blob format for OAK-D cameras"
    )
    parser.add_argument("model_path", type=str, help="Path to the .pt model file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="src/models",
        help="Output directory (default: src/models)",
    )
    parser.add_argument(
        "-s",
        "--img-size",
        type=int,
        nargs=2,
        default=[512, 384],
        metavar=("WIDTH", "HEIGHT"),
        help="Input image size as WIDTH HEIGHT (default: 512 384)",
    )
    parser.add_argument(
        "--shaves",
        type=int,
        default=6,
        help="Number of SHAVE cores for MyriadX (default: 6)",
    )
    parser.add_argument(
        "--openvino-version",
        type=str,
        default="2022.1",
        help="OpenVINO version (default: 2022.1)",
    )

    args = parser.parse_args()

    # Validate input file exists
    if not Path(args.model_path).exists():
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)

    convert_pt_to_blob(
        pt_model_path=args.model_path,
        output_dir=args.output_dir,
        img_size=args.img_size,
        shaves=args.shaves,
        openvino_version=args.openvino_version,
    )


if __name__ == "__main__":
    main()
