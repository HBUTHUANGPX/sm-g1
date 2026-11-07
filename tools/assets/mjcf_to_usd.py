"""Convert the G1 MJCF to USD with the standalone Isaac Sim MJCF importer (no Kit)."""
import argparse
from pathlib import Path
import os


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "g1",
                        help="Asset directory (default: repository data/g1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from pxr import Plug

    import isaacsim.asset.importer.utils.impl as _iu

    Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    src = str(args.data_root / "g1_description" / "g1_29dof.xml")
    dst = str(args.data_root / "g1_usd")
    cfg = MJCFImporterConfig(mjcf_path=os.path.normpath(src), usd_path=os.path.normpath(dst))
    print(MJCFImporter(cfg).import_mjcf())


if __name__ == "__main__":
    main()
