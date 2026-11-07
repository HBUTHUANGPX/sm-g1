# Repository layout

The original ManagerBased SuperMimic implementation is the sole active training path.

- `mimic_baseline/source/mimic_baseline/mimic_baseline/tasks/supermimic/`: environment, managers and Trainer, including G1 adaptations.
- `mimic_baseline/source/`: shared source dependencies reviewed for publication.
- `scripts/`: public training, evaluation and experiment entry points.
- `mimic_baseline/scripts/`: compatibility entry-point wrappers; implementations are in the source package `cli/`.
- `mimic_baseline/tests/g1/`: CPU validation of the G1 integration and preparation tools.
- `tools/assets/`: pinned LAFAN1 download, URDF-to-MJCF and MJCF-to-USD preparation tools.
- `data/g1/`: default local preparation output; ignored by Git.
- `IsaacLab/`: pinned v3.0.0 upstream submodule; adaptations remain outside it.
- `licenses/`: retained upstream attribution.
- `docs/`: architecture, operating guides and compact experimental evidence.

The standalone implementation is available at [historical commit 4727620](https://github.com/HBUTHUANGPX/sm-g1/tree/4727620b31bf3af0bf43adcda1c27e2c815740e1/supermimic_g1). Its tracked source has been removed from the active tree. Existing ignored datasets, logs and checkpoints in older checkouts remain local and are not deleted by this cleanup. A historical result does not validate the active ManagerBased policy.

Run preparation commands from the repository root as shown in the [README](../README.md). Each preparation script accepts `--data-root PATH`; the same root must be supplied to all three scripts. Converter dependencies are loaded only when conversion runs. Download preparation fetches exactly 40 G1 LAFAN1 CSV clips at the pinned dataset revision, plus G1 robot and ground assets.

Publication uses `.github/mimic-publication-files.txt` for explicit source and tool approval, with `.github/check_repository.py` rejecting the removed standalone root and generated or sensitive artifacts. Private historical workspace material is outside this publication scope.
