"""Run a saved search experiment configuration."""
import argparse
import shutil
import tomllib
from pathlib import Path

from experiments.components import run_components
from experiments.controlled_study import run_controlled
from experiments.isolated import run_isolated
from experiments.real_study import run_real_study, run_refinement
from experiments.exploration import run_exploration
from experiments.evaluate import run_evaluation
from experiments.probe_cutoffs import run_probe_cutoffs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text())
    stage = config.get('experiment', {}).get('stage', 'components')
    if stage == 'controlled':
        run_controlled(config, args.output)
    elif stage == 'probe_cutoffs':
        run_probe_cutoffs(config,args.output)
    elif stage == 'evaluate':
        run_evaluation(config, args.output)
    elif stage == 'exploration':
        run_exploration(config, args.output)
    elif stage == 'refine':
        run_refinement(config, args.output)
    elif stage == 'real':
        run_real_study(config, args.output)
    elif stage == 'isolated':
        run_isolated(config, args.output)
    elif stage == 'components':
        run_components(config, args.output)
    else:
        raise ValueError(f'unknown experiment stage: {stage}')
    shutil.copyfile(args.config, args.output/'config.toml')


if __name__ == '__main__':
    main()
