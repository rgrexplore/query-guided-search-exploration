"""Run the small component study from one explicit configuration."""
import argparse
import shutil
import tomllib
from pathlib import Path

from experiments.components import run_components


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    config=tomllib.loads(args.config.read_text())
    if config['data']['kind'] != 'random_signs':
        raise ValueError('this first component runner supports random_signs only')
    run_components(config, args.output)
    shutil.copyfile(args.config, args.output/'config.toml')


if __name__ == '__main__':
    main()
