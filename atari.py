import argparse
import os

def run_sbatch(command, name, partition='gpu-2080ti', dry=False):
    cmd = f'sbatch -p {partition} --job-name {name} train.sbatch {command}'
    print(cmd)
    if not dry:
        os.system(cmd)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('games', nargs='+')
    parser.add_argument('--frames', type=int, default=10_000_000, help='Number of frames to train for')
    parser.add_argument('--model', default='nature-cnn', help='Model to use')
    parser.add_argument('--nsteps', type=int, default=2)

    parser.add_argument('-p', '--partition', default='gpu-2080ti', help='Partition to run on')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Dry run')
    args = parser.parse_args()

    for game in args.games:
        env = f'ALE/{game}-v5'
        for loss in ['crps', 'ndqn', 'td']:
            name = f'{game}-{args.model}-{loss}'
            command = f'python ice/train.py'
            command += f' -n "{name}"'
            command += f' -e "{env}"'
            command += f' -f {args.frames}'
            command += f' --model "{args.model}"'
            command += f' --nsteps {args.nsteps}'
            command += f' --loss {loss}'

            run_sbatch(command, name, args.partition, args.dry_run)

if __name__ == '__main__':
    main()
