import sys
from pathlib import Path

from ...git import git_hash
from .common import SubConfig, ConfigVar, ArgparseVar


class MetaConfig(SubConfig):
    trainer_cls = ConfigVar()
    project_dir = ArgparseVar(
        '--dir', 
        type=Path, 
        help='The project directory',
        parse_argument_fn=lambda self,args: self.parse_project_dir(args)
    )
    model_dir = ConfigVar()
    id_prefix = ArgparseVar(type=str, help='The id prefix for the experiment')
    project_name = ArgparseVar(type=str, help='The wandb project name')
    experiment_name = ArgparseVar(type=str, help='The wandb experiment name')
    command_line = ConfigVar()
    git_hash = ConfigVar()

    def set_defaults(self):
        self.trainer_cls = None
        self.project_dir = Path('./').resolve()
        self.model_dir = None
        self.id_prefix = ''
        self.project_name = None
        self.experiment_name = None
        self.command_line = ' '.join(sys.argv)
        self.git_hash = git_hash()

    def parse_project_dir(self, args):
        if args.dir is not None:
            self.project_dir = args.dir