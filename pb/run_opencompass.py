import os
import os.path as osp

from datetime import datetime

import ipdb
import getpass

from pydantic import BaseModel
from typing import List, Union, Optional
from datetime import datetime
from mmengine.config import Config

from opencompass.runners import SlurmRunner
from opencompass.utils import get_logger, LarkReporter
from opencompass.utils.run import get_config_from_arg, exec_mm_infer_runner
from opencompass.utils.run import fill_infer_cfg, fill_eval_cfg
from opencompass.partitioners import MultimodalNaivePartitioner, SizePartitioner
from opencompass.registry import PARTITIONERS, RUNNERS, build_from_cfg
from opencompass.summarizers import DefaultSummarizer

from opencompass.utils.run import get_config_type
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import NDICLEvalTask, OpenICLInferTask

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


VALID_DATASET_ABBR = ["hellaswag", "bbh", "ARC_c", "ARC_e", "gsm8k", "humaneval",
                      "mbpp", "mmlu", "piqa", "siqa", "race", "squadv2", "superglue",
                      "winogrande", "xsum"]

WITH_SUBSETS = ["bbh", "mmlu", "superglue"]


class OpenCompassConfig(BaseModel):
    dry_run: bool = False
    debug: bool = False
    reuse: Optional[str] = None
    mode: str = "all"
    lark: bool = False
    dlc: bool = False
    slurm: bool = False
    max_partition_size: int = 40000
    gen_task_coef: int = 20
    max_num_workers: int = 10
    max_workers_per_gpu: int = 1
    partition: Optional[str] = None
    mm_eval: bool = False
    quotatype: Optional[str] = None
    dump_eval_details: bool = False
    config: str
    work_dir: str


class EvaluationConfig(OpenCompassConfig):
    db_url: str
    size: int
    seed: Union[int, str]


def fill_model_cfg(cfg: Config):
    for model in cfg.models:
        match model.abbr:
            case 'gpt-3.5-turbo':
                model["key"] = [OPENAI_API_KEY,]
            case 'gpt-4':
                model["key"] = [OPENAI_API_KEY,]
            case 'claude-2':
                model["key"] = CLAUDE_API_KEY
            case 'claude-2.1':
                model["key"] = CLAUDE_API_KEY
            case 'gemini-pro':
                model["key"] = GEMINI_API_KEY
            case _:
                raise NotImplementedError(f"{model.abbr} is not implemented.")
    return cfg


def fill_dataset_cfg(cfg: Config, db_url: str, size: int, seed: Union[int, str]):
    for dset in cfg.datasets:
        if not any([exc in dset.abbr for exc in WITH_SUBSETS]):
            assert dset.abbr in VALID_DATASET_ABBR, f"{dset.abbr} is not implemented"

        dset["db_url"] = db_url
        dset["size"] = size
        dset["seed"] = seed
    return cfg


def fill_infer_args(cfg, args):
    new_cfg = dict(infer=dict(
        partitioner=dict(type=get_config_type(NaivePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLInferTask)),
            lark_bot_url=cfg['lark_bot_url'],
        )), )
    if args.slurm:
        raise NotImplementedError("Slurm runner is not implemented yet.")
        # new_cfg['infer']['runner']['type'] = get_config_type(SlurmRunner)
        # new_cfg['infer']['runner']['partition'] = args.partition
        # new_cfg['infer']['runner']['quotatype'] = args.quotatype
        # new_cfg['infer']['runner']['qos'] = args.qos
        # new_cfg['infer']['runner']['retry'] = args.retry
    elif args.dlc:
        raise NotImplementedError("DLC runner is not implemented yet.")
        # new_cfg['infer']['runner']['type'] = get_config_type(DLCRunner)
        # new_cfg['infer']['runner']['aliyun_cfg'] = Config.fromfile(
        #     args.aliyun_cfg)
        # new_cfg['infer']['runner']['retry'] = args.retry
    else:
        new_cfg['infer']['runner']['type'] = get_config_type(LocalRunner)
        new_cfg['infer']['runner']['max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)


def fill_eval_args(cfg: Config, args: EvaluationConfig):
    new_cfg = dict(
        eval=dict(partitioner=dict(type=get_config_type(NaivePartitioner)),
                  runner=dict(
                      max_num_workers=args.max_num_workers,
                      debug=args.debug,
                      task=dict(type=get_config_type(NDICLEvalTask)),
                      lark_bot_url=cfg['lark_bot_url'],
                  ),
                  )
    )
    if args.slurm:
        raise NotImplementedError("Slurm runner is not implemented yet.")
        # new_cfg['eval']['runner']['type'] = get_config_type(SlurmRunner)
        # new_cfg['eval']['runner']['partition'] = args.partition
        # new_cfg['eval']['runner']['quotatype'] = args.quotatype
        # new_cfg['eval']['runner']['qos'] = args.qos
        # new_cfg['eval']['runner']['retry'] = args.retry
    elif args.dlc:
        raise NotImplementedError("DLC runner is not implemented yet.")
        # new_cfg['eval']['runner']['type'] = get_config_type(DLCRunner)
        # new_cfg['eval']['runner']['aliyun_cfg'] = Config.fromfile(args.aliyun_cfg)
        # new_cfg['eval']['runner']['retry'] = args.retry
    else:
        new_cfg['eval']['runner']['type'] = get_config_type(LocalRunner)
        new_cfg['eval']['runner']['max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)


def run_opencompass(cfg: Config, args: EvaluationConfig):
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', './outputs/default/')

    cfg = fill_dataset_cfg(cfg, args.db_url, args.size, args.seed)
    cfg = fill_model_cfg(cfg)

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                logger.warning('No previous results to reuse!')
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')
    elif args.mode in ['eval', 'viz']:
        raise ValueError('You must specify -r or --reuse when running in eval '
                         'or viz mode!')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}.py')
    cfg.dump(output_config_path)
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path, format_python_code=False)

    # report to lark bot if specify --lark
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        content = f'{getpass.getuser()}\'s task has been launched!'
        LarkReporter(cfg['lark_bot_url']).post(content)

    if args.mode in ['all', 'infer']:
        # When user have specified --slurm or --dlc, or have not set
        # "infer" in config, we will provide a default configuration
        # for infer
        if (args.dlc or args.slurm) and cfg.get('infer', None):
            logger.warning('You have set "infer" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "infer" configuration will be overridden by '
                           'your runtime arguments.')
        # Check whether run multimodal evaluation
        if args.mm_eval:
            partitioner = MultimodalNaivePartitioner(
                osp.join(cfg['work_dir'], 'predictions/'))
            tasks = partitioner(cfg)
            exec_mm_infer_runner(tasks, args, cfg)
            return

        if args.dlc or args.slurm or cfg.get('infer', None) is None:
            # fill_infer_cfg(cfg, args)
            fill_infer_args(cfg, args)

        if args.partition is not None:
            if RUNNERS.get(cfg.infer.runner.type) == SlurmRunner:
                cfg.infer.runner.partition = args.partition
                cfg.infer.runner.quotatype = args.quotatype
        else:
            logger.warning('SlurmRunner is not used, so the partition '
                           'argument is ignored.')
        if args.debug:
            cfg.infer.runner.debug = True
        if args.lark:
            cfg.infer.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'predictions/')
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.infer.runner)
        # Add extra attack config if exists
        if hasattr(cfg, 'attack'):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack
        runner(tasks)

    # evaluate
    if args.mode in ['all', 'eval']:
        # When user have specified --slurm or --dlc, or have not set
        # "eval" in config, we will provide a default configuration
        # for eval
        if (args.dlc or args.slurm) and cfg.get('eval', None):
            logger.warning('You have set "eval" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "eval" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('eval', None) is None:
            # fill_eval_cfg(cfg, args)
            fill_eval_args(cfg, args)
        if args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True

        if args.partition is not None:
            if RUNNERS.get(cfg.eval.runner.type) == SlurmRunner:
                cfg.eval.runner.partition = args.partition
                cfg.eval.runner.quotatype = args.quotatype
            else:
                logger.warning('SlurmRunner is not used, so the partition '
                               'argument is ignored.')
        if args.debug:
            cfg.eval.runner.debug = True
        if args.lark:
            cfg.eval.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.eval.partitioner['out_dir'] = osp.join(cfg['work_dir'], 'results/')
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.eval.runner)
        runner(tasks)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer_cfg = cfg.get('summarizer', {})
        if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
            summarizer_cfg['type'] = DefaultSummarizer
        summarizer_cfg['config'] = cfg
        summarizer = build_from_cfg(summarizer_cfg)
        summarizer.summarize(time_str=cfg_time_str)

    return dir_time_str
