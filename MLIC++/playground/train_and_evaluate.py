import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import torch
import logging
from PIL import ImageFile, Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from utils.logger import setup_logger
from utils.utils import CustomDataParallel, save_checkpoint
from utils.optimizers import configure_optimizers
from utils.training import train_one_epoch
from utils.testing import test_one_epoch, test_model
from loss.rd_loss import RateDistortionLoss
from config.args import train_options
import argparse
from config.config import model_config
from models import *
from torchvision.transforms import functional as F
from pathlib import Path
import json
import glob

def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    parser = argparse.ArgumentParser(description="Train and evaluate script.")
    parser.add_argument('--train_dataset', type=str, required=False, help='Path to training dataset (required unless --eval-only)')
    # Add all train_options arguments except dataset
    parser.add_argument('-exp', '--experiment', default="mlicplus0018mse", type=str, help="Experiment name")
    parser.add_argument('-e', '--epochs', default=500, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument('-n', '--num_workers', type=int, default=8, help="Dataloaders threads (default: %(default)s)")
    # Accept either a single float or a tuple/list-like string e.g. "(0.0018,0.003)" or "0.0018,0.003"
    parser.add_argument('--lambda', dest='lmbda', type=str, default='0.0018', help="Lambda for rate-distortion loss. Use a single float or a comma/tuple-style list")
    parser.add_argument('--metrics', type=str, default="mse", help="Optimized for (default: %(default)s)")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument('--test-batch-size', type=int, default=1, help="Test batch size (default: %(default)s)")
    parser.add_argument('--aux-learning-rate', type=float, default=1e-3, help="Aux learning rate (default: %(default)s)")
    parser.add_argument('--patch-size', type=int, default=256, help="Patch size (default: %(default)s)")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--cuda', default=True, type=bool, help="Use cuda")
    parser.add_argument('--save', default=True, type=bool, help="Save model to disk")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--clip_max_norm', type=float, default=1.0, help="Clip max norm")
    parser.add_argument('-c', '--checkpoint', default=None, type=str, help="pretrained model path")
    parser.add_argument('--world_size', type=int, default=1, help="World size for distributed training")
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:8888', help="URL for distributed training")
    # Evaluation options
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training for each lambda')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, skip training')
    parser.add_argument('--eval-dataset', type=str, default=None, help='Path to evaluation dataset')
    parser.add_argument('--eval-batch-size', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--eval-num-workers', type=int, default=4, help='Evaluation dataloader workers')
    parser.add_argument('--eval-checkpoint', type=str, default=None, help='Checkpoint path to use for evaluation (overrides best)')
    parser.add_argument('--eval-save-dir', type=str, default=None, help='Directory to save evaluation results (default: <experiment>/eval_results)')
    args = parser.parse_args()
    config = model_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if hasattr(args, 'seed') and args.seed is not None:
        seed = args.seed
    else:
        seed = 100 * torch.rand(1).item()
    torch.manual_seed(int(seed))
    import random
    random.seed(int(seed))

    # Validate dataset arguments according to modes:
    # - If --eval-only is set, --train_dataset is not required but --eval-dataset must be provided
    # - If not eval-only, training requires --train_dataset
    if args.eval_only:
        if args.eval_dataset is None:
            parser.error("--eval-only requires --eval-dataset to be specified (path to eval images)")
    else:
        # not eval-only -> training mode
        if args.train_dataset is None:
            parser.error("Training mode requires --train_dataset to be specified (or run with --eval-only)")

    # Parse lambda(s): support string forms like "(0.001, 0.002)", "0.001,0.002" or single "0.0018"
    def parse_lambdas(s):
        if isinstance(s, (list, tuple)):
            return [float(x) for x in s]
        raw = str(s).strip()
        if raw.startswith('(') and raw.endswith(')'):
            raw = raw[1:-1]
        parts = [p.strip() for p in raw.split(',') if p.strip() != '']
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"Unable to parse --lambda value: {s}")

    lmbda_list = parse_lambdas(args.lmbda)

    # experiments root folder
    experiments_root = './experiments'
    if not os.path.exists(experiments_root):
        os.makedirs(experiments_root)

    # If eval-only mode, allow running evaluation by only providing --eval-dataset and --experiment
    if args.eval_only:
        # require eval dataset
        if args.eval_dataset is None:
            parser.error("--eval-only requires --eval-dataset to be specified (path to eval images)")

        # find experiment folders matching the experiment name prefix
        pattern = os.path.join(experiments_root, f"{args.experiment}*")
        exp_dirs = sorted(glob.glob(pattern))
        # filter to directories only
        exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

        # if no matching experiment directories, try exact match under experiments_root
        if not exp_dirs:
            direct = os.path.join(experiments_root, args.experiment)
            if os.path.isdir(direct):
                exp_dirs = [direct]

        # If the user provided a single explicit checkpoint, evaluate that and exit
        if args.eval_checkpoint is not None and not exp_dirs:
            # evaluate the single checkpoint
            exp_dir = os.path.join(experiments_root, args.experiment)
            ckpt_path = args.eval_checkpoint
            if args.eval_save_dir:
                eval_subdir = os.path.basename(os.path.normpath(args.eval_save_dir))
                save_dir = os.path.join(exp_dir, eval_subdir)
            else:
                save_dir = os.path.join(exp_dir, 'eval_results')
            os.makedirs(save_dir, exist_ok=True)
            setup_logger(f'test_{args.experiment}', save_dir, 'test_log', level=logging.INFO, screen=True, tofile=True)
            logger_test = logging.getLogger(f'test_{args.experiment}')
            # load model and run evaluation
            config_local = config
            net_eval = MLICPlusPlus(config_local)
            net_eval = net_eval.to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get('state_dict', ckpt)
            net_eval.load_state_dict(state)
            test_dataset = ImageFolder(args.eval_dataset, split='test', transform=transforms.ToTensor())
            try:
                test_dataset.samples.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)
            except Exception:
                pass
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                shuffle=False,
                pin_memory=(device == 'cuda')
            )
            test_model(test_dataloader, net_eval, logger_test, save_dir, ckpt.get('epoch', 0))
            # write summary
            summary = {
                'experiment': args.experiment,
                'checkpoint': os.path.abspath(ckpt_path),
                'eval_save_dir': os.path.abspath(save_dir),
            }
            try:
                summary_path = os.path.join(exp_dir, 'eval_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger_test.info(f'Evaluation summary written to {summary_path}')
            except Exception as e:
                logger_test.warning(f'Could not write evaluation summary: {e}')
            return

        if not exp_dirs:
            parser.error(f'No experiment folders matching "{args.experiment}" found under {experiments_root}.')

        # iterate matching experiment dirs and evaluate their best checkpoint
        for exp_dir in exp_dirs:
            exp_name = os.path.basename(exp_dir)
            ckpt_dir = os.path.join(exp_dir, 'checkpoints')
            candidate = os.path.join(ckpt_dir, 'checkpoint_best_loss.pth.tar')
            if os.path.exists(candidate):
                ckpt_path = candidate
            else:
                checkpoints = sorted(Path(ckpt_dir).glob('checkpoint_*.pth.tar'))
                ckpt_path = str(checkpoints[-1]) if checkpoints else None
            if ckpt_path is None or not os.path.exists(ckpt_path):
                print(f"No checkpoint found for {exp_name} under {ckpt_dir} -> skipping")
                continue
            if args.eval_save_dir:
                eval_subdir = os.path.basename(os.path.normpath(args.eval_save_dir))
                save_dir = os.path.join(exp_dir, eval_subdir)
            else:
                save_dir = os.path.join(exp_dir, 'eval_results')
            os.makedirs(save_dir, exist_ok=True)
            setup_logger(f'test_{exp_name}', save_dir, 'test_log', level=logging.INFO, screen=True, tofile=True)
            logger_test = logging.getLogger(f'test_{exp_name}')
            # load model
            config_local = config
            net_eval = MLICPlusPlus(config_local)
            net_eval = net_eval.to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get('state_dict', ckpt)
            net_eval.load_state_dict(state)
            # prepare dataloader
            test_dataset = ImageFolder(args.eval_dataset, split='test', transform=transforms.ToTensor())
            try:
                test_dataset.samples.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)
            except Exception:
                pass
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                shuffle=False,
                pin_memory=(device == 'cuda')
            )
            # run evaluation
            test_model(test_dataloader, net_eval, logger_test, save_dir, ckpt.get('epoch', 0))
            # write summary
            summary = {
                'experiment': exp_name,
                'checkpoint': os.path.abspath(ckpt_path),
                'checkpoint_epoch': int(ckpt.get('epoch', 0)) if ckpt is not None else None,
                'checkpoint_loss': float(ckpt.get('loss')) if ckpt is not None and 'loss' in ckpt else None,
                'eval_save_dir': os.path.abspath(save_dir),
            }
            try:
                summary_path = os.path.join(exp_dir, 'eval_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger_test.info(f'Evaluation summary written to {summary_path}')
            except Exception as e:
                logger_test.warning(f'Could not write evaluation summary: {e}')
        return

    def safe_random_crop(image, size):
        """Safely perform a random crop, ensuring the crop size is not larger than the image size."""
        if image.size[0] < size[0] or image.size[1] < size[1]:
            # Resize the image to at least the crop size while maintaining aspect ratio
            image = F.resize(image, size)
        return F.crop(image, *transforms.RandomCrop.get_params(image, size))

    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: safe_random_crop(img, (args.patch_size, args.patch_size))),
        transforms.ToTensor()
    ])

    def safe_center_crop(image, size):
        """Safely perform a center crop, ensuring the crop size is not larger than the image size."""
        if image.size[0] < size[0] or image.size[1] < size[1]:
            # Resize the image to at least the crop size while maintaining aspect ratio
            image = F.resize(image, size)
        return F.center_crop(image, size)

    test_transforms = transforms.Compose([
        transforms.Lambda(lambda img: safe_center_crop(img, (args.patch_size, args.patch_size))),
        transforms.ToTensor()
    ])

    # Only prepare training dataloaders when not running eval-only
    if not args.eval_only:
        train_dataset = ImageFolder(args.train_dataset, split="train", transform=train_transforms)
        val_dataset = ImageFolder(args.train_dataset, split="valid", transform=test_transforms)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )
    else:
        train_dataloader = None
        val_dataloader = None

    # Loop over lambda values and create one experiment per lambda
    import torch.optim as optim
    import random as _random
    for l in lmbda_list:
        # format lambda string for folder/logger names
        l_str = ('%g' % l).replace('.', 'p').replace('-', 'm')
        exp_name = f"{args.experiment}_{l_str}"
        exp_dir = os.path.join(experiments_root, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        # per-lambda loggers and tensorboard
        setup_logger(f'train_{l_str}', exp_dir, 'train_' + exp_name, level=logging.INFO, screen=True, tofile=True)
        setup_logger(f'val_{l_str}', exp_dir, 'val_' + exp_name, level=logging.INFO, screen=True, tofile=True)
        logger_train = logging.getLogger(f'train_{l_str}')
        logger_val = logging.getLogger(f'val_{l_str}')
        tb_logger = SummaryWriter(log_dir=os.path.join('./tb_logger', exp_name))

        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        # Evaluation helper (moved up so we can support eval-only without building/train)
        def run_evaluation(checkpoint_path=None):
            # Determine checkpoint
            if checkpoint_path is None:
                # choose best checkpoint if exists
                candidate = os.path.join(ckpt_dir, 'checkpoint_best_loss.pth.tar')
                if os.path.exists(candidate):
                    checkpoint_path_local = candidate
                else:
                    # fallback to last epoch checkpoint
                    checkpoints = sorted(Path(ckpt_dir).glob('checkpoint_*.pth.tar'))
                    checkpoint_path_local = str(checkpoints[-1]) if checkpoints else None
            else:
                checkpoint_path_local = checkpoint_path

            if checkpoint_path_local is None or not os.path.exists(checkpoint_path_local):
                logger_val.error(f'No checkpoint found for evaluation at {checkpoint_path_local}')
                return None

            if args.eval_save_dir:
                eval_subdir = os.path.basename(os.path.normpath(args.eval_save_dir))
                save_dir = os.path.join(exp_dir, eval_subdir)
            else:
                save_dir = os.path.join(exp_dir, 'eval_results')
            os.makedirs(save_dir, exist_ok=True)
            # setup logger for eval
            setup_logger(f'test_{l_str}', save_dir, 'test_log', level=logging.INFO, screen=True, tofile=True)
            logger_test = logging.getLogger(f'test_{l_str}')

            # load model
            config_local = config
            net_eval = MLICPlusPlus(config_local)
            net_eval = net_eval.to(device)
            ckpt = torch.load(checkpoint_path_local, map_location=device)
            state = ckpt.get('state_dict', ckpt)
            net_eval.load_state_dict(state)

            # prepare dataloader
            if args.eval_dataset is None:
                logger_test.error('No eval dataset provided. Use --eval-dataset to set test data path.')
                return
            test_dataset = ImageFolder(args.eval_dataset, split='test', transform=transforms.ToTensor())
            # attempt to sort samples by numeric filename if possible
            try:
                test_dataset.samples.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)
            except Exception:
                pass

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                shuffle=False,
                pin_memory=(device == 'cuda')
            )

            # run evaluation using test_model helper
            test_model(test_dataloader, net_eval, logger_test, save_dir, ckpt.get('epoch', 0))

            # Collect summary information (checkpoint info + eval folder). test_model currently logs metrics
            summary = {
                'lambda': l,
                'checkpoint': os.path.abspath(checkpoint_path_local),
                'checkpoint_epoch': int(ckpt.get('epoch', 0)) if ckpt is not None else None,
                'checkpoint_loss': float(ckpt.get('loss')) if ckpt is not None and 'loss' in ckpt else None,
                'eval_save_dir': os.path.abspath(save_dir),
                'eval_metrics': None  # placeholder; test_model logs metrics but does not return structured metrics
            }

            # write summary json into experiment folder
            try:
                summary_path = os.path.join(exp_dir, 'eval_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger_test.info(f'Evaluation summary written to {summary_path}')
            except Exception as e:
                logger_test.warning(f'Could not write evaluation summary: {e}')

            return summary

        # If eval-only, run evaluation and skip training entirely for this lambda
        if args.eval_only:
            logger_val.info(f'Running eval-only for {exp_name} using dataset {args.eval_dataset}')
            summary = run_evaluation(args.eval_checkpoint)
            # close tensorboard writer for this lambda
            try:
                tb_logger.close()
            except Exception:
                pass
            # move to next lambda
            continue

        # reset seeds for reproducibility per experiment
        torch.manual_seed(int(seed))
        _random.seed(int(seed))

        # build model, optimizers and criterion per lambda
        net = MLICPlusPlus(config=config)
        if args.cuda and torch.cuda.device_count() > 1:
            net = CustomDataParallel(net)
        net = net.to(device)

        if args.checkpoint:
            try:
                ckpt = torch.load(args.checkpoint, map_location=device)
                state = ckpt.get('state_dict', ckpt)
                net.load_state_dict(state)
                logger_train.info(f'Loaded checkpoint {args.checkpoint}')
            except Exception as e:
                logger_train.warning(f'Could not load checkpoint {args.checkpoint}: {e}')

        optimizer, aux_optimizer = configure_optimizers(net, args)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)
        criterion = RateDistortionLoss(lmbda=l, metrics=args.metrics)

        start_epoch = 0
        best_loss = 1e10
        current_step = 0

        logger_train.info(args)
        logger_train.info(config)
        logger_train.info(net)
        logger_train.info(optimizer)
        optimizer.param_groups[0]['lr'] = args.learning_rate

        for epoch in range(start_epoch, args.epochs):
            logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            current_step = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                logger_train,
                tb_logger,
                current_step,
            )

            loss = test_one_epoch(epoch, val_dataloader, net, criterion, logger_val, tb_logger)

            lr_scheduler.step()
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            # update and save
            try:
                net.update(force=True)
            except Exception:
                # some models may not implement update
                pass

            if args.save and is_best:
                # Only save the checkpoint when it's the best so far.
                # save_checkpoint will also create a copy named checkpoint_best_loss.pth.tar
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    True,
                    os.path.join(ckpt_dir, "checkpoint_best_loss.pth.tar"),
                )
                logger_val.info('best checkpoint saved.')

        # close tensorboard writer for this lambda
        try:
            tb_logger.close()
        except Exception:
            pass

        # Evaluation helper
        def run_evaluation(checkpoint_path=None):
            # Determine checkpoint
            if checkpoint_path is None:
                # choose best checkpoint if exists
                candidate = os.path.join(ckpt_dir, 'checkpoint_best_loss.pth.tar')
                if os.path.exists(candidate):
                    checkpoint_path_local = candidate
                else:
                    # fallback to last epoch checkpoint
                    checkpoints = sorted(Path(ckpt_dir).glob('checkpoint_*.pth.tar'))
                    checkpoint_path_local = str(checkpoints[-1]) if checkpoints else None
            else:
                checkpoint_path_local = checkpoint_path

            if checkpoint_path_local is None or not os.path.exists(checkpoint_path_local):
                logger_val.error(f'No checkpoint found for evaluation at {checkpoint_path_local}')
                return None

            save_dir = args.eval_save_dir or os.path.join(exp_dir, 'eval_results')
            os.makedirs(save_dir, exist_ok=True)
            # setup logger for eval
            setup_logger(f'test_{l_str}', save_dir, 'test_log', level=logging.INFO, screen=True, tofile=True)
            logger_test = logging.getLogger(f'test_{l_str}')

            # load model
            config_local = config
            net_eval = MLICPlusPlus(config_local)
            net_eval = net_eval.to(device)
            ckpt = torch.load(checkpoint_path_local, map_location=device)
            state = ckpt.get('state_dict', ckpt)
            net_eval.load_state_dict(state)

            # prepare dataloader
            if args.eval_dataset is None:
                logger_test.error('No eval dataset provided. Use --eval-dataset to set test data path.')
                return
            test_dataset = ImageFolder(args.eval_dataset, split='test', transform=transforms.ToTensor())
            # attempt to sort samples by numeric filename if possible
            try:
                test_dataset.samples.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)
            except Exception:
                pass

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                shuffle=False,
                pin_memory=(device == 'cuda')
            )

            # run evaluation using test_model helper
            test_model(test_dataloader, net_eval, logger_test, save_dir, ckpt.get('epoch', 0))

            # Collect summary information (checkpoint info + eval folder). test_model currently logs metrics
            summary = {
                'lambda': l,
                'checkpoint': os.path.abspath(checkpoint_path_local),
                'checkpoint_epoch': int(ckpt.get('epoch', 0)) if ckpt is not None else None,
                'checkpoint_loss': float(ckpt.get('loss')) if ckpt is not None and 'loss' in ckpt else None,
                'eval_save_dir': os.path.abspath(save_dir),
                'eval_metrics': None  # placeholder; test_model logs metrics but does not return structured metrics
            }

            # write summary json into experiment folder
            try:
                summary_path = os.path.join(exp_dir, 'eval_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger_test.info(f'Evaluation summary written to {summary_path}')
            except Exception as e:
                logger_test.warning(f'Could not write evaluation summary: {e}')

            return summary

        # Run evaluation-only mode was handled earlier (we continue'd).
        # If an eval dataset is provided, run evaluation now (after training)
        if args.eval_dataset is not None and not args.eval_only:
            summary = run_evaluation(args.eval_checkpoint)



if __name__ == "__main__":
    main()
