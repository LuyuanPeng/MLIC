import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import logging
from torchvision import transforms
from compressai.datasets import ImageFolder
from config.config import model_config
from models import *
from utils.testing import test_model
from utils.logger import setup_logger
from pathlib import Path
def main():
    experiment = "mlicplus0483mse-coral"
    checkpoint_path = f"./experiments/{experiment}/checkpoints/checkpoint_best_loss.pth.tar"  # Update if needed
    test_dataset_path = "/home/luyuan/Data/image_compression/coral_camera"
    batch_size = 1
    num_workers = 4
    save_dir = f"./experiments/{experiment}/eval_results"
    os.makedirs(save_dir, exist_ok=True)

    # Set up logger
    setup_logger('test', save_dir, 'test_log', level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    # Model and config
    config = model_config()
    net = MLICPlusPlus(config)
    net = net.cuda()
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])

    # Test data loader (no transform)
    test_dataset = ImageFolder(test_dataset_path,split='test',transform=transforms.ToTensor())
    test_dataset.samples.sort(key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else Path(x).stem)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Run evaluation
    test_model(test_dataloader, net, logger_test, save_dir, checkpoint["epoch"])

if __name__ == "__main__":
    main()
