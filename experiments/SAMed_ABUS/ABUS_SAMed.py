#Add repo path to the system path
from pathlib import Path
import os, sys
repo_path= Path.cwd().resolve()
while '.gitignore' not in os.listdir(repo_path): # while not in the root of the repo
    repo_path = repo_path.parent #go up one level
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Libraries
from sklearn.model_selection import KFold
import SimpleITK
import numpy as np
import argparse
import yaml

# accelerate
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import logging
from tqdm import tqdm
from importlib import import_module
import wandb
from sklearn.metrics import jaccard_score
# from scipy.ndimage import zoom
# from einops import repeat
# from icecream import ic
# from PIL import Image

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    EnsureTyped,
    EnsureChannelFirstd,
    Resized,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAffined,
    OneOf,
)

# extra imports
sys.path.append(str(repo_path / 'SAMed'))
from SAMed.segment_anything import sam_model_registry
from SAMed.utils import DiceLoss #, Focal_loss

# from SAMed.segment_anything import build_sam, SamPredictor
# from SAMed.segment_anything.modeling import Sam

class ABUS_dataset(Dataset):
    
    def __init__(self, list_dir, transform=None):
        self.transform = transform  # using transform in torch!
        images = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i in list_dir[0]]
        labels = [SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(str(i))) for i in list_dir[1]]

        self.sample_list = list(zip(images,labels))
        
        self.resize=Compose([Resized(keys=["label"], spatial_size=(64, 64),mode=['nearest'])])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        
        if self.transform:
            sample=self.transform({"image": self.sample_list[idx][0], "label": self.sample_list[idx][1]})
        
        sample['low_res_label']=self.resize({"label":sample['label']})['label'][0]
        sample['label']=sample['label'][0]
        return sample
    
def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    """Compute the loss of the network using a linear combination of cross entropy and dice loss.

    Args:
        outputs (Torch.Tensor): output of the network
        low_res_label_batch (Torch.Tensor): low resolution version of the label (mask)
        ce_loss (functional): CrossEntropyLoss
        dice_loss (functional): Dice loss from SAMed
        dice_weight (float, optional): parametrization the linear combination (high=more importance to dice). Defaults to 0.8.

    Returns:
        float: loss, loss_ce, loss_dice floats
    """
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

logger = get_logger(__name__)

def main(fold_n:int, train_ids:list, val_ids:list):
    """main function needs split fold number and train and val ids

    Args:
        fold_n (int): fold number
        train_ids (list): list with train ids, ints
        val_ids (list): list with val ids, ints
    """

    # read and set config file
    config_path = 'config_file.yaml' # configuration file path (beter to call it from the args parser)
    with open(config_path) as file: # expects the config file to be in the same directory
        config = yaml.load(file, Loader=yaml.FullLoader)
    args = argparse.Namespace(**config) # parse the config fil
    run_name = args.run_name   

    # paths
    experiment_path = Path.cwd().resolve() # where the script is running
    data_path = repo_path / args.data_path # path to data
    checkpoint_dir = repo_path / 'checkpoints' # SAMed checkpoints
    logging_dir = experiment_path / f'results/{run_name}/fold{fold_n}/logs' # path for logging
    lora_weights = experiment_path / f'results/{run_name}/fold{fold_n}/weights' # the lora parameters folder is created to save the weights
    os.makedirs(lora_weights,exist_ok=True)

    # accelerator
    accelerator_project_config = ProjectConfiguration()
    accelerator = Accelerator( # start accelerator
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to, # logger (wandb or tensorboard)
        project_dir=logging_dir, # defined above
        project_config=accelerator_project_config, # project config defined above
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    # training seed
    if args.training_seed is not None:
        set_seed(args.training_seed) # accelerate seed
        logger.info(f"Set seed {args.training_seed} for training.")
    # Transforms
    deform = Rand2DElasticd(
        keys=["image", "label"],
        prob=0.5,
        spacing=(7, 7),
        magnitude_range=(1, 2),
        rotate_range=(np.pi / 6,),
        scale_range=(0.2, 0.2),
        translate_range=(20, 20),
        padding_mode="zeros",
        mode=['bilinear','nearest']
    )

    affine = RandAffined(
        keys=["image", "label"],
        prob=0.5,
        rotate_range=(np.pi / 6),
        scale_range=(0.2, 0.2),
        translate_range=(20, 20),
        padding_mode="zeros",
        mode=['bilinear','nearest']

    )

    train_transform = Compose(
        [
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'), # to add one channel dim to the label (1,256,256)

            ScaleIntensityd(keys=["image"]), # to scale the image intensity to [0,1]

            ## train-specific transforms
            RandRotated(keys=["image", "label"], range_x=(-np.pi / 12, np.pi / 12), prob=0.5, keep_size=True,mode=['bilinear','nearest']),

            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),

            RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5,mode=['area','nearest']),

            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
            RandGaussianNoised(keys=["image"], mean=0, std=0.1, prob=0.5),

            # TODO check this
            OneOf(transforms=[affine, deform], weights=[0.8, 0.2]), # apply one of the two transforms with the given weights
            ##

            Resized(keys=["image", "label"], spatial_size=(256, 256),mode=['area','nearest']),

            EnsureTyped(keys=["image"] ), # ensure it is a torch tensor or np array
        ]
    )
    train_transform.set_random_state(seed=args.training_seed) # set the seed for the transforms

    val_transform = Compose(
        [
            EnsureChannelFirstd(keys=['label'], channel_dim='no_channel'),

            ScaleIntensityd(keys=["image"]),

            Resized(keys=["image", "label"], spatial_size=(256, 256),mode=['area','nearest']),
            EnsureTyped(keys=["image"])
        ])

    # dataset
    path_images = (data_path / "image_mha")
    path_labels = (data_path / "label_mha")
    # get all files in the folder in a list, only mha files
    image_files = sorted([file for file in os.listdir(path_images) if file.endswith('.mha')])
    # now, we will check if the path has at least one of the ids in the train_ids list
    train_files = [file for file in image_files if any(f'id_{id}_' in file for id in train_ids)]
    val_files = [file for file in image_files if any(f'id_{id}_' in file for id in val_ids)]

    image_files = np.array([path_images / i for i in train_files])
    label_files = np.array([path_labels / i for i in train_files])
    list_train = [image_files, label_files] # this is what we will pass to the dataset <-

    image_files = np.array([path_images / i for i in val_files])
    label_files = np.array([path_labels / i for i in val_files])
    list_val = [image_files, label_files] # this is what we will pass to the dataset <-

    # define datasets, notice that the inder depends on the fold
    db_train = ABUS_dataset(transform=train_transform,list_dir=list_train)
    db_val = ABUS_dataset(transform=val_transform,list_dir=list_val)   

    # define dataloaders
    trainloader = DataLoader(db_train, batch_size=args.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=args.val_batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # get SAM model
    sam, _ = sam_model_registry['vit_b'](image_size=256,
                                        num_classes=args.num_classes,
                                        checkpoint=str(checkpoint_dir / 'sam_vit_b_01ec64.pth'),
                                        pixel_mean=[0, 0, 0],
                                        pixel_std=[1, 1, 1])
    # load lora model
    pkg = import_module('sam_lora_image_encoder')
    net = pkg.LoRA_Sam(sam, 4) # lora rank is 4
    if args.pretrained_path: # load pretrained weights
        net.load_lora_parameters(str(repo_path / args.pretrained_path))
    model=net
    model.train()
    
    # metrics
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes + 1)
    # max iterations
    max_iterations = args.max_epoch * len(trainloader)  

    # optimizer
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
    else:
        b_lr = args.base_lr
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',verbose=True)

    # prepare with accelerator
    model, optimizer, trainloader, valloader, scheduler = accelerator.prepare(
            model, optimizer, trainloader, valloader, scheduler
        )
    
    # Initialize trackers
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, config=vars(args))
        wandb.save(str(config_path)) if args.report_to=="wandb" else None

    # logging
    logger.info("***** Running training *****")
    logger.info(f"Max number of iterations {max_iterations}")
    logger.info(f"The length of train set is: {len(db_train)}")
    logger.info(f"The length of val set is: {len(db_val)}")
    logger.info(f"Num batches each epoch = {len(trainloader)}")
    logger.info(f'Number of batches for validation: {len(valloader)}')
    logger.info(f"Num Epochs = {args.max_epoch}")
    logger.info(f"Instantaneous batch size per device = {args.train_batch_size}")

    # init useful variables
    iterator = tqdm(range(args.max_epoch), desc="Training", unit="epoch")
    iter_num = 0
    best_performance = 100100

    for epoch_num in iterator:
        # lists
        train_loss_ce = []
        train_loss_dice = []
        val_loss_ce = []
        val_loss_dice = []
        
        # training time
        for sampled_batch in trainloader:

            with accelerator.accumulate(model): # forward and loss computing
            
                # load batches
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w] # label used only for showing in the log
                low_res_label_batch = sampled_batch['low_res_label'] # for logging

                assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}' #check the intensity range of the image
                
                # forward and loss computing
                outputs = model(batched_input = image_batch, multimask_output = args.multimask_output, image_size = 256)
                loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, 0.8)
                accelerator.backward(loss)
                optimizer.step()
                if args.warmup and iter_num < args.warmup_period: # if in warmup period, adjust learning rate
                    lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                else: # if not in warmup period, adjust learning rate
                    if args.warmup:
                        shift_iter = iter_num - args.warmup_period
                        assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    else: # if not warm up at all, leave the shift as the iter number
                        shift_iter = iter_num
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

                optimizer.zero_grad()

            if accelerator.sync_gradients: # update iter number
                # progress_bar.update(1)
                # update the iter number
                iter_num += 1 

            # logging
            logs = {"loss": loss.detach().item(), "loss_ce": loss_ce.detach().item(), "loss_dice": loss_dice.detach().item(), "lr": lr_} # or lr_
            iterator.set_postfix(**logs)
            accelerator.log(values=logs, step=iter_num)
            # append lists
            train_loss_ce.append(loss_ce.detach().cpu().numpy())
            train_loss_dice.append(loss_dice.detach().cpu().numpy())
            # show to user
            # logging.info(f'iteration {iter_num} : loss : {loss.item()}, loss_ce: {loss_ce.item()}, loss_dice: {loss_dice.item()} ,lr:{optimizer.param_groups[0]["lr"]}')
            
            if iter_num % 20 == 0: # log training examples every 20 iterations
                # image
                image = image_batch[1, 0:1, :, :].cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                # prediction
                output_masks = outputs['masks'].detach().cpu()
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True)
                # ground truth
                labs = label_batch[1, ...].unsqueeze(0) * 50
                labs = labs.cpu().numpy()
                
                # logging images    
                accelerator.log(
                    {
                        "training_example": [
                            wandb.Image(image, caption="image"),
                            wandb.Image(output_masks[1, ...] * 50, caption="prediction"),
                            wandb.Image(labs, caption="ground truth"),
                        ]
                    },
                    step=iter_num,
                )

        # train logging after each epoch
        train_loss_ce_mean = np.mean(train_loss_ce)
        train_loss_dice_mean = np.mean(train_loss_dice)
        logs_epoch = {"train_total_loss": train_loss_ce_mean+train_loss_dice_mean, "train_loss_ce": train_loss_ce_mean, "train_loss_dice": train_loss_dice_mean}
        accelerator.log(values=logs_epoch, step=iter_num)
        
        # validation time
        model.eval()
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            low_res_label_batch = sampled_batch['low_res_label']
            
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            
            # forward and losses computing
            outputs = model(image_batch, args.multimask_output, 256)
            loss, loss_ce, loss_dice = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, 0.8)
            # append lists
            val_loss_ce.append(loss_ce.detach().cpu().numpy())
            val_loss_dice.append(loss_dice.detach().cpu().numpy())
            
            if i_batch % (len(valloader)-1) == 0:
                # image
                image = image_batch[1, 0:1, :, :].cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                # prediction
                output_masks = outputs['masks'].detach().cpu()
                output_masks = torch.argmax(torch.softmax(output_masks, dim=1), dim=1, keepdim=True) # we take the max prob per pixel
                # ground truth
                labs = label_batch[1, ...].unsqueeze(0) * 50
                labs = labs.cpu().numpy()
                # logging
                accelerator.log(
                    {
                        "validation_example": [
                            wandb.Image(image, caption="image"),
                            wandb.Image(output_masks[1, ...] * 50, caption="prediction"),
                            wandb.Image(labs, caption="ground truth"),
                        ]
                    },
                    step=iter_num,
                )
        # validation logging after each epoch
        val_loss_ce_mean = np.mean(val_loss_ce)
        val_loss_dice_mean = np.mean(val_loss_dice)

        logs_epoch = {'val_total_loss': val_loss_ce_mean+val_loss_dice_mean, 'val_loss_ce': val_loss_ce_mean, 'val_loss_dice': val_loss_dice_mean}
        accelerator.log(logs_epoch, step=iter_num)

        # update learning rate
        scheduler.step(val_loss_ce_mean+val_loss_dice_mean)
        
        # saving model
        if val_loss_dice_mean < best_performance: # if val dice is better, save model
            best_performance=val_loss_dice_mean
            save_mode_path = os.path.join(lora_weights, f'epoch_{str(epoch_num)}.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logger.info(f"\nSaving model to {save_mode_path}")

        if epoch_num >= args.max_epoch - 1: # save model at the last epoch
            save_mode_path = os.path.join(lora_weights, f'epoch_{str(epoch_num)}.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logger.info(f"\nSave last epoch model to {save_mode_path}")
            iterator.close()
        model.train()
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == '__main__':
    # read and set config file
    config_path = 'config_file.yaml' # configuration file path (beter to call it from the args parser)
    with open(config_path) as file: # expects the config file to be in the same directory
        config = yaml.load(file, Loader=yaml.FullLoader)
    args = argparse.Namespace(**config) # parse the config fil
    
    ### make the split and get files list. We make a split of our 100 patient ids
    kf = KFold(n_splits=args.num_folds,shuffle=args.split_shuffle,random_state=args.split_seed)
    for fold_n, (train_ids, val_ids) in enumerate(kf.split(range(100))):
        main(fold_n, train_ids, val_ids) 
        break # only one fold for now