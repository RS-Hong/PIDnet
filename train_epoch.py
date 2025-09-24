import os
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import SegmentationDataset, seg_dataset_collate
from utils.utils_fit_pidnet import fit_one_epoch
from nets.pidnet import PIDNet
from nets.pidnet_training import PIDNetLoss, get_lr_scheduler, set_optimizer_lr
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.callbacks import LossHistory, EvalCallback

def main():
    Cuda = True
    distributed = False
    fp16 = False
    num_classes = 2
    input_shape = [512, 512]
    pretrained = True
    model_path = ""
    dataset_path = "VOCdevkit"
    batch_size = 8
    Init_lr = 3e-4  # 降低学习率以提高稳定性
    Min_lr = Init_lr * 0.01
    num_epochs = 200  # 基于 epoch 控制训练
    dice_loss = False
    weight_decay = 0.01
    lr_decay_type = "poly"
    save_period = 5  # 每 5 epoch 保存一次
    save_dir = "logs"
    tensorboard_dir = "logs"

    # 检查 logs 目录权限
    try:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        with open(os.path.join(tensorboard_dir, "test_write.txt"), 'w') as f:
            f.write("Test write permission")
        os.remove(os.path.join(tensorboard_dir, "test_write.txt"))
        print(f"Logs directory {save_dir} is writable")
    except Exception as e:
        print(f"Error accessing logs directory: {e}")
        return

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if Cuda else "cpu")
        local_rank = 0

    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    model = PIDNet(num_classes=num_classes, aux=True).to(device)
    loss_func = PIDNetLoss(ignore_index=255).to(device)

    if pretrained:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            load_key = len(pretrained_dict)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Load {load_key} keys from {model_path}")
        else:
            print(f"Pretrained model {model_path} not found")

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    train_dataset = SegmentationDataset(train_lines, input_shape, num_classes, True, dataset_path)
    val_dataset = SegmentationDataset(val_lines, input_shape, num_classes, False, dataset_path)

    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(not distributed),
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_dataset_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=seg_dataset_collate
    )

    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=Init_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    total_iters = num_epochs * len(train_loader)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, total_iters)

    loss_history = LossHistory(tensorboard_dir, model, input_shape=input_shape)
    eval_callback = EvalCallback(
        model, input_shape, num_classes, val_lines, dataset_path, tensorboard_dir, Cuda,
        miou_out_path=".temp_miou_out", eval_flag=True, period=1
    )

    best_mIoU = 0.0
    global_iter = 0

    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        print(f"Start Train")
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="it/s") as pbar:
            total_loss = 0
            for iteration, batch in enumerate(train_loader):
                images, targets, edge_targets = batch
                images = images.to(device)
                targets = targets.to(device)
                edge_targets = edge_targets.to(device)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=fp16):
                    outputs = model(images)
                    loss = loss_func(outputs, targets, edge_targets)

                if fp16:
                    from torch.cuda.amp import GradScaler
                    scaler = GradScaler()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                set_optimizer_lr(optimizer, lr_scheduler_func, global_iter, total_iters)
                total_loss += loss.item()

                global_iter += 1
                pbar.set_postfix(**{"total_loss": total_loss / (iteration + 1), "lr": optimizer.param_groups[0]["lr"]})
                pbar.update(1)

        print(f"Finish Train\nStart Validation")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="it/s") as val_pbar:
                for val_batch in val_loader:
                    val_images, val_targets, val_edge_targets = val_batch
                    val_images = val_images.to(device)
                    val_targets = val_targets.to(device)
                    val_edge_targets = val_edge_targets.to(device)
                    val_outputs = model(val_images)
                    val_loss += loss_func(val_outputs, val_targets, val_edge_targets).item()
                    val_pbar.update(1)
        val_loss /= len(val_loader)

        eval_callback.on_epoch_end(epoch, model_eval=model_without_ddp)

        with open(os.path.join(tensorboard_dir, "epoch_miou.txt"), 'r') as f:
            miou_lines = f.readlines()
            val_mIoU = float(miou_lines[-1].strip()) / 100.0 if miou_lines else 0.0

        if local_rank == 0:
            loss_history.append_loss(epoch, total_loss / len(train_loader), val_loss)
            print(f"Finish Validation\nEpoch:{epoch + 1}/{num_epochs}\nTotal Loss: {total_loss / len(train_loader):.3f} || Val Loss: {val_loss:.3f} || Val mIoU: {val_mIoU:.3f}")
            if val_mIoU > best_mIoU:
                best_mIoU = val_mIoU
                torch.save(model_without_ddp.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
                print("Save best model to best_epoch_weights.pth")
            if (epoch + 1) % save_period == 0:
                torch.save(model_without_ddp.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}_weights.pth"))

    if local_rank == 0:
        loss_history.writer.close()
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()