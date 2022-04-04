from utils import dotdict
from utils import seed_everything

# from utils import increment_path
import yaml

# from importlib import import_module
# import os

from experiment import Experiment

import torch


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    # save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(device)
    assert False

    # # -- dataset
    # dataset_module = getattr(
    #     import_module("dataset"), args.dataset
    # )  # default: MaskBaseDataset
    # dataset = dataset_module(
    #     data_dir=data_dir,
    # )
    # num_classes = dataset.num_classes  # 18

    # # -- augmentation
    # transform_module = getattr(
    #     import_module("dataset"), args.augmentation
    # )  # default: BaseAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    # dataset.set_transform(transform)

    # # -- data_loader
    # train_set, val_set = dataset.split_dataset()

    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=True,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     val_set,
    #     batch_size=args.valid_batch_size,
    #     num_workers=multiprocessing.cpu_count() // 2,
    #     shuffle=False,
    #     pin_memory=use_cuda,
    #     drop_last=True,
    # )

    # # -- model
    # model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model_module(num_classes=num_classes).to(device)
    # model = torch.nn.DataParallel(model)

    # # -- loss & metric
    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    # opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    # optimizer = opt_module(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     weight_decay=5e-4,
    # )
    # scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # # -- logging
    # logger = SummaryWriter(log_dir=save_dir)
    # with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
    #     json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # best_val_acc = 0
    # best_val_loss = np.inf
    # for epoch in range(args.epochs):
    #     # train loop
    #     model.train()
    #     loss_value = 0
    #     matches = 0
    #     for idx, train_batch in enumerate(train_loader):
    #         inputs, labels = train_batch
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()

    #         outs = model(inputs)
    #         preds = torch.argmax(outs, dim=-1)
    #         loss = criterion(outs, labels)

    #         loss.backward()
    #         optimizer.step()

    #         loss_value += loss.item()
    #         matches += (preds == labels).sum().item()
    #         if (idx + 1) % args.log_interval == 0:
    #             train_loss = loss_value / args.log_interval
    #             train_acc = matches / args.batch_size / args.log_interval
    #             current_lr = get_lr(optimizer)
    #             print(
    #                 f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
    #                 f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
    #             )
    #             logger.add_scalar(
    #                 "Train/loss", train_loss, epoch * len(train_loader) + idx
    #             )
    #             logger.add_scalar(
    #                 "Train/accuracy", train_acc, epoch * len(train_loader) + idx
    #             )

    #             loss_value = 0
    #             matches = 0

    #     scheduler.step()

    #     # val loop
    #     with torch.no_grad():
    #         print("Calculating validation results...")
    #         model.eval()
    #         val_loss_items = []
    #         val_acc_items = []
    #         figure = None
    #         for val_batch in val_loader:
    #             inputs, labels = val_batch
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             outs = model(inputs)
    #             preds = torch.argmax(outs, dim=-1)

    #             loss_item = criterion(outs, labels).item()
    #             acc_item = (labels == preds).sum().item()
    #             val_loss_items.append(loss_item)
    #             val_acc_items.append(acc_item)

    #             if figure is None:
    #                 inputs_np = (
    #                     torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
    #                 )
    #                 inputs_np = dataset_module.denormalize_image(
    #                     inputs_np, dataset.mean, dataset.std
    #                 )
    #                 figure = grid_image(
    #                     inputs_np,
    #                     labels,
    #                     preds,
    #                     n=16,
    #                     shuffle=args.dataset != "MaskSplitByProfileDataset",
    #                 )

    #         val_loss = np.sum(val_loss_items) / len(val_loader)
    #         val_acc = np.sum(val_acc_items) / len(val_set)
    #         best_val_loss = min(best_val_loss, val_loss)
    #         if val_acc > best_val_acc:
    #             print(
    #                 f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
    #             )
    #             torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
    #             best_val_acc = val_acc
    #         torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
    #         print(
    #             f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
    #             f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
    #         )
    #         logger.add_scalar("Val/loss", val_loss, epoch)
    #         logger.add_scalar("Val/accuracy", val_acc, epoch)
    #         logger.add_figure("results", figure, epoch)
    #         print()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # # Data and model checkpoints directories
    # parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    # parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    # parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    # parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    # parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    # parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    # parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    # parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    # parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # # Container environment
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    # parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = None
    with open("train_config.yaml") as f:
        tmp_args = yaml.load(f, Loader=yaml.FullLoader)
        args = dotdict(tmp_args)

    if args is None:
        print("could not found config.yaml file")
        exit()

    for k, v in tmp_args.items():
        args[k] = v

    print("-" * 20, "arguments list", "-" * 20)
    for k, v in args.items():
        print(k, ":", v)
    print("-" * 50)

    # 모델 args
    # TODO: 모델 args 파싱하여 dotdict형태로 저장
    margs = None
    experiment = Experiment(args, margs)

    # args = parser.parse_args()
    # print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
