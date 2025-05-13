from matplotlib import pyplot as plt
import torch
from model.model import DUDnCNN, DnCNN, UDnCNN
import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod
import numpy as np


class NeuralNetwork(nn.Module, ABC):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class StatsManager:
    def __init__(self):
        self.init()

    def __repr__(self):
        return self.__class__.__name__

    def init(self):
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, x=None, y=None, d=None):
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        return self.running_loss / self.number_update


class Experiment:
    def __init__(
        self,
        net,
        train_set,
        val_set,
        optimizer,
        stats_manager,
        output_dir=None,
        batch_size=16,
        perform_validation_during_training=False,
    ):
        self.train_loader = td.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        self.val_loader = td.DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        self.history = []
        if output_dir is None:
            output_dir = "experiment_{}".format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")
        locs = {k: v for k, v in locals().items() if k != "self"}
        self.__dict__.update(locs)
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                if f.read().strip() != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: checkpoint conflict."
                    )
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        return len(self.history)

    def setting(self):
        return {
            "Net": self.net,
            "TrainSet": self.train_set,
            "ValSet": self.val_set,
            "Optimizer": self.optimizer,
            "StatsManager": self.stats_manager,
            "BatchSize": self.batch_size,
            "PerformValidationDuringTraining": self.perform_validation_during_training,
        }

    def __repr__(self):
        return "\n".join("{}({})".format(k, v) for k, v in self.setting().items())

    def state_dict(self):
        return {
            "Net": self.net.state_dict(),
            "Optimizer": self.optimizer.state_dict(),
            "History": self.history,
        }

    def load_state_dict(self, checkpoint):
        self.net.load_state_dict(checkpoint["Net"])
        self.optimizer.load_state_dict(checkpoint["Optimizer"])
        self.history = checkpoint["History"]
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)

    def save(self):
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, "w") as f:
            print(self, file=f)

    def load(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.net.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, plot=None):
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()
            for x, d in self.train_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)
                self.optimizer.zero_grad()
                y = self.net.forward(x)
                loss = self.net.criterion(y, d)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    self.stats_manager.accumulate(loss.item(), x, y, d)
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
                print(
                    "Epoch {} | Time: {:.2f}s | Training Loss: {:.6f}".format(
                        self.epoch, time.time() - s, self.history[-1]
                    )
                )

            else:
                self.history.append((self.stats_manager.summarize(), self.evaluate()))
                # print("Epoch {} | Time: {:.2f}s | Training Loss: {:.6f} | Evaluation Loss: {:.6f}".format(
                #     self.epoch, time.time() - s, self.history[-1][0], self.history[-1][1]))

                print(
                    "Epoch {} | Time: {:.2f}s | Training Loss: {:.6f} | Evaluation Loss: {:.6f}".format(
                        self.epoch,
                        time.time() - s,
                        self.history[-1][0]["loss"],
                        self.history[-1][1]["loss"],
                    )
                )
            self.save()
            if plot:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self):
        self.stats_manager.init()
        self.net.eval()
        with torch.no_grad():
            for x, d in self.val_loader:
                x, d = x.to(self.net.device), d.to(self.net.device)
                y = self.net.forward(x)
                loss = self.net.criterion(y, d)
                self.stats_manager.accumulate(loss.item(), x, y, d)
        self.net.train()
        return self.stats_manager.summarize()


def imshow(image, ax=plt):
    image = image.to("cpu").numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis("off")
    return h


def plot(exp, fig, axes, noisy, visu_rate=2):
    if exp.epoch % visu_rate != 0:
        return
    with torch.no_grad():
        denoised = exp.net(noisy[None].to(exp.net.device))[0]
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    imshow(noisy, ax=axes[0][0])
    axes[0][0].set_title("Noisy image")

    imshow(denoised, ax=axes[0][1])
    axes[0][1].set_title("Denoised image")

    axes[1][0].plot(
        [exp.history[k][0]["loss"] for k in range(exp.epoch)], label="training loss"
    )
    axes[1][0].set_ylabel("Loss")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].legend()

    axes[1][1].plot(
        [exp.history[k][0]["PSNR"] for k in range(exp.epoch)], label="training psnr"
    )
    axes[1][1].set_ylabel("PSNR")
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].legend()

    plt.tight_layout()
    fig.canvas.draw()


class NNRegressor(NeuralNetwork):

    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()

    def criterion(self, y, d):
        return self.mse(y, d)


class DenoisingStatsManager(StatsManager):

    def __init__(self):
        super(DenoisingStatsManager, self).__init__()

    def init(self):
        super(DenoisingStatsManager, self).init()
        self.running_psnr = 0

    def accumulate(self, loss, x, y, d):
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)
        n = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        self.running_psnr += 10 * torch.log10(4 * n / (torch.norm(y - d) ** 2))

    def summarize(self):
        loss = super(DenoisingStatsManager, self).summarize()
        psnr = self.running_psnr / self.number_update
        return {"loss": loss, "PSNR": psnr.cpu()}


class Args:
    def __init__(self):
        self.output_dir = "checkpoints1/"
        self.num_epochs = 10
        self.D = 6
        self.C = 64
        self.plot = True
        self.model = "dudncnn"
        self.lr = 1e-3
        self.image_size = (256, 256)
        self.test_image_size = (256, 256)
        self.batch_size = 8
        self.sigma = 30


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    if args.model == "dncnn":
        net = DnCNN(args.D, C=args.C).to(device)
    elif args.model == "udncnn":
        net = UDnCNN(args.D, C=args.C).to(device)
    elif args.model == "dudncnn":
        net = DUDnCNN(args.D, C=args.C).to(device)
    else:
        raise NameError("Please enter: dncnn, udncnn, or dudncnn")

    # optimizer
    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = DenoisingStatsManager()
    train_dataset = None
    val_dataset = None
    # experiment
    exp = Experiment(
        net,
        train_dataset,
        val_dataset,
        adam,
        stats_manager,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        perform_validation_during_training=True,
    )

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(
            num_epochs=args.num_epochs,
            plot=lambda exp: plot(exp, fig=fig, axes=axes, noisy=val_dataset[73][0]),
        )
    else:
        exp.run(num_epochs=args.num_epochs)
