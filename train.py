#  Author: Paul-Jason Mello
#  Date: June 5th, 2023


#  General Libraries
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

#  Torch Libraries
import torch
import torch.nn.functional as func
import torch.utils.data
import torchvision
from torch import optim, nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

#  Misc. Libraries
import os
import math
import time
import datetime
import matplotlib.style as mplstyle

#  Model Libraries
from unet import unet

#  Image Libraries
from image_transform import RGBTransform, GrayscaleTransform

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(torch.cuda.current_device())

#  Set Seed for Reproducibility
#  seed = 3407  # https://arxiv.org/abs/2109.08203
seed = np.random.randint(0, 1_000_000)  # random seed
np.random.seed(seed)
torch.manual_seed(seed)

matplotlib.use("Agg")
mplstyle.use(["dark_background", "fast"])

"""
Algorithm 1 Training                                                            
1:  repeat  
2:      x_0 ∼ q(x_0)      
3:      t ∼ Uniform({1, . . . , T })    
4:       ∼ N (0, I)      
5:      Take gradient descent step on                                          
            ∇θ ||  − _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t) || ^ 2        
6: until converged                                                             
"""


def main():
    # Default Parameters: lr: 2e-4, epochs: 10, batch_size: 128, dataset: "mnist", schedule: "linear",
    # lossMetric: "MSE", prediction_Objective: "noise", weightedSNR: True, steps: 1000, t_steps: 1000

    lr = 2e-4
    epochs = 25
    batch_size = 512

    # Batch Size | Loss   | Time per Epoch | (@ 5 Epochs, "linear")
           # 256 | 0.0353 | 1:02
           # 128 | 0.0305 | 0:52
           # 64  | 0.0278 | 1:02

    # TODO fix "Not Working" datasets
    dataset = "fashion-mnist"  # "mnist", "fashion-mnist", "cifar10", "celeba"

    # Noise Scheduling should be automated based on image size. See https://arxiv.org/abs/2301.10972 for more details.
    # https: // arxiv.org / abs / 2102.09672, https: // arxiv.org / abs / 2212.11972
    # "auto" is the default setting, which is based on the image size. "linear" is the default setting for image sizes 32x32 and below.
    # TODO: fix "Not Working" noise schedules              (SNR doesnt generate)
    schedule = "auto"  # "linear", "cosine" "sigmoid", "auto"   | No Working: "snrCosine", "SNR"     |    sigmoid > cosine > linear > snrCosine

    # TODO: fix "Not Working" loss metrics
    loss_metric = "MSE"  # Working: "MSE", "L1"  | Not Working: "KL", "PSNR", "ELBO", "SCORE"  | "MSE" > "L1"

    prediction_objective = "noise"  # Working: "noise", "recon"   | Not Working: "X0" (DDIM)

    weighted_snr = False
    pretrained_model = False

    steps = 1000
    tSteps = 1000

    epsilon = train(loss_metric, dataset, steps, tSteps, epochs, batch_size, lr, schedule, prediction_objective, weighted_snr, pretrained_model)
    epsilon.run()

def printSystemDynamics():
    print("Cuda: ", torch.cuda.is_available())
    print("Device: ", device)
    print("Device Count: ", torch.cuda.device_count())
    print("Cuda Version: ", torch.version.cuda)


def printModelInfo(model):
    print("\nModel Info:")
    print("\tModel: ", model)
    print("\tModel Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


def ELBO(target, epsilon):
    return torch.mean(target) - torch.log(torch.mean(torch.exp(target - epsilon)))


def getKLDivergence(p, q):
    p = p / p.sum(dim=(1, 2), keepdim=True)  # Normalize the input probability distribution
    q = q / q.sum(dim=(1, 2), keepdim=True)  # Normalize the target probability distribution
    kl = (p * (p / q).log()).sum(dim=(1, 2)).mean()
    return kl


def getExtract(tensor: torch.Tensor, t: torch.Tensor, X):  # Extracts the correct value from a tensor
    out = tensor.gather(-1, t.cpu()).float()
    return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device)


class train:
    def __init__(self, loss_metric, dataset, steps, tSteps, epochs, batch_size, lr, schedule, prediction_objective, weighted_snr, pretrained_model):
        super().__init__()

        self.lr = lr
        self.steps = steps
        self.epochs = epochs
        self.tSteps = tSteps
        self.loadData = dataset
        self.scheduler = schedule
        self.batchSize = batch_size
        self.snrWeight = weighted_snr
        self.lossMetric = loss_metric
        self.pretrainedModel = pretrained_model  # Use Pretrained Model
        self.predictionObjective = prediction_objective

        self.loss = 0.0
        self.epochCounter = 0
        self.lossList = []

        if dataset == "mnist" or dataset == "fashion-mnist":
            self.numChannels = 1
            self.imageSize = 28
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "cifar10":
            self.numChannels = 3
            self.imageSize = 32
            if schedule == "auto":
                self.scheduler = "linear"
        elif dataset == "celeba":
            self.numChannels = 3
            self.imageSize = 64
            if schedule == "auto":
                self.scheduler = "sigmoid"

        self.Beta = self.getSchedule(self.scheduler)  # Noise Schedule
        Alpha = 1.0 - self.Beta  # Alpha Schedule

        self.Alpha_Bar = torch.cumprod(Alpha, dim=0)  # Product Value of Alpha
        self.Sqrt_Alpha_Cumprod = torch.sqrt(self.Alpha_Bar)  # Square Root of Product Value of Alpha
        self.Sqrt_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Bar)  # Square Root of 1 - Product Value of Alpha
        self.Sqrt_Recipricol_Alpha_Cumprod = torch.sqrt(1.0 / self.Alpha_Bar)  # Square Root of Reciprocal of Product Value of Alpha

        # https://arxiv.org/abs/2303.09556
        snr = self.Alpha_Bar / (1 - self.Alpha_Bar)
        self.snrClip = snr.clone()
        if self.snrWeight:
            self.snrClip.clamp_(max=5)

    def getSchedule(self, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 2e-2, self.steps)
        elif schedule == "cosine":
            return self.getCosineSchedule()
        elif schedule == "sigmoid":
            return self.getSigmoidSchedule(-3, 3)
        elif schedule == "SNR":
            return self.getSNRSchedule()
        elif schedule == "snrCosine":
            return self.getLogSNRCosineSchedule(-10, 10)

    def getCosineSchedule(self):
        x = torch.linspace(0, self.tSteps, self.steps + 1)
        y = torch.cos(((x / self.tSteps) + 0.01) / (1 + 0.01) * torch.pi * 0.5) ** 2
        z = y / y[0]
        return torch.clip((1 - (z[1:] / z[:-1])), 0.0001, 0.9999)

    def getSigmoidSchedule(self, start, end):
        sequence = torch.linspace(0, self.tSteps, self.steps + 1, dtype=torch.float64) / self.tSteps
        v_start = torch.tensor(start / 1).sigmoid()
        v_end = torch.tensor(end / 1).sigmoid()
        alpha = (-((sequence * (end - start) + start) / 1).sigmoid() + v_end) / (v_end - v_start)
        alpha = alpha / alpha[0]
        betas = 1 - (alpha[1:] / alpha[:-1])
        return torch.clip(betas, 0, 0.999)

    def getLogSNRCosineSchedule(self, SNR_min, SNR_max):
        snr_values = torch.linspace(0, self.tSteps, self.steps + 1, dtype=torch.float64) / self.tSteps
        min_val = math.atan(math.exp(-0.5 * SNR_max))
        max_val = math.atan(math.exp(-0.5 * SNR_min))
        return torch.log(torch.tan(min_val + snr_values * (max_val - min_val)).clamp(min=1e-20))

    def getSNRSchedule(self):
        snr_values = torch.linspace(0, self.tSteps, self.steps + 1, dtype=torch.float64) / self.tSteps
        noise_std = torch.sqrt(torch.tensor(1) / snr_values - torch.tensor(1))
        betas = 1 - (noise_std[1:] / noise_std[:-1]) ** 2
        return torch.clip(betas, 0, 0.999)

    def getPSNR(self, input, target):
        if self.numChannels == 3:
            pixelMax = 255.0
        else:
            pixelMax = 1.0
        mse = torch.mean((input - target) ** 2)
        return torch.log10(pixelMax / torch.sqrt(mse))

    def getModel(self):
        if self.pretrainedModel:
            if os.path.isfile(f"./data/ddpm model/{self.loadData} ddpm E100.pt"):
                model = torch.load(f"./data/ddpm model/{self.loadData} ddpm E100.pt")
            else:
                raise Exception("Pretrained Model Not Found")
        else:
            # 128, 2, (4, 8, 16), (1,2,4), 4
            model = unet(
                in_channels=self.numChannels,  # 1 grey scale image, 3 for RGB      |
                model_channels=32,  # Number of Channels in the Model               |   Tradeoff: Large Complexity vs. Lower Performance    |  {96, 128, 192, 256, ...}
                out_channels=self.numChannels,  # 1 grey scale image, 3 for RGB     |
                num_res_blocks=2,  # Number of Residual Blocks                      |   Tradeoff: Large Complexity vs. Lower Performance    |  {2, 3, 4, ...}
                attention_resolutions=(8, 16),  # Attention Resolutions             |   Capture spatial information                         |  (tuple) {8, 16, 32, 64, ...}
                dropout=0,  # Dropout Rate                                          |
                channel_mult=(1, 2, 4),  # Channel Multiplier                       |   Number of Channels Multiplied at each Layer         |  (tuple) {1, 2, 3, 4, ...}
                conv_resample=True,  # Convolutional Resampling                     |   Inc/Dec Spatial Resolution using Convolution        |  (bool) {True, False}
                num_heads=2  # Number of Attention Heads                            |   Number of Attention Heads in Multi-Head Attention   |  {4, 8, 16, 32, ...}
            ).to(device)

        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, eps=1e-8)
        return model, optimizer

    def getDataset(self):
        train_data = None  # Initialize Datasets
        grey = GrayscaleTransform(self.imageSize, self.numChannels)
        rgb = RGBTransform(self.imageSize)

        # testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        if self.loadData == "mnist":
            train_data = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=grey)
        elif self.loadData == "fashion-mnist":
            train_data = torchvision.datasets.FashionMNIST(root="./data", download=True, train=True, transform=grey)
        elif self.loadData == "cifar10":
            train_data = torchvision.datasets.CIFAR10(root="./data", download=True, train=True, transform=rgb)
        elif self.loadData == "celeba":
            train_data = torchvision.datasets.CelebA(root="./data", download=True, transform=rgb)

        dataset = DataLoader(train_data, batch_size=self.batchSize, shuffle=True, pin_memory=True, num_workers=4)

        if self.loadData == "celeba":
            Label = train_data.identity
        else:
            Label = train_data.targets

        return dataset, Label  # testData

    def plotGraphHelper(self, xlabel, ylabel, title, savePath):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(savePath)
        plt.close()

    def plotGraph(self, arr, label, xlabel, ylabel, title, savePath):
        plt.figure(figsize=(10, 10))
        plt.plot(arr, label=label, markersize=2)
        plt.legend()
        self.plotGraphHelper(xlabel, ylabel, title, savePath)

    def printTrainingInfo(self):
        print("\nHyperparameters:")
        print("\tData: ", self.loadData)
        print("\tEpochs: ", self.epochs)
        print("\tSteps: ", self.steps)
        print("\tBatch Size: ", self.batchSize)
        print("\tLearning Rate: ", self.lr)
        print("\tScheduler: ", self.scheduler)
        print("\tLoss Metric: ", self.lossMetric)
        print("\tPrediction Objective: ", self.predictionObjective)

    def qSample(self, X0, t):  # Sample from q(Xt | X0) = N(x_t; sqrt(alpha_bar_t) * x_0, sqrt(1 - alpha_bar_t) * noise, t
        noise = torch.randn_like(X0)  # Sample from N(0, I)
        QSample = getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * X0 + getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * noise  # Sample from q(Xt | X0)
        return QSample, noise

    def predV(self, X0, t, noise):
        return getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * noise - getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * X0

    def gradientDescent(self, model, X0):  # TRAINING 1: Data = 2: X_0 ∼ q(x_0)
        target = snr = None
        loss = 0

        t = torch.randint(0, self.steps, (X0.shape[0],), device=device).long()  # TRAINING 3: t ∼ Uniform({1, . . . , T })
        XT, epsilon = self.qSample(X0, t)  # TRAINING 2: X_t ∼ q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, sqrt(1 - alpha_bar_t) * noise, t)

        if self.predictionObjective == "noise":
            snr = self.snrClip / self.Alpha_Bar / (1 - self.Alpha_Bar)
            target = epsilon
        elif self.predictionObjective == "recon":
            snr = self.snrClip / (self.Alpha_Bar / (1 - self.Alpha_Bar) + 1)
            target = self.predV(X0, t, epsilon)
        elif self.predictionObjective == "X0":
            snr = self.snrClip
            target = X0

        # TRAINING 5: ∇θ ||  − (TRAINING 3) || ^ 2
        with autocast():
            epsilon_theta = model(XT, t)  # TRAINING 3: UNet(XT, t) = _θ * (√ (̄α_t) * x_0 + √(1−α_t) * , t)
            if self.lossMetric == "MSE":
                loss = func.mse_loss(epsilon_theta, target)
            elif self.lossMetric == "L1":
                loss = func.l1_loss(epsilon_theta, target)

        if self.snrWeight:
            loss = torch.mean(loss * getExtract(snr, t, loss.shape))  # Reweighting Loss by SNR

        # Block has worked in the past, must update for new scaler code, including grad_fn and gradients
        """elif self.lossMetric == "KL":  # Store Bought KL Divergence
            loss = func.kl_div(target, epsilon, reduction="batchmean")
        elif self.lossMetric == "KL":  # Custom KL Divergence
            outputs = torch.softmax(target, dim=-1)
            target = torch.ones_like(outputs) / outputs.size(1)
            loss = self.getKLDivergence(outputs, target)  # 4: L(θ) = E_{x_0, x_t, _θ} [KL(_θ, )]  |  TRAINING 2: L(θ) = E_{x_0, x_t, _θ} [KL(_θ, )]
        elif self.lossMetric == "ELBO":
            loss = self.ELBO(target, epsilon)
        elif self.lossMetric == "SCORE":
            mu = 0.1
            loss = self.score_matching_loss(model, XT, t, epsilon_theta, epsilon, mu)
        elif self.lossMetric == "PSNR":
            loss = self.getPSNR(target, epsilon)  # 4: L(θ) = E_{x_0, x_t, _θ} [PSNR(_θ, )]  |  TRAINING 2: L(θ) = E_{x_0, x_t, _θ} [PSNR(_θ, )]"""

        return loss

    def trainingLoop(self, dataset, model, scaler, optimizer):

        for idx, (X0, labels) in enumerate(dataset):  # TRAINING 1: Repeated Loop
            X0, labels = X0.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = self.gradientDescent(model, X0.to(device))  # TRAINING 4: θ ← θ − α ∇θL(θ)
            self.loss = loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if int(idx % 10) == 0:
                print(f"\t  T: {idx:05d}/{len(dataset)} | Loss: {round(self.loss.item(), 11)}")

        self.epochCounter += 1

    def run(self):

        # Print System Dynamics
        printSystemDynamics()

        # Get Data
        print("\n\t\t...Loading Data...")

        dataset, label = self.getDataset()

        # Get Models
        print("\n\t\t...Loading Models...")
        model, optimizer = self.getModel()

        # Waiting for Triton to support Windows
        # model = torch.compile(model)

        # Print Model Information
        # self.PrintModelInfo(model)

        # Print Training Information
        self.printTrainingInfo()

        startTime = time.time()
        print("\nTraining Start Time: " + str(datetime.datetime.now()))

        scaler = GradScaler()

        while self.epochCounter != self.epochs:
            print(f"\n     -------------- Epoch {self.epochCounter} -------------- ")
            self.trainingLoop(dataset, model, scaler, optimizer)  # Sampling done intermittently during training
            self.lossList.append(format(self.loss.item()) if self.lossMetric == "KL" else self.loss.item())

        print(f"\n     -------------- Epoch {self.epochCounter} -------------- ")
        print(f"\t  Final   Model Loss: \t{round(self.loss.item(), 10)}")
        print(f"\t  Average Model Loss: \t{round(sum(self.lossList) / len(self.lossList), 10)}")  # Average of last 5 losses
        print(f"\t  Minimum Model Loss: \t{round(min(self.lossList), 10)}")

        torch.save(model, f"data/ddpm model/{self.loadData} ddpm E{self.epochs}.pt") # Save Model

        endTime = time.time()
        print("\nTraining Completion Time: " + str(datetime.datetime.now()))
        print(f"Total Training Time: {(endTime - startTime) / 60:.2f} mins")

        self.plotGraph(self.lossList, "Loss", "Epoch", "Loss", "Training Loss", f"Images/{self.loadData}/graphs/E{self.epochs} Training Loss.jpg")

if __name__ == "__main__":
    main()
