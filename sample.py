#  Author: Paul-Jason Mello
#  Date: June 5th, 2023


#  General Libraries
import numpy as np
import matplotlib
import matplotlib.style as mplstyle
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

#  Torch Libraries
import torch
import torchvision
import torch.utils.data
import torch.nn.functional as func
from torch import optim, nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.cuda.amp import autocast

#  Misc. Libraries
import os
import math
import time
import datetime
from tqdm import tqdm

#  Model Libraries
from unet import unet

#  Image Libraries
from image_transform import RGBTransform, GrayscaleTransform, ConvertToImage

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
Algorithm 2 Sampling                                                            
1: xT ∼ N (0, I)                                                                
2: for t = T, . . . , 1 do                                                      
3:      z ∼ N (0, I) if t > 1, else z = 0                                       
4:      x_t−1 = 1/(√(α_t)) * (x_t − (1−α_t)/√(1−α_t) * _θ(x_t, t)) + σtz      
5: end for
6: return x_0
"""


def main():
    # Default Parameters: lr: 2e-4, epochs: 10, batch_size: 128, dataset: "mnist", schedule: "linear", lossMetric: "MSE",
    # prediction_Objective: "noise", weightedSNR: True, sample_count: 10, steps: 1000, t_steps: 1000

    generate_count = 5
    batch_size = 512

    # TODO fix "Not Working" datasets
    dataset = "celeba"  # "mnist", "fashion-mnist", "cifar10", "celeba"

    # Noise Scheduling should be automated based on image size. See https://arxiv.org/abs/2301.10972 for more details.
    # https: // arxiv.org / abs / 2102.09672, https: // arxiv.org / abs / 2212.11972
    # "auto" is the default setting, which is based on the image size. "linear" is the default setting for image sizes 32x32 and below.

    # TODO: fix "Not Working" noise schedules              (SNR doesnt generate)
    schedule = "auto"  # "linear", "cosine" "sigmoid", "auto"   | No Working: "snrCosine", "SNR"     |    sigmoid > cosine > linear > snrCosine

    prediction_objective = "noise"  # Working: "noise", "recon"   | Not Working: "X0" (DDIM)

    images_per_sample = 10  # Number of Samples to take from the diffusion process for sequence generation

    steps = 100
    tSteps = 100

    DDPM = sample(dataset, steps, tSteps, generate_count, batch_size, images_per_sample, schedule, prediction_objective)
    DDPM.run()


class sample:
    def __init__(self, dataset, steps, tSteps, generate_count, batch_size, images_per_sample, schedule, prediction_objective):
        super().__init__()

        self.steps = steps
        self.tSteps = tSteps
        self.loadData = dataset
        self.scheduler = schedule
        self.batchSize = batch_size
        self.images_per_sample = images_per_sample
        self.predictionObjective = prediction_objective
        self.generateCount = generate_count

        self.plotEvery = 1
        self.seriesFrequency = int(self.tSteps / self.images_per_sample)

        self.saveGif = True  # Gif of X_hat_0 from Sampling
        self.collectSampleCharts = True  # Collect SNR Plots
        self.collectSequencePlots = True  # Collect Sequence Plots

        self.minimalData = False  # Use Minimal Data Subset, for testing purposes
        self.minDataSize = 10_000  # Size of minimalData Subset

        self.generationCount = 0
        self.datasetLength = 0

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
        Sqrt_Sigma = self.Beta  # Sigma Squared
        Alpha = 1.0 - self.Beta  # Alpha Schedule

        self.Alpha_Bar = torch.cumprod(Alpha, dim=0)  # Product Value of Alpha
        self.Sqrt_Alpha_Cumprod = torch.sqrt(self.Alpha_Bar)  # Square Root of Product Value of Alpha
        Alpha_Cumprod_Previous = func.pad(self.Alpha_Bar[:-1], (1, 0), value=1.0)  # Previous Product Value of Alpha   # Never forget the two months I lost to this bug
        self.Sqrt_1_Minus_Alpha_Cumprod = torch.sqrt(1.0 - self.Alpha_Bar)  # Square Root of 1 - Product Value of Alpha
        Log_one_minus_Alpha_Cumprod = torch.log(1.0 - self.Alpha_Bar)  # Log of 1 - Product Value of Alpha

        self.Sqrt_Recipricol_Alpha_Cumprod = torch.sqrt(1.0 / self.Alpha_Bar)  # Square Root of Reciprocal of Product Value of Alpha
        self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1 = torch.sqrt(1.0 / self.Alpha_Bar - 1)  # Square Root of Reciprocal of Product Value of Alpha - 1

        self.Posterior_Variance = self.Beta * (1.0 - Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar)  # Var(x_{t-1} | x_t, x_0)
        self.Posterior_Log_Variance_Clamp = torch.log(self.Posterior_Variance.clamp(min=1e-20))  # Log of Var(x_{t-1} | x_t, x_0)
        self.Posterior1 = (self.Beta * torch.sqrt(Alpha_Cumprod_Previous) / (1.0 - self.Alpha_Bar))  # 1 / (Var(x_{t-1} | x_t, x_0))
        self.Posterior2 = (1.0 - Alpha_Cumprod_Previous) * torch.sqrt(Alpha) / (1.0 - self.Alpha_Bar)  # (1 - Alpha_{t-1}) / (Var(x_{t-1} | x_t, x_0))

        # https://arxiv.org/abs/2303.09556
        snr = self.Alpha_Bar / (1 - self.Alpha_Bar)
        self.snrClip = snr.clone()

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

    def getSNRSchedule(self):
        snr_values = torch.linspace(0, self.tSteps, self.steps + 1, dtype=torch.float64) / self.tSteps
        noise_std = torch.sqrt(torch.tensor(1) / snr_values - torch.tensor(1))
        betas = 1 - (noise_std[1:] / noise_std[:-1]) ** 2
        return torch.clip(betas, 0, 0.999)

    def getLogSNRCosineSchedule(self, SNR_min, SNR_max):
        snr_values = torch.linspace(0, self.tSteps, self.steps + 1, dtype=torch.float64) / self.tSteps
        min_val = math.atan(math.exp(-0.5 * SNR_max))
        max_val = math.atan(math.exp(-0.5 * SNR_min))
        return torch.log(torch.tan(min_val + snr_values * (max_val - min_val)).clamp(min=1e-20))

    def getModel(self):
        if os.path.isfile(f"./data/ddpm model/{self.loadData} ddpm E100.pt"):
            model = torch.load(f"./data/ddpm model/{self.loadData} ddpm E100.pt")
        elif os.path.isfile(f"./data/ddpm model/{self.loadData} ddpm E25.pt"):
            model = torch.load(f"./data/ddpm model/{self.loadData} ddpm E25.pt")
        else:
            raise FileNotFoundError("Pretrained model not found. Please train a new model  or select a pretrained one.")
        model = nn.DataParallel(model) # Use multiple GPUs (Untested)
        optimizer = optim.Adam(model.parameters(), lr=2e-4, eps=1e-8)
        return model, optimizer

    def getDataset(self):
        train_data = None  # Initialize Datasets
        grey = GrayscaleTransform(self.imageSize, self.numChannels)
        RGB = RGBTransform(self.imageSize)

        # testData = torchvision.datasets.MNIST(root=".", download=True, train=False, transform=dataTransformation)
        if self.loadData == "mnist":
            train_data = torchvision.datasets.MNIST(root="./data", download=True, train=True, transform=grey)
        elif self.loadData == "fashion-mnist":
            train_data = torchvision.datasets.FashionMNIST(root="./data", download=True, train=True, transform=grey)
        elif self.loadData == "cifar10":
            train_data = torchvision.datasets.CIFAR10(root="./data", download=True, train=True, transform=RGB)
        elif self.loadData == "celeba":
            train_data = torchvision.datasets.CelebA(root="./data", download=True, transform=RGB)

        if self.minimalData == False:
            DataSet = DataLoader(train_data, batch_size=self.batchSize, shuffle=True, pin_memory=True, num_workers=4)
        else:
            # Subset of self.minDataSize Images for Training and Sampling
            subset = list(np.random.choice(np.arange(0, len(train_data)), self.minDataSize, replace=False))
            DataSet = DataLoader(train_data, batch_size=self.batchSize, pin_memory=True, num_workers=4, sampler=SubsetRandomSampler(subset))
        if self.loadData == "celeba":
            Label = train_data.identity
        else:
            Label = train_data.targets
        return train_data, DataSet, Label  # testData

    def getNormHistData(self, data_0, data_1, binCount):
        data_0 = data_0.cpu().detach().numpy().flatten()
        data_1 = data_1.cpu().detach().numpy().flatten()
        bin_range = (min(np.min(data_0), np.min(data_1)), max(np.max(data_0), np.max(data_1)))
        bin_interval = (bin_range[1] - bin_range[0]) / binCount  # 50 Bins
        bins = np.arange(bin_range[0], bin_range[1] + bin_interval, bin_interval)
        return data_0, data_1, bins

    @torch.no_grad()
    def getNoiseHistogram(self, noise, pred_noise):
        plt.figure(figsize=(7, 7))
        np_noise, np_pred_noise, bins = self.getNormHistData(noise, pred_noise, binCount=128)
        plt.hist(np_noise, density=True, bins=bins, alpha=.60, label="True Noise")
        plt.hist(np_pred_noise, density=True, bins=bins, alpha=.60, label="Prediction Noise")
        self.plotGraphHelper("Noise", "Frequency", "Noise Histogram", "Images/" + str(self.loadData) + "/noise/Epoch =" + str(self.generationCount) + " (" + str(self.scheduler) + ") Noise Histogram.jpg")

    @torch.no_grad()
    def getDiffuionGif(self, model, image):
        convertToImage = ConvertToImage()
        fig, ax = plt.subplots()
        imgs = []
        frameInput = None
        img = torch.randn_like(image, device=device)  # Noise for initial image
        for step in tqdm(reversed(range(0, self.steps))):
            img = self.p_sample(model, img, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
            if self.numChannels == 1:
                frameInput = plt.imshow(convertToImage(img[0]), cmap="gray")
            elif self.numChannels == 3:
                frameInput = plt.imshow(convertToImage(img[0]))
            imgs.append([frameInput])
        plt.title("" + str(self.loadData) + " diffusion")
        plt.axis("off")
        animate = ArtistAnimation(fig, imgs, interval=25, blit=True, repeat_delay=50, repeat=True)  # interval was 25
        animate.save("Images/" + str(self.loadData) + "/" + str(self.loadData) + " diffusion.gif")
        fig.clf()

    def prepLabels(self, labels):
        if self.loadData == "cifar10":
            labels_tensor = torch.tensor(labels, device=device)
            label = labels_tensor[:self.batchSize].clone().detach().to(device)
        else:
            label = labels[:self.batchSize].clone().detach().to(device)
        idx = np.arange(len(label))
        np.random.shuffle(idx)
        shuffled = idx[:self.batchSize]
        shuffled = torch.tensor(shuffled).to(device)
        return label, shuffled

    def saveImage(self, img, msg):
        convertToImage = ConvertToImage()
        if len(img.shape) == 4:
            img = img[0, :, :, :]

        if self.numChannels == 1:
            plt.imshow(convertToImage(img), cmap="gray")
        elif self.numChannels == 3:
            plt.imshow(convertToImage(img))
        plt.title("Gen = " + str("{:.6f}".format(self.generationCount)))
        plt.axis("off")
        plt.savefig(msg)
        plt.close()

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

    def plotSNRGraphs(self, FP, RP, xLabel, yLabel, title, savePath, label1, label2):
        plt.figure(figsize=(10, 10))
        plt.plot(FP, label=label1, markersize=3)
        plt.plot(RP, label=label2, markersize=3)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.legend()
        plt.savefig(savePath)
        plt.close()

    def printSystemDynamics(self):
        print("Cuda: ", torch.cuda.is_available())
        print("Device Count: ", torch.cuda.device_count())
        print("Device: ", device)
        print("Device Name: ", torch.cuda.get_device_name(device))

        print("\nImages Per. Sample: ", self.images_per_sample)
        print("Steps Between Images: ", self.seriesFrequency)

    def printModelInfo(self, model):
        print("\nModel Info:")
        print("\tModel: ", model)
        print("\tModel Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    def printSampleInfo(self):
        print("\nHyperparameters:")
        print("\tData: ", self.loadData)
        print("\tSteps: ", self.steps)
        print("\tGeneration Couint: ", self.generateCount)
        print("\tPrediction Objective: ", self.predictionObjective)

    def diffusionSubplotting(self, img, step, ax):
        ax.set_title("Time = " + str(step) + "")
        ax.axis("off")
        if self.numChannels == 1:
            ax.imshow((img[0].cpu().squeeze().numpy() + 1.0) * 255 / 2, cmap="gray")
        elif self.numChannels == 3:
            ax.imshow(np.transpose((img.cpu().numpy() + 1.0) / 2.0, (1, 2, 0)))

    def histogramPlotHelper(self, model, img, xlabel, ylabel, title, legend1, legend2, t, ax):
        prediction = model(img, t)
        norm_img, norm_pred, bins = self.getNormHistData(img, prediction, binCount=128)
        ax.hist(norm_img, density=True, bins=bins, alpha=.60, label=legend1)
        ax.hist(norm_pred, density=True, bins=bins, alpha=.60, label=legend2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    @torch.no_grad()
    def plotDistributionHistogram(self, model, image, t):
        num_subplots = (self.steps // self.seriesFrequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 15, 5))
        img = torch.randn_like(image, device=device)  # Noise for initial image
        step = self.steps
        title = "Distribution at Time = " + str(step) + " " + str(self.scheduler) + ""
        self.histogramPlotHelper(model=model, img=img, xlabel="Noise", ylabel="Frequency", title=title, legend1="Noise Distribution", legend2="Predicted Distribution", t=t, ax=axes[0])
        axes[0].legend()
        counter = 1
        for step in tqdm(reversed(range(0, self.steps))):
            img = self.p_sample(model, img, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
            if step % self.seriesFrequency == 0:
                title = "Distribution at Time = " + str(step) + " (" + str(self.scheduler) + ")"
                self.histogramPlotHelper(model=model, img=img, xlabel="Noise", ylabel="Frequency", title=title, legend1="Noise Distribution", legend2="Predicted Distribution", t=t, ax=axes[counter])
                axes[counter].legend()
                counter += 1
        plt.savefig("Images/" + str(self.loadData) + "/sequence plots/distribution series/Epoch = " + str(self.generationCount) + ".jpg")
        plt.close()

    @torch.no_grad()
    def plotForwardProcessImageCorruption(self, model, XT):
        num_subplots = (self.steps // self.seriesFrequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        counter = 0
        for step in tqdm(range(0, self.steps)):
            if step % self.seriesFrequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
            XT = self.p_sample(model, XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[counter])  # Plot final image
        plt.savefig("Images/" + str(self.loadData) + "/Example of Gradual " + str(self.loadData) + " Corruption.jpg")
        plt.close()
        return XT

    @torch.no_grad()
    def plotReverseProcessImageSynthesis(self, model, XT):
        num_subplots = (self.steps // self.seriesFrequency) + 1
        fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=((self.images_per_sample / 2) * 5, 4))
        XT = torch.randn_like(XT, device=device)  # Noise for initial image
        self.diffusionSubplotting(XT[0], self.steps, ax=axes[0])  # Plot initial noise
        counter = 1
        for step in tqdm(reversed(range(0, self.steps))):
            XT = self.p_sample(model, XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
            if step % self.seriesFrequency == 0:
                self.diffusionSubplotting(XT[0], step, ax=axes[counter])  # Plot image at each step
                counter += 1
        plt.savefig("Images/" + str(self.loadData) + "/sequence plots/image series/Epoch = " + str(self.generationCount) + ".jpg")
        plt.close()

    @torch.no_grad()
    def plotForwardTrajectories(self, model, X0, t, savePath):
        XT = X0
        vlb = []
        for step in tqdm(range(0, self.steps)):
            XT = self.p_sample(model, XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
            vlb.append(self.vlb(model, X0, XT, t).mean().item())  # VLB Calculation
        self.plotGraph(arr=vlb, label="Variational Lower Bound", xlabel="Timestep", ylabel="Variational Lower Bound", title="Forward Process Variational Lower Bound", savePath=savePath + " B FP VLB.jpg")
        return vlb, XT

    @torch.no_grad()
    def plotReverseTrajectories(self, model, X0, XT, t, savePath):
        vlb = []
        for step in tqdm(reversed(range(0, self.steps))):
            XT = self.p_sample(model, XT, torch.full((self.batchSize,), step, device=device, dtype=torch.long))
            vlb.append(self.vlb(model, X0, XT, t).mean().item())  # VLB Calculation
        self.plotGraph(arr=vlb, label="Variational Lower Bound", xlabel="Timestep", ylabel="Variational Lower Bound", title="Reverse Process Variational Lower Bound", savePath=savePath + " E RP VLB.jpg")
        return vlb, XT

    def vlb(self, model, X0, Xt, t):  # Variational lower bound
        posterior_mean, posterior_model_variance, _ = self.q_posterior_mean_variance(X0, Xt, t)
        prior_mean, prior_model_variance, _ = self.p_prior_mean_variance(model, Xt, t)
        log_likelihood = self.log_likelihood(X0, Xt, t)
        kl_divergence = self.kl_divergence(posterior_mean, posterior_model_variance, prior_mean, prior_model_variance)
        return (log_likelihood - kl_divergence).cpu().numpy()

    def log_likelihood(self, X0, Xt, t):  # log p(x_t | x_{t-1}, x_0)
        mean, variance = self.p_posterior_mean_variance(X0, Xt, t)
        return -0.5 * torch.log(2 * math.pi * variance) - 0.5 * (Xt - mean) ** 2 / variance

    def kl_divergence(self, posterior_mean, posterior_model_variance, prior_mean, prior_model_variance):  # KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t))
        return 0.5 * (torch.log(prior_model_variance) - torch.log(posterior_model_variance) + (posterior_model_variance + (posterior_mean - prior_mean) ** 2) / prior_model_variance - 1)

    def p_posterior_mean_variance(self, X0, Xt, t):  # p(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + \
                         self.getExtract(self.Posterior2, t, Xt.shape) * Xt  # p(x_{t-1} | x_t, x_0) = N(posterior1_mean, posterior_model)
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)  # p(x_{t-1} | x_t, x_0) = N(posterior2_mean, posterior_model_variance)
        return posterior_mean, posterior_model_variance

    def getExtract(self, tensor: torch.Tensor, t: torch.Tensor, X):  # Extracts the correct value from a tensor
        out = tensor.gather(-1, t.cpu()).float()
        return out.reshape(t.shape[0], *((1,) * (len(X) - 1))).to(t.device)

    def q_posterior_mean_variance(self, X0, Xt, t):  # q(x_{t-1} | x_t, x_0) = N(posterior_mean, posterior_model_variance)
        posterior_mean = self.getExtract(self.Posterior1, t, Xt.shape) * X0 + \
                         self.getExtract(self.Posterior2, t, Xt.shape) * Xt
        posterior_model_variance = self.getExtract(self.Posterior_Variance, t, Xt.shape)
        posterior_model_log_variance_clamp = self.getExtract(self.Posterior_Log_Variance_Clamp, t, Xt.shape)
        return posterior_mean, posterior_model_variance, posterior_model_log_variance_clamp

    def q_sample(self, X0, t):  # Sample from q(Xt | X0) = N(x_t; sqrt(alpha_bar_t) * x_0, sqrt(1 - alpha_bar_t) * noise, t
        noise = torch.randn_like(X0)  # Sample from N(0, I)
        QSample = self.getExtract(self.Sqrt_Alpha_Cumprod, t, X0.shape) * X0 + \
                  self.getExtract(self.Sqrt_1_Minus_Alpha_Cumprod, t, X0.shape) * noise  # Sample from q(Xt | X0)
        return QSample, noise

    def pred_X0_from_XT(self, Xt, noise, t):  # p(x_{t-1} | x_t)
        return self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod, t, Xt.shape) * Xt - \
            self.getExtract(self.Sqrt_Recipricol_Alpha_Cumprod_Minus_1, t, Xt.shape) * noise  # Sample from p(x_{t-1} | x_t)


    def p_prior_mean_variance(self, model, Xt, t):  # Sample from p_{theta}(x_{t-1} | x_t) & q(x_{t-1} | x_t, x_0)
        with autocast():
            X0_prediction = self.pred_X0_from_XT(Xt.float(), model(Xt.float(), t), t)  # p(x_{t-1} | x_t)
        if self.loadData != "gaussian":
            X0_prediction = torch.clamp(X0_prediction, -1., 1.)  # Clamp to [-1, 1]
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(X0_prediction, Xt, t)  # Sample from q(x_{t-1} | x_t, x_0)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, model, sample, t):  # Sample from p_{theta}(x_{t-1} | x_t) = N(x_{t-1}; UNet(x_{t}, t), sigma_bar_t * I)
        mean, posterior_variance, posterior_log_variance = self.p_prior_mean_variance(model, sample, t)  # Sample from p_{theta}(x_{t-1} | x_t)
        noise = torch.randn_like(sample)  # Sample from N(0, I)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(sample.shape) - 1))))  # Mask for t != 0
        return mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise  # Sample from p_{theta}(x_{t-1} | x_t)

    def generate(self, model, X0, idx):
        t = torch.randint(0, self.steps, (X0.shape[0],), device=device).long()
        XT, epsilon = self.q_sample(X0, t)
        with autocast():
            epsilon_theta = model(XT, t)

        # When idx == 0, we are at the start of the epoch to print images  |  When idx == [after last batch], we are at the end of the epoch to print images
        if idx == 0 and self.generateCount % self.plotEvery == 0 and self.generateCount != 0:
            savePath = str("Images/" + str(self.loadData) + "/graphs/SNR/E=" + str(self.generationCount) + "")

            print("\t\t...Generating Images...")
            plt.figure(figsize=(10, 10))
            self.saveImage(X0, savePath + " A Input.jpg")
            fp_vlb, final_noise = self.plotForwardTrajectories(model, X0, t, savePath)

            plt.figure(figsize=(10, 10))
            self.saveImage(final_noise, savePath + " C Corrupted.jpg")
            rp_vlb, X0_Hat = self.plotReverseTrajectories(model, X0, final_noise, t, savePath)

            plt.figure(figsize=(10, 10))
            self.saveImage(X0_Hat, savePath + " F Reconstructed.jpg")

        if idx == 0 and self.collectSequencePlots:
            print("\t\t...Generating Image Sequences... \n")
            self.getNoiseHistogram(epsilon, epsilon_theta)
            self.plotDistributionHistogram(model, X0, t)
            self.plotReverseProcessImageSynthesis(model, X0)

    def sampleLoop(self, dataset, model):
        for idx, (X0, labels) in enumerate(dataset):
            X0, labels = X0.to(device), labels.to(device)
            self.generate(model, X0.to(device), idx)
        self.generationCount += 1

    def run(self):

        # Print System Dynamics
        self.printSystemDynamics()

        # Get Data
        print("\n\t\t...Loading Data...")

        train_data, dataset, label = self.getDataset()
        img = next(iter(dataset))[0].to(device)
        self.datasetLength = len(train_data)

        # Get Models
        print("\n\t\t...Loading Models...")
        model, optimizer = self.getModel()

        # Waiting for Triton to support Windows
        # model = torch.compile(model)

        self.saveImage(img, "Images/" + str(self.loadData) + "/Example Input Image.jpg")

        if self.collectSequencePlots:
            print("\n\t\t...Generating Example Corruption Sequence...")
            finalNoiseImg = self.plotForwardProcessImageCorruption(model, img)

        # Print Model Information
        # self.PrintModelInfo(model)

        startTime = time.time()
        print("\nSample Start Time: " + str(datetime.datetime.now()))

        while self.generationCount != self.generateCount:
            print(f"\n     ------------ Generation {self.generationCount} ------------ ")
            self.sampleLoop(dataset, model)  # Sampling done intermittently during Sample

        endTime = time.time()
        print("\nSample Completion Time: " + str(datetime.datetime.now()))
        print(f"Total Sampling Time: {(endTime - startTime) / 60:.2f} mins")

        if self.saveGif:
            print("\n\t   ...Generating Gif... ")
            self.getDiffuionGif(model, img)

if __name__ == "__main__":
    main()