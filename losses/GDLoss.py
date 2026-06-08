import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import find_loss_def


class AlphaHead(nn.Module):
    """Predict sample-wise rain-streak orientation weight alpha in (0, 1)."""

    def __init__(self, in_ch=6, hidden=64, init_alpha=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        init_alpha = float(init_alpha)
        init_alpha = min(max(init_alpha, 1e-6), 1 - 1e-6)
        b0 = torch.log(torch.tensor(init_alpha) / (1 - torch.tensor(init_alpha)))
        with torch.no_grad():
            self.fc2.weight.zero_()
            self.fc2.bias.fill_(b0.item())

    def forward(self, clean, pred):
        x = torch.cat([clean, pred], dim=1)
        f = self.conv(x)
        z = f.mean(dim=(2, 3))
        z = F.relu(self.fc1(z), inplace=True)
        alpha = torch.sigmoid(self.fc2(z))
        return alpha.view(-1, 1, 1, 1)


class Loss(nn.Module):
    def __init__(self, init_alpha=0.15, alpha_reg_lambda=0.0, alpha_prior=0.15):
        super(Loss, self).__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0),
        )

        self.loss = find_loss_def('CharbonnierLoss')()
        self.alpha_head = AlphaHead(in_ch=6, hidden=64, init_alpha=init_alpha)
        self.alpha_reg_lambda = float(alpha_reg_lambda)
        self.alpha_prior = float(alpha_prior)

    def compute_gradient(self, image, alpha):
        _, channels, _, _ = image.shape
        sobel_x = self.sobel_x.to(image.device).repeat(channels, 1, 1, 1)
        sobel_y = self.sobel_y.to(image.device).repeat(channels, 1, 1, 1)

        grad_x = F.conv2d(image, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(image, sobel_y, padding=1, groups=channels)

        grad_mag_sq = alpha * (grad_x ** 2) + (1 - alpha) * (grad_y ** 2)
        return torch.sqrt(torch.clamp(grad_mag_sq, min=1e-6))

    def forward(self, pred_image, clean_image):
        alpha = self.alpha_head(clean_image, pred_image)

        grad_clean = self.compute_gradient(clean_image, alpha)
        grad_pred = self.compute_gradient(pred_image, alpha)

        loss = self.loss(grad_clean, grad_pred)

        if self.alpha_reg_lambda > 0:
            prior = (alpha.view(alpha.size(0), -1).mean(dim=1) - self.alpha_prior) ** 2
            loss = loss + self.alpha_reg_lambda * prior.mean()

        return loss
