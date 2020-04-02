import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import numbers
import random


def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
    hue[Cmax== r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6. # making hue range as [0, 1.0)
    hue = hue.unsqueeze(dim=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.] = 0.
    saturation = saturation.to(_device)
    saturation = saturation.unsqueeze(dim=1)

    value = Cmax
    value = value.to(_device)
    value = value.unsqueeze(dim=1)

    return torch.cat((hue, saturation, value), dim=1)#.type(torch.FloatTensor).to(_device)
    # return hue, saturation, value

def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = hsv.device

    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]

    c = value * saturation
    x = - c * (torch.abs((hue / 60.) % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)

    rgb_prime = torch.zeros_like(hsv).to(_device)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb.to(_device)

    return torch.clamp(rgb, 0, 1)

class RandomResizeCropLayer(nn.Module):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        '''
            Inception Crop

            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        '''
        super(RandomResizeCropLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)
        self.scale = scale
        self.ratio = ratio

    def forward(self, input):
        _device = input.device
        N, _, width, height = input.shape

        _theta = self._eye.repeat(N, 1, 1)

        # N * 10 trial
        area = height * width
        target_area = np.random.uniform(*self.scale, N * 10) * area
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = np.exp(np.random.uniform(*log_ratio, N * 10))

        # If doesn't satisfy ratio condition, then do central crop
        w = np.round(np.sqrt(target_area * aspect_ratio))
        h = np.round(np.sqrt(target_area / aspect_ratio))
        cond = (0 < w) * (w <= width) * (0 < h) * (h <= height)
        w = w[cond]
        h = h[cond]
        if len(w) > N:
            inds = np.random.choice(len(w), N, replace=False)
            w = w[inds]
            h = h[inds]
        transform_len = len(w)

        r_w_bias = np.random.randint(w - width, width - w + 1) / width
        r_h_bias = np.random.randint(h - height, height - h + 1) / height
        w = w / width
        h = h / height

        _theta[:transform_len, 0, 0] = torch.tensor(w, device=_device)
        _theta[:transform_len, 1, 1] = torch.tensor(h, device=_device)
        _theta[:transform_len, 0, 2] = torch.tensor(r_w_bias, device=_device)
        _theta[:transform_len, 1, 2] = torch.tensor(r_h_bias, device=_device)

        grid = F.affine_grid(_theta, input.size())
        output = F.grid_sample(input, grid, padding_mode='reflection')

        return output

class HorizontalFlipLayer(nn.Module):
    def __init__(self):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(HorizontalFlipLayer, self).__init__()

        _eye = torch.eye(2, 3)
        self.register_buffer('_eye', _eye)

    def forward(self, input):
        _device = input.device

        N = input.size(0)
        _theta = self._eye.repeat(N, 1, 1)
        r_sign = torch.bernoulli(torch.ones(N, device=_device) * 0.5) * 2 - 1
        _theta[:, 0, 0] = r_sign
        grid = F.affine_grid(_theta, input.size())
        output = F.grid_sample(input, grid, padding_mode='reflection')
        return output

class RandomColorGrayLayer(nn.Module):
    def __init__(self, p=0.0):
        super(RandomColorGrayLayer, self).__init__()
        self.prob = p

    def forward(self, input):
        _device = input.device

        inds = torch.tensor(np.random.choice(
            [True, False], len(input), p=[self.prob, 1 - self.prob])).cpu()#.to(_device)

        R = input[inds, 0, :, :]
        G = input[inds, 1, :, :]
        B = input[inds, 2, :, :]

        L = R * 299. / 1000. + G * 587. / 1000. + B * 114. / 1000.
        L = L.unsqueeze(dim=1)

        input[inds] = torch.cat((L, L, L), dim=1)
        outputs = input

        return outputs

class ColorJitterLayer(nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0):
        super(ColorJitterLayer, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.prob = p

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def convert_gray(self, x):
        # Same as PIL.convert('L')

        R = x[:, 0, :, :]
        G = x[:, 1, :, :]
        B = x[:, 2, :, :]

        L = R * 299. / 1000. + G * 587. / 1000. + B * 114. / 1000.

        return torch.cat((L, L, L), dim=1)

    def adjust_contrast(self, x):
        """
            Args:
                x: torch tensor img (rgb type)

            Factor: torch tensor with same length as x
                    0 gives gray solid image, 1 gives original image,

            Returns:
                torch tensor image: Brightness adjusted
        """

        _device = x.device
        factor = torch.tensor(np.random.uniform(self.contrast[0],
                                                self.contrast[1],
                                                len(x)), dtype=torch.float).to(_device)

        means = torch.mean(x, dim=(2, 3), keepdim=True)

        return torch.clamp((x - means)
                           * factor.view(len(x), 1, 1, 1) + means, 0, 1)

    def adjust_hue(self, x):

        _device = x.device
        factor = torch.tensor(np.random.uniform(self.hue[0],
                                                self.hue[1],
                                                len(x)), dtype=torch.float).to(_device)

        x = rgb2hsv(x)
        h = x[:, 0, :, :]
        h += (factor.view(len(x), 1, 1) * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        x = hsv2rgb(x)

        return x

    def adjust_brightness(self, x):
        """
            Args:
                x: torch tensor img (rgb type)

            Factor:
                torch tensor with same length as x
                0 gives black image, 1 gives original image,
                2 gives the brightness factor of 2.

            Returns:
                torch tensor image: Brightness adjusted
        """

        _device = x.device
        factor = torch.tensor(np.random.uniform(self.brightness[0],
                                                self.brightness[1],
                                                len(x)), dtype=torch.float).to(_device)

        x = rgb2hsv(x)
        x[:, 2, :, :] = torch.clamp(x[:, 2, :, :]
                                     * factor.view(len(x), 1, 1), 0, 1)
        x = hsv2rgb(x)

        return torch.clamp(x, 0, 1)

    def adjust_saturate(self, x):
        """
            Args:
                x: torch tensor img (rgb type)

            Factor:
                torch tensor with same length as x
                0 gives black image and white, 1 gives original image,
                2 gives the brightness factor of 2.

            Returns:
                torch tensor image: Brightness adjusted
        """

        _device = x.device
        factor = torch.tensor(np.random.uniform(self.saturation[0],
                                                self.saturation[1],
                                                len(x)), dtype=torch.float).to(_device)

        x = rgb2hsv(x)
        x[:, 1, :, :] = torch.clamp(x[:, 1, :, :]
                                    * factor.view(len(x), 1, 1), 0, 1)
        x = hsv2rgb(x)

        return torch.clamp(x, 0, 1)

    def transform(self, inputs):

        # Shuffle transform
        transform_list = [self.adjust_brightness, self.adjust_contrast,
                          self.adjust_hue, self.adjust_saturate]
        #transform_list = [self.adjust_saturate]

        # Shuffle transform
        random.shuffle(transform_list)

        for t in transform_list:
            inputs = t(inputs)

        return inputs

    def forward(self, inputs):

        _device = inputs.device

        inds = torch.tensor(np.random.choice(
            [True, False], len(inputs), p=[self.prob, 1 - self.prob])).to(_device)

        inputs[inds] = self.transform(inputs[inds])

        return inputs