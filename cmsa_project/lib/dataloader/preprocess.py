import random
import numpy as np
from PIL import Image, ImageEnhance

from torchvision.transforms import ToTensor as torchtotensor
from torchvision.transforms import ColorJitter


class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class Compose_img(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Random_color_jitter(object):
    def __init__(self, prob):
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    # def __call__(self, imgs, labels):
    #     if random.random() < self.prob:
    #         res_img = []
    #         res_label = []
    #         # b = 0.1, c = 0.03, s=0.03
    #         tf = ColorJitter(brightness=[0.01, 0.1], contrast=[0.01, 0.03], saturation=[0.01, 0.03], hue=0)
    #         for img, label in zip(imgs, labels):
    #             img = tf(img)
    #             res_img.append(img)
    #             res_label.append(label)
    #         return res_img, res_label
    #     else:
    #         return imgs, labels

    def __call__(self, imgs, labels):
        if random.random() < self.prob:
            res_img = []
            res_label = []
            bright_intensity = random.randint(1, 5) / 10.0
            contrast_intensity = random.randint(1, 5) / 10.0
            color_intensity = random.randint(0, 2) / 10.0
            sharp_intensity = random.randint(0, 3) / 10.0
            for img, label in zip(imgs, labels):
                img = ImageEnhance.Brightness(img).enhance(bright_intensity)
                img = ImageEnhance.Contrast(img).enhance(contrast_intensity)
                img = ImageEnhance.Color(img).enhance(color_intensity)
                img = ImageEnhance.Sharpness(img).enhance(sharp_intensity)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Random_rotation(object):
    def __init__(self, prob, angle=15):
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob
        self.angle = angle

    def __call__(self, imgs, labels):
        if random.random() < self.prob:
            res_img = []
            res_label = []
            r = self.angle
            angle = np.random.randint(-r, r)
            for img, label in zip(imgs, labels):
                img = img.rotate(angle, Image.BICUBIC)
                label = label.rotate(angle, Image.BICUBIC)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Random_crop_Resize_Video(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class Random_horizontal_flip_video(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        """
        :param prob: should be (0,1)
        """
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels):
        """
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        """
        if random.random() < self.prob:
            res_img = []
            res_label = []
            for img, label in zip(imgs, labels):
                img, label = self._horizontal_flip(img, label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Resize_video(object):
    def __init__(self, height, width, img_only=False):
        self.height = height
        self.width = width
        self.img_only = img_only

    def __call__(self, imgs, labels=None):
        res_img = []
        res_label = []
        if labels is not None:
            if not self.img_only:
                for img, label in zip(imgs, labels):
                    res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
                    res_label.append(label.resize((self.width, self.height), Image.NEAREST))
                return res_img, res_label
            else:
                for img in imgs:
                    res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
                return res_img, labels
        else:
            for img in imgs:
                res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            return res_img


class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels=None):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        if labels is not None:
            return res_img, labels
        else:
            return res_img


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, img):
        for i in range(3):
            img[:, :, i] -= float(self.mean[i])
        for i in range(3):
            img[:, :, i] /= float(self.std[i])
        return img


class toTensor_video(object):
    def __init__(self, img_only=False):
        self.totensor = torchtotensor()
        self.img_only = img_only

    def __call__(self, imgs, labels=None):
        res_img = []
        res_label = []
        if labels is not None:
            if not self.img_only:
                for img, label in zip(imgs, labels):
                    img, label = self.totensor(img), self.totensor(label).long()
                    res_img.append(img)
                    res_label.append(label)
                return res_img, res_label
            else:
                for img in imgs:
                    img = self.totensor(img)
                    res_img.append(img)
                return res_img, labels
        else:
            for img in imgs:
                img = self.totensor(img)
                res_img.append(img)
            return res_img
