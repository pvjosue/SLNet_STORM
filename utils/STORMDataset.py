import torch
from torch.utils import data
from tifffile import imread
import numpy as np
import torch.nn.functional as F


class STORMDatasetFull(data.Dataset):
    def __init__(self, imgs_path, img_shape, images_to_use=None, temporal_shifts=[0, 1, 2],
                 use_random_shifts=False, maxWorkers=10):
        # Initialize arguments
        self.images_to_use = images_to_use
        self.temporal_shifts = temporal_shifts
        self.n_frames = len(temporal_shifts)
        self.use_random_shifts = use_random_shifts
        self.img_shape = img_shape


        self.img_dataset = imread(imgs_path, maxworkers=maxWorkers)
        n_frames, h, w = self.img_dataset.shape

        # Calculate the median of the whole dataset
        self.median, self.indexes = torch.from_numpy(self.img_dataset.astype(np.float32)).median(dim=0)

        # If no images are specified to use, list sequentially
        if images_to_use is None:
            images_to_use = list(range(n_frames))
        self.n_images = min(len(images_to_use), n_frames)

        n_images_to_load = max(images_to_use) + max(temporal_shifts) + 1

        # Create image storage
        self.stacked_views = torch.zeros(n_images_to_load, self.img_shape[0], self.img_shape[1], dtype=torch.float32)

        for nImg in range(n_images_to_load):

            # Load the images indicated from the user
            curr_img = nImg  # images_to_use[nImg]

            image = torch.from_numpy(np.array(self.img_dataset[curr_img, :, :]).astype(np.float32)).type(torch.float32)

            image = self.pad_img_to_min(image)
            self.stacked_views[nImg, ...] = image

        print('Loaded ' + str(self.n_images))

    def __len__(self):
        """Denotes the total number of samples"""
        return self.n_images

    def get_n_temporal_frames(self):
        return len(self.temporal_shifts)

    def get_max(self):
        """Get max intensity from images for normalization"""
        return self.stacked_views.float().max().type(self.stacked_views.type()), \
                self.stacked_views.float().max().type(self.stacked_views.type())

    def get_statistics(self):
        """Get mean and standard deviation from images for normalization"""
        return self.stacked_views.float().mean().type(
            self.stacked_views.type()), self.stacked_views.float().std().type(self.stacked_views.type())

    def standarize(self, stats=None):
        mean_imgs, std_imgs, mean_imgs_s, std_imgs_s, mean_vols, std_vols = stats
        self.stacked_views[...] = (self.stacked_views[...] - mean_imgs) / std_imgs

    def pad_img_to_min(self, image):
        min_size = min(image.shape[-2:])
        img_pad = [min_size - image.shape[-1], min_size - image.shape[-2]]
        img_pad = [img_pad[0] // 2, img_pad[0] // 2, img_pad[1], img_pad[1]]
        image = F.pad(image.unsqueeze(0).unsqueeze(0), img_pad)[0, 0]
        return image

    def __getitem__(self, index):
        n_frames = self.get_n_temporal_frames()
        new_index = self.images_to_use[index]

        temporal_shifts_ixs = self.temporal_shifts
        # if self.use_random_shifts:
        #     temporal_shifts_ixs = torch.randint(0, self.n_images-1,[3]).numpy()
        #     newIndex = 0

        indices = [new_index + temporal_shifts_ixs[i] for i in range(n_frames)]

        views_out = self.stacked_views[indices, ...]

        return views_out

    @staticmethod
    def read_tiff_stack(filename, out_datatype=torch.float32):
        tiffarray = imread(filename)
        try:
            max_val = torch.iinfo(out_datatype).max
        except:
            max_val = torch.finfo(out_datatype).max
        out = np.clip(tiffarray.raw_images, 0, max_val)
        return torch.from_numpy(out).permute(1, 2, 0).type(out_datatype)
