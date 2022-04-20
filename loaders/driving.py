from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import re
import cv2
from pathlib import Path
import torchvision.transforms as transforms

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def augment(image, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    return image, steering_angle

transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 127.5) - 1.0)])

class DrivingData(Dataset):
    def __init__(
            self, file_path, device,
            target_size=(320, 240), # (width, height)
            crop_size = (200, 200)):

        self.processed_data_file = Path(file_path)
        self.device = device
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)

        # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.imgs = None
        self.ground_truths = []

        if self.processed_data_file.is_file():
            self._load_processed_data()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x / 127.5) - 1.0)
        ])

    def _load_processed_data(self):
        if self.imgs is None:
            print("Loading processed data from {}...".format(self.processed_data_file))
            with np.load(self.processed_data_file) as data:
                self.imgs = data['imgs']
                self.ground_truths = data['ground_truths']
            print("Images: ", self.imgs.shape)
            print("Ground truths: ", self.ground_truths.shape)
            print("Done!")

    def _decode_experiment_dir(self, dir_subpath):
        # Load steerings or labels in the experiment dir
        steerings_filename = os.path.join(dir_subpath, "sync_steering.txt")
        labels_filename = os.path.join(dir_subpath, "labels.txt")

        # Try to load steerings first. Make sure that the steering angle or the
        # label file is in the first column. Note also that the first line are
        # comments so it should be skipped.
        try:
            ground_truths = np.loadtxt(steerings_filename, usecols=0,
                                  delimiter=',', skiprows=1)
            exp_type = 1
        except OSError as e:
            # Try load collision labels if there are no steerings
            try:
                ground_truths = np.loadtxt(labels_filename, usecols=0)
                exp_type = 0
            except OSError as e:
                print("Neither steerings nor labels found in dir {}".format(
                dir_subpath))
                raise IOError


        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    self.filenames.append(os.path.relpath(absolute_path,
                            self.directory))
                    self.ground_truths.append(ground_truths[frame_number])
                    self.exp_type.append(exp_type)
                    self.samples += 1

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=False),
                key=lambda tpl: tpl[0])

    def __len__(self):
        return self.ground_truths.shape[0]

    def __getitem__(self, index):
        img = self.imgs[index]
        steering_angle = self.ground_truths[index]
        return img, steering_angle


def get_iterator_driving(file_path, device, batch_size=1, num_workers=0):
    """
    returns an iterator over DRIVING dataset batches
    :param file_path: path to .npz file containing a list of tuples
         each of them representing a path to an image and it class
    :param device:
    :param batch_size:
    :return: torch.utils.DataLoader object constructed from DRIVING dataset object
    """

    dataset = DrivingData(file_path, device=device)
    if 'test' in file_path or 'validation' in file_path:
        iterator = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    else:
        iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return iterator