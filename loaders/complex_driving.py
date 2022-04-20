from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
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

class ComplexDrivingData(Dataset):
    def __init__(
            self, file_path, device,
            target_size=(320, 240), # (width, height)
            crop_size = (200, 200),
            color_mode='grayscale'):

        self.data_file = Path(file_path)
        self.device = device
        self.target_size = tuple(target_size)
        self.crop_size = tuple(crop_size)

        # Idea = associate each filename with a corresponding steering or label
        self.filenames = []
        self.imgs = None
        self.ground_truths = []

        if self.data_file.is_file():
            self._load_data()

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            self.image_shape = self.crop_size + (3,)
        else:
            self.image_shape = self.crop_size + (1,)

    def _load_data(self):
        if self.imgs is None:
            print("Loading data from {}...".format(self.data_file))
            with np.load(self.data_file) as data:
                self.filenames = data['file_names']
                self.ground_truths = data['ground_truths']
            print("Filenames: ", self.filenames.shape)
            print("Ground truths: ", self.ground_truths.shape)
            print("Done!")

    def __len__(self):
        return self.ground_truths.shape[0]

    def __getitem__(self, index):
        img_filename = os.path.join("data", self.filenames[index])
        img = load_img(img_filename, self.color_mode == "grayscale", self.target_size, self.crop_size)
        steering_angle = self.ground_truths[index]
        return img, steering_angle

def load_img(path, grayscale=False, target_size=None, crop_size=None):
    """
    Load an image.
    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.
        crop_size: Either `None` (default to original size)
            or tuple of ints `(img_width, img_height)`.
    # Returns
        Image as numpy array.
    """

    img = cv2.imread(path)
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, target_size)

    if crop_size:
        img = central_image_crop(img, crop_size[0], crop_size[1])

    if grayscale:
        img = img.reshape((1, img.shape[0], img.shape[1]))

    return np.asarray(img, dtype=np.float32)

def central_image_crop(img, crop_width=150, crop_heigth=150):
    """
    Crop the input image centered in width and starting from the bottom
    in height.
    # Arguments:
        crop_width: Width of the crop.
        crop_heigth: Height of the crop.
    # Returns:
        Cropped image.
    """
    half_the_width = int(img.shape[1] / 2)
    img = img[img.shape[0] - crop_heigth: img.shape[0],
              half_the_width - int(crop_width / 2):
              half_the_width + int(crop_width / 2)]
    return img

def get_iterator_complex_driving(file_path, device, batch_size=1, num_workers=0):
    """
    returns an iterator over UDACITY-DRIVING dataset batches
    :param file_path: path to .npz file containing a list of tuples
         each of them representing a path to an image and it class
    :param device:
    :param batch_size:
    :return: torch.utils.DataLoader object constructed from UDACITY-DRIVING dataset object
    """

    dataset = ComplexDrivingData(file_path, device=device)
    if 'test' in file_path or 'validation' in file_path:
        iterator = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    else:
        iterator = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    return iterator