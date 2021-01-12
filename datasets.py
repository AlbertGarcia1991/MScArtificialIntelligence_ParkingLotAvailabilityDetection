########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                    INM705 - Garcia Plaza, Albert / Bohkary, Syed                                     #
#                                 Code hacked from https://github.com/ultralytics/yolov3                               #
########################################################################################################################

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from utils import xyxy2xywh

current_path = os.getcwd()  # get current working path

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']

class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, (self.img_size, self.img_size))[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


class LoadImagesAndLabels(Dataset):
    """
    Load the images and label files (.txt) from the specified train and test paths.
    """
    def __init__(self, path, img_size, batch_size=16):
        """
        Store the images and its labels file.
        Arguments of the function are:
            :param path: directory of the dataset indexer file (block1, block2, train or other user-specified).
            :param img_size: default YOLOv3 working image size (416x416).
            :param batch_size: number of images per batch.
        """
        # Read the lines of the dataset indexer file (each line is one image path).
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()]
            self.img_files = [current_path + x for x in self.img_files]
        self.n = len(self.img_files)  # number of images on the given dataset indexer file

        self.img_size = img_size  # store the img_size argument (from model.img_size) as an attribute of current class

        # batch indices list: elements with same value are included in the same batch (e.g. [1, 1, 1, 2, 2, 2, 3, 3, 3]
        #   contains three batches with 3 elements each batch.
        self.batch = np.floor(np.arange(self.n) / batch_size).astype(np.int)
        nb = self.batch[-1] + 1  # number of batches

        # Store labels file directories (each image has its own label file, with same name but file-format .txt instead
        #   of .jpg.
        self.label_files = [x.replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files]

        # Preload labels
        self.imgs = [None] * self.n
        self.labels = [None] * self.n

    def __len__(self):
        """
        Function to get the length of the loaded dataset
            :return: length of the given dataset.
        """
        return len(self.img_files)

    def __getitem__(self, index):
        """
        Function that gives the main information of each image to run the network.
        Arguments of the function are:
            :param index: index of the image to be retrieved (referred to the loaded dataset)
        And returns:
            :return img: Torch tensor containing the loaded image (index given), with shape
                torch.Size([3, img_size, img_size]), where 3 is the number of channels.
            :return labels_out: Torch tensor containing the image labels, with shape torch.Size([nl, 6]), where nl is
                the number of labels into the image and 6 stands for the six entries that defines each bounding box
                (detected on image, class, x position, y position, width, and height)
            :return img_path: the image directory as string (e.g. '/home/user/YOLO/data/train/current_image.jpg')
        """
        img_path = self.img_files[index]  # path to the image
        label_path = self.label_files[index]  # path to the image's labels file

        # Load image
        img, (h0, w0), (h, w) = load_image(self, index)  # get the image (as Torch tensor), original and new dimensions

        # Letterbox
        img, pad = letterbox(img, (w, h))

        # Load labels
        labels = []
        if os.path.isfile(label_path):
            x = self.labels[index]
            if x is None:  # labels not preloaded
                with open(label_path, 'r') as f:
                    # store the image's labels into a Numpy array.
                    x = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)

            # If there ir more than one label on the .txt labels files. In other words, if there is any object on the
            #   ground truth image to be detected.
            if x.size > 0:
                labels = x.copy()
                # Convert labels to absolute dimensions in xyxy format (top-left and bottom-right corners), and add the
                #   extra pad added to the squared image returned by the letterbox.
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # top-left x corner
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # top-left x corner
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + pad[0]  # bottom-right x corner
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + pad[1]  # bottom-right y corner

        nL = len(labels)  # number of labels
        labels_out = torch.zeros((nL, 6))  # initialise the Torch tensor where storing the image's labels (bboxes)
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5]) # convert xyxy to xywh

            # Normalize coordinates again to values compressed in the range [0, 1] (as were presented on the .txt
            #   original labels).
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

            # Store the .txt file content into the Torch tensor already created
            labels_out[:, 1:] = torch.from_numpy(labels)  # we have add an extra first column to store images indices

        # Convert from BGR (cv2 loading style) to RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)  # allocate in continuous memory spaces
        img = torch.from_numpy(img)  # store the image into Torch tensor

        return img, labels_out, img_path

    @staticmethod
    def collate_fn(batch):
        """
        When processing the Torch tensors containing the labels, we have to add the image index to build the targets
        on next steps. When the image is processed through the DataLoader, the collate_fn proceeding is performed.
        Arguments of the function are:
            :param batch: all triplets (image, labels, path) contained in the whole current batch loaded by the
                DataLoader.
        And returns:
            :return img: Torch tensor containing the image.
            :return label: Torch tensor containing the labels.
            :return path: image's path.
        """
        img, label, path= zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i

        # Post-process the tensors to has its original shape
        img = torch.stack(img, 0)
        label = torch.cat(label, 0)

        return img, label, path


def load_image(self, index):
    """
    Loads one image from the dataset (given the index) and re-size it to specified model.img_size.
    Arguments of the function are:
        :param index: index of the image to be re-sized (from all loaded list)
    And returns:
        :return img: the resize image as Torch Tensor.
        :return (h0, w0): tuple with image's original height and width respectively.
        :return (h, w): tuple with image's re-sized height and width respectively.
    """
    img = self.imgs[index]  # check if the image has been already re-sized

    if img is None:  # if is the first time that the current image is passed through this function
        img_path = self.img_files[index]  # image path
        img = cv2.imread(img_path)  # load the image in BGR (openCV loads images in BGR format rather than standard RGB)
        h0, w0 = img.shape[:2]  # original image's dimensions (for CNRPark 1000x750)

        r = self.img_size / max(h0, w0)  # ratio between the target size (416) and the original image max dimension
        if r < 1:  # always resize down
            interp = cv2.INTER_AREA  # LINEAR interpolation method
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            h, w = img.shape[:2]
        return img, (h0, w0), (h, w)  # img, hw_original, hw_resized

    else:  # if the image has been already passed through this function, transformations already done.
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def letterbox(img, img_size):
    """
    Given the input image (usually rectangular), make the image square-shaped adding extrapolation pixels at the image
    short side (at both sides) until has the same dimension than the large image's side.
    Arguments of the function are:
        :param img: original image (after being converted to max. side length 416 -or custom YOLO working size) pixels.
        :param img_size: original image, also after being converted, dimensions (following format (width, height)
    And the outpust:
        :return img: image reshaped until be squared.
        :return (dw, dh)): horizontal (width) and vertical (height) padding added to make the image squared.
    """
    # Compute total padding (width and height padding)
    new_unpad = int(round(img_size[0])), int(round(img_size[1]))
    dw, dh = 416 - new_unpad[0], 416 - new_unpad[1]  # wh padding

    # Divide padding into 2 sides
    dw /= 2
    dh /= 2

    # Compute the top, bottom, left and right padding ad apply it to the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add to the original image the necessary padding at each side to make the image squared.
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    return img, (dw, dh)
