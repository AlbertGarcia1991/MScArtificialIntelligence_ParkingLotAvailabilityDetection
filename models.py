########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                    INM705 - Garcia Plaza, Albert / Bohkary, Syed                                     #
#                                 Code hacked from https://github.com/ultralytics/yolov3                               #
########################################################################################################################

import torch.nn.functional as F

from utils import *


def create_modules(module_defs, img_size):
    """
    Creation of each functional module as PyTorch's nn.Sequential module definitions.
    Arguments of the function are:
        :param module_defs: list of dictionaries containing the module type and parameters each.
        :param img_size: Square length size of images to be processed by the network.
    The output is:
        :return module_list: list of functional modules (functional module means that after giving an input the layer
            returns its output).
        :return routs_binary: boolean list where all are False entries except on residual layers (these will fed
            other layers downstream).
    """
    hyperparams = module_defs.pop(0)  # popt out the first block of the .cfg which contains UNUSED hyperparameters
    output_filters = [3]  # number of channels of the images (3 = RGB)

    # Initialise the listing of modules (this will contain just module title and its parameters -not functional layer).
    module_list = nn.ModuleList()

    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1  # variable that will store the number of YOLO layers already created

    # For each layer stored in module_defs array (each layer stored as a dictionary).
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()  # start the PyTorch's nn.Sequential() functionality (and module creation)

        if mdef['type'] == 'convolutional':  # if layer is convolutional (standard convolutional operations)
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            size = mdef['size']
            stride = mdef['stride']
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=size,
                                                   stride=stride,
                                                   padding=(size - 1) // 2,
                                                   bias=not bn))  # layers without batch normalize have bias
            if bn:  # if convolutional layer has batch normalization
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.003, eps=1E-4))
            # If does not have bn, the layers will be used as residual block (will be summed up in a downstream layer)
            else:
                routs.append(i)  # append the layer position (index) to be fed when the residual is called

            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))

        elif mdef['type'] == 'maxpool':  # if layer is maxpooling (standard maxpool operations)
            size = mdef['size']
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=(size - 1) // 2)  # maxpooling operation

            if size == 2 and stride == 1:  # only one layer located just after upsampling process starts
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool  # 5 maxpool layers along the downsampling process

        elif mdef['type'] == 'upsample':  # if layer is upsample (standard upsampling operations)
            modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # if layer is upsample (residual sum of previous layer(s) output(s))
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])

        elif mdef['type'] == 'yolo':  # if layer is YOLO
            yolo_index += 1  # update the number of YOLO layers already created

            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # each YOLO layer will have different anchors
                                nc=mdef['classes'],
                                img_size=img_size,
                                yolo_index=yolo_index)

        module_list.append(modules)
        output_filters.append(filters)  # store the number of output channels of the current defined layer

    # Set the boolean routs_binary list and switch to True the right indices
    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True

    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
        YOLO layers are created a an object, with its initialised parameters and forward behaviour as specific method.
    """
    def __init__(self, anchors, nc, img_size, yolo_index):
        """
        When a Darknet object is initialised, its basic properties are created.
        Arguments of the function are:
            :param anchors: list of pre-defined anchors.
            :param nc: number of training classes.
            :param img_size: size that images will have when performing the inference process.
            :param yolo_index: index of the YOLO layer created (1, 2 or 3)
        """
        super(YOLOLayer, self).__init__()  # Inherit object attributes from parent class Torch's nn.Module

        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.na = len(anchors)  # number of anchors
        self.nc = nc  # number of classes
        self.nx = 0  # initialize number of gridpoints along x image's direction to zero
        self.ny = 0  # initialize number of gridpoints along y image's direction to zero

        # Number of outputs (no) is [bbox x position score, bbox y position score, bbox width score, bbox height score,
        #   objectness score -detect whether or not there is an object-, and class score -probability that the detected
        #   object belongs to each class].
        self.no = nc + 5

    def forward(self, p, img_size, out):
        bs, _, ny, nx = p.shape  # store the batch size, and gridpoints along y and x direction respectively.

        # If the current number of gridpoints along both image's directions is not the input number of gridpoints,
        #   generate them through create_grid() function.
        if (self.nx, self.ny) != (nx, ny):
            create_grids(self, img_size, (nx, ny))  # gridpoints are stored as class attributes

        # Re-shape the predictions (p) vector from input shape [N, na*no, ny, nx] to working shape [N, na, ny, nx, no],
        #   where N is the number of images per bath, na is the number of applied anchors (3 at each YOLO layer), ny and
        #   nx are the number of gridpoints (first YOLO layer has 13x13, second 26x26, and last 52x52), and no is the
        #   number of parameters defining each bounding box (no = number of classes, x position, y position, width and
        #   height).
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:  # when training, return the predicted bounding boxes stored in 'p'
            return p

        # When testing, return the same output than before (p), and also the same values passed along a sigmoid and
        #   re-sized to image's original dimensions (io).
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy  # pass along sigmoid function and relocate
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # re-size to original image's dimensions
            io[..., :4] *= self.stride
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    Networks with Darknet architecture objects
    """

    def __init__(self, cfg):
        """
        When a Darknet object is initialised, its basic properties are created.
        Arguments of the function are:
            :param cfg: path to the .cfg file.
        """
        super(Darknet, self).__init__()  # Inherit object attributes from parent class Torch's nn.Module

        # YOLOv3 resizes input images to the specified (img_size, img_size) dimensions in order to perform the inference
        self.img_size = 416  # default YOLOv3 max size

        self.module_defs = parse_model_cfg(cfg)  # load the list of modules and its parameters as object attribute

        # Create the behaviour of each layer (module_list) and the residual pipelines (routs)
        self.module_list, self.routs = create_modules(self.module_defs, self.img_size)

        # Get the position (index) of each YOLO layer
        self.yolo_layers = [index for index, layer in enumerate(self.module_defs) if layer['type'] == 'yolo']

    def forward(self, x):
        """
        Setting of the global forward pass across the whole Darknet network.
        Arguments of the function are:
            :param x: the images batched with 3 channels and size (img_size, img_size). Torch tensor with shape
                [N, 3, img_size, img_size], where N is the number of images per batch.
        """
        img_size = x.shape[-2:]
        yolo_out, out = [], []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']

            # Perform the conv, upsample or maxpool functionality, as defined in Pytorch's modules with our parameters.
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)

            # Perform the route functionality, where the output of previous layers, specified in mdef['layers] are
            #   concatenated to the current output tensor (residual block behaviour).
            elif mtype == 'route':  # concat
                layers = mdef['layers']
                if len(layers) == 1:
                    x = out[layers[0]]
                else:
                    try:
                        x = torch.cat([out[i] for i in layers], 1)
                    except:  # apply stride 2 for darknet reorganization layer
                        #out[layers[1]] = F.interpolate(out[layers[1]], scale_factor=[0.5, 0.5])
                        new_shape = out[layers[0]].shape[-2:]
                        out[layers[1]] = F.interpolate(out[layers[1]], size=new_shape)
                        x = torch.cat([out[i] for i in layers], 1)

            # Perform the YOLO functionality, specified on the forward method belonging to the YOLOLayer object. The
            #   first time that this layer is evaluated, the input is a Torch tensor with shape [N, 21, 13, 13] where
            #   N is the number of images per batch again, 21 are the number of channels and 13x13 is the current number
            #   of gridpoints along y and x direction after the whole process of downsampling. The second time that
            #   this layer is evaluated the inputs has now size of [N, 21, 26, 26], the same number of channels but now
            #   the number of gridpoints is bigger as the network starts to upsample after the first YOLO layer is
            #   reached. Finally, the third YOLO layer has input of [N, 21, 52, 52], again the only change is the number
            #   of gridpoints following the upsampling process (network's last layer!).
            elif mtype == 'yolo':
                yolo_out.append(module(x, img_size, out))

            out.append(x if self.routs[i] else [])  # store output values to be concatenated on route layers

        if self.training:  # when training, output only the results obtained at YOLO layers
            return yolo_out

        else:  # when testing, returns predicted bounding boxes and re-sized inferred objects
            io, p = zip(*yolo_out)  # inference output, training output
            io = torch.cat(io, 1)
            return io, p


def create_grids(self, img_size, ng):
    """
    Create the grid along the image where predict bounding boxes.
    Arguments of the function are:
        :param img_size: image working size (typically 416x416).
        :param ng: number of gridpoints along both axis.
    """
    nx, ny = ng  # x and y grid size
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)  # separation (in pixels) between two consecutive grid points

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    # Creation of the indices for all gridpoints in the image
    self.grid_xy = torch.stack((xv, yv), 2).to('cuda').type(torch.float32).view((1, 1, ny, nx, 2))

    # Store grid parameters as class attributes
    # Re-size anchors depending on working image size obtaining Torch tensor with shape [1, 3, 1, 1, 2]
    self.anchor_vec = self.anchors.to('cuda') / self.stride
    # Re-shape self.anchor_vec to Torch tensor shape [3, 2]
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(torch.float32)
    self.ng = torch.Tensor(ng).to('cuda')
    self.nx = nx
    self.ny = ny


def load_darknet_weights(self, weights):
    """
    Load the pre-trained weights from a Darknet's format given file path
    Arguments of the function are:
        :param weights: file path of the Darknet's weights.
    """

    # Establish cutoffs (load layers between 0 and cutoff), where on YOLOv3 tiny is at 15th layer
    cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    # "Move" all stored weights to the PyTorch networks already created
    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':  # only convolutional layers have weights
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw
