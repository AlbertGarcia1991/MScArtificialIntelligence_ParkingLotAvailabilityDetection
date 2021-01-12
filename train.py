########################################################################################################################
#                               MSc Artificial Intelligence - City, University of London                               #
#                                    INM705 - Garcia Plaza, Albert / Bohkary, Syed                                     #
########################################################################################################################

import os
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn

from test import *
from models import *
from datasets import *
from utils import *


def train(
        data=1,
        weights='weights/yolov3-tiny.conv.15',
        epochs=300,
        batch_size=16,
        accumulate=4,
        optimizer_method='sgd',
        initial_lr=0.01,
        optim_param=0,
        weight_decay=0,
        conf_thres=0.001,
        iou_thres=0.25,
        iou_method=None):
    """
    Train the network doing the whole process of loading the config file, creating the Darknet network and loading the
    weights, setting hyper-parameters, training during the specified number of epochs and, finally, test at each epoch
    using the 'test' dataset and store the results of different metrics. Arguments of the function are:
        :param data: if '1' train with block1 dataset, if '2' train with block2. Otherwise, specify the name of the
            .data file which contains information about the training and testing dataset indexers files to compute
            (paths, number of classes and .names file path (following the regular YOLOv3 .data format).
        :param weights: path of the network's weights file (can be either Darknet or PyTorch format).
        :param epochs: number of epochs to train the network.
        :param batch_size: number of images to load each training iteration.
        :param accumulate: number of batches to accumulate after each weight updating.
        :param optimizer_method: backpropagation optimizer algorithm; choose between sgd, adaprop, rmsprop, adadelta,
            adam, or amsgrad.
        :param initial_lr: learning rate to apply at the selected optimizer algorithm.
        :param optim_param: when the optimizer algorithm requires other parameter(s), pass using this argument.
        :param weight_decay: l2 regularization parameter.
        :param conf_thres: objectness score threshold to decide whether or not the object is detected.
        :param iou_thres: threshold used to check if more than one bounding boxes are pointing the same object.
        :param iou_method: selection of the IoU metric; choose between None (regular IoU), 'd' (DIoU), 'c' (CIoU), or
            'g' (GIoU).
    """
    # Select the .data path depending on the 'data' argument
    if data == 1:
        data_path = 'data/block1.data'
    elif data == 2:
        data_path = 'data/block2.data'
    else:
        data_path = data

    # Load parameters from the .data file (number of classes, train and test indexes paths, and .names file path)
    data_dict = parse_data_cfg(data_path)  # load the information inside .data file into a dictionary
    train_path = data_dict['train']  # train dataset indexer file
    test_path = data_dict['test']  # test dataset indexer file
    n_classes = int(data_dict['classes'])  # number of classes

    # Set internal performance variables
    torch.manual_seed(0)  # ensure that random generated parameters are always the same to compare results
    cudnn.deterministic = True  # CuDNN will only use deterministic algorithms (given the same inputs, same output)
    cudnn.benchmark = False  # CuDNN will always use the same algorithms (if True, CuDNN will choose best algorithm)

    # Set initial
    start_epoch = 0
    best_fitness = 0.0

    # Initialize Darknet network
    if n_classes == 1:
        cfg = 'cfg/YOLOv3Tiny_SC.cfg'  # load single-class network architecture
    else:
        cfg = 'cfg/YOLOv3Tiny.cfg'  # load multi-class network architecture
    model = Darknet(cfg).to('cuda')  # load the Darknet model following the modules specified on the .cfg file

    # Set the optimizer algorithm and its parameters
    pg0 = []
    for k, v in dict(model.named_parameters()).items():
        pg0 += [v]

    if optimizer_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=optim_param, weight_decay=weight_decay)
    elif optimizer_method == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_method == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=initial_lr, momentum=optim_param, weight_decay=weight_decay)
    elif optimizer_method == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    elif optimizer_method == 'amsgrad':
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay, amsgrad=True)
    else:
        raise ValueError('The optimizer algorithm passed as argument has not been implemented.')

    # Load weights files (Pytorch or Darknet formats)
    if weights.endswith('.pt'):  # if the given weights file has the PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cuda')['model'], strict=False)
        # When PyTorch's weights are loaded, continue from the last stored epoch
        start_epoch = torch.load(weights, map_location='cuda')['epoch'] + 1
    elif len(weights) > 0:  # if the given weights file has the Darknet format
        load_darknet_weights(model, weights)

    # Load of train and test datasets, and the batch size.
    test_dataset = LoadImagesAndLabels(test_path, model.img_size, batch_size)
    train_dataset = LoadImagesAndLabels(train_path, model.img_size, batch_size)
    batch_size = min(batch_size, len(train_dataset))

    # Set up the number of workers, and train and test DataLoaders.
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              num_workers=nw,
                                              shuffle=True,
                                              pin_memory=True,
                                              collate_fn=train_dataset.collate_fn)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=test_dataset.collate_fn)

    # Model parameters
    model.nc = n_classes  # attach number of classes to model attributes

    # Start training
    nb = len(trainloader)  # number of batches

    maps = np.zeros(n_classes)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'

    # epoch start ------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()  # set the model's training flag to True (in our model, this enables batch normalization)

        # Losses initialised with zeros: mloss = [loss of bounding boxes (this can be split in 4 individual losses:
        #   bounding box x and y position, and bounding box width and height), loss objectness, loss classes, and total
        #   loss (is the sum of the previous independent 3 losses)].
        mloss = torch.zeros(4).to('cuda')

        # Message prompt on Python console showing the main parameters results at each epoch. It shows the current
        #   epoch and the individuals and total (summed up) losses.
        print(('\n' + '%10s' * 5) % ('Epoch', 'Bbox', 'Object', 'Classes', 'Total'))
        pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar

        # batch start --------------------------------------------------------------------------------------------------
        # Pbar contains the current batch images, ground truth bounding boxes and paths to image and labels files.
        #   -i: for loop index
        #   -imgs: Torch tensor [N, 3, img_size, img_size] where N is the number of images per batch, 3 are the three
        #       channels of each image (RGB), and img_size is the working YOLOv3 image (after re-sizing and letterbox)
        #   -targets: Torch tensor [N*Li, 6] where N is the number os batches and Li is the number of labels at each
        #       image, 6 are the six columns of the labels format (first is the image index in the batch -from 0 to N-1,
        #       and the other five are the class, bounding box x, y positions, and bounding box width and height.
        #   -paths: array with the path to all images in the batch (array with N length).
        for i, (imgs, targets, paths) in pbar:
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to('cuda').float() / 255.0  # normalize the images between 0 and 1
            targets = targets.to('cuda')  # move targets to CUDA processing

            # Run model and obtain the predictions on the batched images. The output of model(imgs) is an array with
            #   three entries with the following shape [[N, 3, Ny1, Nx1, 7], [N, 3, Ny2, Nx2, 7], [N, 3, Ny3, Nx3, 7]],
            #   where:
            #   -N: the batch size.
            #   -3 stands for the number of anchors (3 anchors at each YOLO layer/gridpoint)
            #   -Nxi and Nyi: gridpoints where the object detections is performed (first layer makes a 13x13 grid,
            #       second layer makes 26x26 and third 52x52).
            #   -7 stands for the parameters determining each bounding box.
            pred = model(imgs)  # run the inference process and store predicted (pred) the bounding boxes

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model, iou_thres, iou_method)

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            # Compute gradient
            loss.backward()  # performing the backward pass to compute all the gradients using the calculated loss

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()  # performing the weights updating using the computed loss through the backward pass
                optimizer.zero_grad()  # set to zero the current stored gradients (to avoid accumulation on next step)

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            s = ('%10s' * 1 + '%10.3g' * 4) % ('%g/%g' % (epoch, epochs - 1), *mloss)
            pbar.set_description(s)  # show losses on the displayed progress bar

        # end batch ----------------------------------------------------------------------------------------------------

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        results, maps = test(cfg,
                             data_path,
                             model=model,
                             dataloader=testloader,
                             batch_size=batch_size,
                             img_size=model.img_size,
                             conf_thres=conf_thres,
                             iou_thres=iou_thres,
                             iou_method=iou_method)

        # Create a new folder to store the current training session results
        results_folder_name = str(iou_method)
        results_path = current_path + "/results/" + results_folder_name
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        else:
            pass

        # Write epoch results
        with open(results_path + '/results.txt', 'a+') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))
        # If current test results better than the best one stored until now, set current iteration as best one.
        if fi > best_fitness:
            best_fitness = fi

        # Save training results (file showing results and Pytorch weights)
        if final_epoch:
            # Write the results file with all epochs info (loss, IoU, ...)
            with open(results_path + '/results.txt', 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.state_dict() if hasattr(model, 'module') else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}
            torch.save(chkpt, results_path + '/last.pt')  # save last checkpoint
            if best_fitness == fi:  # if current iteration has best results until now, save current weights
                torch.save(chkpt, results_path + '/best.pt')
            del chkpt  # delete current checkpoint

    # end epoch --------------------------------------------------------------------------------------------------------
    torch.cuda.empty_cache()
