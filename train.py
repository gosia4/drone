import tensorflow as tf
import numpy as np
import os
import shutil
import json
import matplotlib.pyplot as plt
from yolo_v3_model import yolo_v3
from dataset import Dataset
from utils import read_class_names, decode, loss_func

""" 
Main function to train YOLO_V3 model
"""

# Train options
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_CLASSES = "drone/thermographic_data/classes.txt"
TRAIN_NUM_OF_CLASSES = len(read_class_names(TRAIN_CLASSES))
TRAIN_MODEL_NAME = "model_2"
TRAIN_ANNOT_PATH = "drone/thermographic_data/train"
TRAIN_LOGDIR = "log" + '/' + TRAIN_MODEL_NAME
TRAIN_CHECKPOINTS_FOLDER = "checkpoints" + '/' + TRAIN_MODEL_NAME
TRAIN_BATCH_SIZE = 4
TRAIN_INPUT_SIZE = 416
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 1
TRAIN_EPOCHS = 10
TRAIN_DECAY = 0.8
TRAIN_DECAY_STEPS = 50.0

# TEST options
TEST_ANNOT_PATH = "drone/thermographic_data/validate"
TEST_BATCH_SIZE = 4
TEST_INPUT_SIZE = 416
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45

# YOLO options
YOLO_STRIDES = [8, 16, 32]
YOLO_IOU_LOSS_THRESH = 0.5
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_INPUT_SIZE = 416
YOLO_BATCH_FRAMES = 5
YOLO_PREPROCESS_IOU_THRESH = 0.3
YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]]


def main():
    """ main function """

    def train_step(image_data, target):

        """ function to apply gradients to train yolo_v3 model """

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:

            # obtain yolo_output from model
            yolo_output = yolo_v3_model(image_data)

            # intialise loss variables to zero
            giou_loss = conf_loss = prob_loss = 0

            # iterate over 3 scales
            for i in range(3):
                # decode resepctive yolo_output from each scale
                pred_result = decode(yolo_output=yolo_output[i], num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE,
                                     classes=TRAIN_NUM_OF_CLASSES, strides=YOLO_STRIDES, anchors=YOLO_ANCHORS,
                                     index=i)

                # compute loss with loss function
                loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
                                       YOLO_IOU_LOSS_THRESH)

                # update corresponding losses
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            # sum up losses
            total_loss = giou_loss + conf_loss + prob_loss

            # computes model gradient for all trainable variables using operations recorded in context of this tape
            gradients = tape.gradient(total_loss, yolo_v3_model.trainable_variables)

            # apply model gradients to all trainable variables
            optimizer.apply_gradients(zip(gradients, yolo_v3_model.trainable_variables))

            # increment global steps
            global_steps.assign_add(1)

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            if global_steps < warmup_steps:

                lr = global_steps / warmup_steps * TRAIN_LR_INIT

            else:

                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            # if global_steps.numpy() < 100:

            #     lr = TRAIN_LR_INIT

            # else:

            #     lr = TRAIN_LR_INIT * pow(TRAIN_DECAY, (global_steps.numpy() / TRAIN_DECAY_STEPS))

            # assign learning rate to optimizer
            optimizer.learning_rate.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.learning_rate, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()

        return global_steps.numpy(), optimizer.learning_rate.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def validate_step(image_data, target):

        """ function to return the losses for the model during validation step """

        # obtain yolo_output from model
        yolo_output = yolo_v3_model(image_data)

        # intialise loss variables to zero
        giou_loss = conf_loss = prob_loss = 0

        # iterate over 3 scales
        for i in range(3):
            # decode resepctive yolo_output from each scale
            pred_result = decode(yolo_output=yolo_output[i], num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE,
                                 classes=TRAIN_NUM_OF_CLASSES, strides=YOLO_STRIDES, anchors=YOLO_ANCHORS,
                                 index=i)

            # compute loss with loss function
            loss_items = loss_func(pred_result, yolo_output[i], *target[i], TRAIN_NUM_OF_CLASSES, YOLO_INPUT_SIZE,
                                   YOLO_IOU_LOSS_THRESH)

            # update corresponding losses
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        # sum up losses
        total_loss = giou_loss + conf_loss + prob_loss

        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # obtain and print list of gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')

    # if there is gpu available
    if len(gpus) > 0:

        try:

            # ensure that only necessary memory is allocated for gpu
            tf.config.experimental.set_memory_growth(gpus[0], True)

        except RuntimeError:

            pass

    # if log directory for tensorboard exist
    if os.path.exists(TRAIN_LOGDIR):
        # remove entire directory
        shutil.rmtree(TRAIN_LOGDIR)

    # creates a summary file writer training and validation for the given log directory
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    # instantiate train and test set
    trainset = Dataset(dataset_type='train', annot_path=TRAIN_ANNOT_PATH, batch_size=TRAIN_BATCH_SIZE,
                       train_input_size=TRAIN_INPUT_SIZE, strides=YOLO_STRIDES, classes_file_path=TRAIN_CLASSES,
                       anchors=YOLO_ANCHORS, anchor_per_scale=YOLO_ANCHOR_PER_SCALE,
                       max_bbox_per_scale=YOLO_MAX_BBOX_PER_SCALE, batch_frames=YOLO_BATCH_FRAMES,
                       iou_threshold=YOLO_PREPROCESS_IOU_THRESH)
    testset = Dataset(dataset_type='test', annot_path=TEST_ANNOT_PATH, batch_size=TEST_BATCH_SIZE,
                      train_input_size=TEST_INPUT_SIZE, strides=YOLO_STRIDES, classes_file_path=TRAIN_CLASSES,
                      anchors=YOLO_ANCHORS, anchor_per_scale=YOLO_ANCHOR_PER_SCALE,
                      max_bbox_per_scale=YOLO_MAX_BBOX_PER_SCALE, batch_frames=YOLO_BATCH_FRAMES,
                      iou_threshold=YOLO_PREPROCESS_IOU_THRESH)
    print(len(trainset))
    print(len(testset))
    # obtain the num of steps per epoch
    steps_per_epoch = len(trainset)

    # variable to track number of steps throughout training
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

    # steps during warmup stage of training
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch

    # training steps
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    # create the yolo_v3_model
    yolo_v3_model = yolo_v3(num_of_anchor_bbox=YOLO_ANCHOR_PER_SCALE, classes=TRAIN_NUM_OF_CLASSES,
                            checkpoint_dir=TRAIN_CHECKPOINTS_FOLDER, model_name=TRAIN_MODEL_NAME)

    # Zmienne do śledzenia strat i numeru epoki
    start_epoch = 0
    all_train_losses = []
    all_val_losses = []

    # train from last saved checkpoint if true
    if TRAIN_FROM_CHECKPOINT:
        if os.path.exists(yolo_v3_model.checkpoint_path + '.index'):
            yolo_v3_model.load_weights(yolo_v3_model.checkpoint_path).expect_partial()
            if os.path.exists(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'last_epoch.txt')):
                with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'last_epoch.txt'), 'r') as f:
                    start_epoch = int(f.read())
            if os.path.exists(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'train_losses.json')):
                with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'train_losses.json'), 'r') as f:
                    all_train_losses = json.load(f)
            if os.path.exists(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'val_losses.json')):
                with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'val_losses.json'), 'r') as f:
                    all_val_losses = json.load(f)

    # initialise default adam optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_LR_INIT)

    # initialise large best validation loss variable to track best_val_loss
    best_val_loss = np.inf

    # iterate over number of epochs
    for epoch in range(start_epoch, TRAIN_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{TRAIN_EPOCHS}")
        epoch_train_losses = []
        # iterate over image and target in trainset
        for image_data, target in trainset:
            # obtain metrics from train step for given image and target
            results = train_step(image_data, target)

            # update corresponding losses
            total_loss = results[5]
            epoch_train_losses.append(total_loss)

        # intialise losses for validation to zero
        epoch_val_losses = []

        # iterate over valdiation testset
        for image_data, target in testset:
            # obtain losses from validation set
            results = validate_step(image_data, target)

            # update corresponding losses
            total_loss = results[3]
            epoch_val_losses.append(total_loss)

        # calculate mean loss for the epoch
        train_epoch_loss = np.mean(epoch_train_losses)
        val_epoch_loss = np.mean(epoch_val_losses)

        # print relevant metrics
        print(f'epoch:{epoch + 1}, train_loss:{train_epoch_loss:.2f}, val_loss:{val_epoch_loss:.2f}')

        # save best validation if avg loss from current epoch is less than best known model
        if TRAIN_SAVE_BEST_ONLY:
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                yolo_v3_model.save_weights(yolo_v3_model.best_checkpoint_path)
        else:
            yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)

        # append losses
        all_train_losses.append(train_epoch_loss)
        all_val_losses.append(val_epoch_loss)

        # save current epoch number and losses to file
        with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'last_epoch.txt'), 'w') as f:
            f.write(str(epoch + 1))

        with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'train_losses.json'), 'w') as f:
            json.dump(all_train_losses, f)
        with open(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'val_losses.json'), 'w') as f:
            json.dump(all_val_losses, f)

        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", val_epoch_loss, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", results[0], step=epoch)
            tf.summary.scalar("validate_loss/conf_val", results[1], step=epoch)
            tf.summary.scalar("validate_loss/prob_val", results[2], step=epoch)
        validate_writer.flush()

        # print relevant data and metrics for validation
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(results[0], results[1], results[2], val_epoch_loss))

    # plot losses
    plt.figure()
    plt.plot(range(1, len(all_train_losses) + 1), all_train_losses, label='Train Loss')
    plt.plot(range(1, len(all_val_losses) + 1), all_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(TRAIN_CHECKPOINTS_FOLDER, 'loss_plot.png'))
    plt.show()


if __name__ == '__main__':
    main()
