import tensorflow as tf
import numpy as np
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from yolo_v3_model import yolo_v3
from dataset import Dataset
from utils import read_class_names, decode, loss_func

""" 
Main function to train YOLO_V3 model
"""

# Train options
TRAIN_SAVE_BEST_ONLY = False  # saves only best model according validation loss (True recommended)
TRAIN_CLASSES = "thermographic_data/classes.txt"
TRAIN_NUM_OF_CLASSES = len(read_class_names(TRAIN_CLASSES))
TRAIN_MODEL_NAME = "model_2"
TRAIN_ANNOT_PATH = "thermographic_data/train"
TRAIN_LOGDIR = "log" + '/' + TRAIN_MODEL_NAME
TRAIN_CHECKPOINTS_FOLDER = "checkpoints" + '/' + TRAIN_MODEL_NAME
TRAIN_BATCH_SIZE = 4
TRAIN_INPUT_SIZE = 416
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 3
TRAIN_EPOCHS = 30
TRAIN_DECAY = 0.8
TRAIN_DECAY_STEPS = 50.0

# TEST options
TEST_ANNOT_PATH = "thermographic_data/validate"
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

images_folder = r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/train/images/free_1"
labels_folder = r"C:/Users/gosia/drone-detection/drone-detection/thermographic_data/train/labels/free_1"


def load_bounding_box_points_from_label_file(label_file):
    points = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                # Znormalizowane wartości cx, cy, width, height
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tl = (cx - w / 2, cy - h / 2)
                tr = (cx + w / 2, cy - h / 2)
                bl = (cx - w / 2, cy + h / 2)
                br = (cx + w / 2, cy + h / 2)
                points.extend([tl, tr, bl, br])
    return points


def display_images_with_crosses(images_folder, labels_folder):
    for image_filename in os.listdir(images_folder):
        if image_filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, image_filename)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_filename)

            image = cv2.imread(image_path)
            if image is None:
                continue

            # Wczytaj współrzędne punktów z pliku etykiet
            points = load_bounding_box_points_from_label_file(label_path)

            # Przekształć znormalizowane współrzędne na współrzędne piksela
            height, width, _ = image.shape
            pixel_points = [(int(x * width), int(y * height)) for x, y in points]

            for point in pixel_points:
                cv2.drawMarker(image, point, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            cv2.imshow(image_filename, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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
            if global_steps < warmup_steps:

                lr = global_steps / warmup_steps * TRAIN_LR_INIT

            else:

                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))

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

    # train from last saved checkpoint if true
    if TRAIN_FROM_CHECKPOINT:
        # load weights of last saved checkpoint
        yolo_v3_model.load_weights(yolo_v3_model.checkpoint_path).expect_partial()

    # initialise default adam optimise
    optimizer = tf.keras.optimizers.Adam(learning_rate=TRAIN_LR_INIT)

    # initialise large best validation loss varaible to track best_val_loss
    best_val_loss = np.inf

    batch_train_losses, batch_val_losses = [], []
    train_losses, val_losses = [], []
    # display_images_with_crosses(images_folder, labels_folder)

    # iterate over the number of epochs
    for epoch in range(TRAIN_EPOCHS):

        # initiate variable to store total training loss
        total_loss = 0.0

        # iterate over the number of batches
        for image_data, target in trainset:
            results = train_step(image_data, target)
            batch_train_losses.append(results[-1])

        train_losses.append(np.mean(batch_train_losses))

        # print results per epoch
        print(
            f"Epoch: {epoch + 1:2.0f}, step: {results[0]:5.0f}, lr: {results[1]:.6f}, giou_loss: {results[2]:7.2f}, conf_loss: {results[3]:7.2f}, prob_loss: {results[4]:7.2f}, total_loss: {results[5]:7.2f}")

        # add validation loss to tensorboard every epoch
        if len(testset) > 0:

            # initiate variable to store total validation loss
            total_val_loss = 0.0

            # iterate over validation data
            for image_data, target in testset:
                results = validate_step(image_data, target)
                batch_val_losses.append(results[-1])

            val_losses.append(np.mean(batch_val_losses))

            # compute average validation loss over epoch
            ave_val_loss = total_val_loss / len(testset)

            with validate_writer.as_default():
                tf.summary.scalar("Validate_loss/total_val_loss", ave_val_loss, step=epoch)

            validate_writer.flush()

            # save model for best validation loss
            if TRAIN_SAVE_BEST_ONLY and best_val_loss > ave_val_loss:

                # update best validation loss
                best_val_loss = ave_val_loss

                # save best validation loss model
                yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)
                print(f"\nModel weights saved at epoch {epoch + 1}")
            else:
                epoch_checkpoint_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, f"epoch_{epoch + 1}.weights.h5")
                yolo_v3_model.save_weights(epoch_checkpoint_path)
                print(f"\nModel weights saved at epoch {epoch + 1}")

        else:
            epoch_checkpoint_path = os.path.join(TRAIN_CHECKPOINTS_FOLDER, f"epoch_{epoch + 1}.weights.h5")
            yolo_v3_model.save_weights(epoch_checkpoint_path)
            print(f"\nModel weights saved at epoch {epoch + 1}")
            # save model
            # yolo_v3_model.save_weights(yolo_v3_model.checkpoint_path)
            # print(f"\nModel weights saved at epoch {epoch+1}")

    # plot the training and validation loss over epochs
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # save the plot to a file
    plt.savefig('training_validation_loss.png')


if __name__ == '__main__':
    main()
