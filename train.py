import pickle
import time
import warnings

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from dan import DAN
from remtime import *

warnings.filterwarnings("ignore")

LEARNING_RATE = 0.0005
BATCH_SIZE = 25
N_EPOCHS = 2
REG_PENALTY = 0
PER = 0.2
NUM_IMAGES = 599900
NUM_TEST_IMAGES = 199900


imgs = tf.compat.v1.placeholder("float", [None, 224, 224, 3], name="image_placeholder")
values = tf.compat.v1.placeholder("float", [None, 5], name="value_placeholder")

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.compat.v1.Session(config=config) as sess:

    model = DAN(imgs, REG_PENALTY=REG_PENALTY, preprocess="vggface")
    output = model.output
    cost = tf.compat.v1.reduce_mean(tf.compat.v1.squared_difference(model.output, values)) + model.cost_reg
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    tr_reader = tf.compat.v1.TFRecordReader()
    tr_filename_queue = tf.compat.v1.train.string_input_producer(
        ["train_full.tfrecords"], num_epochs=2 * N_EPOCHS
    )
    _, tr_serialized_example = tr_reader.read(tr_filename_queue)
    # Decode the record read by the reader
    tr_feature = {
        "train/image": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "train/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    tr_features = tf.compat.v1.parse_single_example(tr_serialized_example, features=tr_feature)
    # Convert the image data from string back to the numbers
    tr_image = tf.compat.v1.decode_raw(tr_features["train/image"], tf.compat.v1.uint8)
    tr_label = tf.compat.v1.decode_raw(tr_features["train/label"], tf.compat.v1.float32)
    # Reshape image data into the original shape
    tr_image = tf.compat.v1.reshape(tr_image, [224, 224, 3])
    tr_label = tf.compat.v1.reshape(tr_label, [5])
    tr_images, tr_labels = tf.compat.v1.train.shuffle_batch(
        [tr_image, tr_label],
        batch_size=BATCH_SIZE,
        capacity=100,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )

    val_reader = tf.compat.v1.TFRecordReader()
    val_filename_queue = tf.compat.v1.train.string_input_producer(
        ["val_full.tfrecords"], num_epochs=N_EPOCHS
    )
    _, val_serialized_example = val_reader.read(val_filename_queue)
    # Decode the record read by the reader
    val_feature = {
        "val/image": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "val/label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
    }
    val_features = tf.compat.v1.parse_single_example(val_serialized_example, features=val_feature)
    # Convert the image data from string back to the numbers
    val_image = tf.compat.v1.decode_raw(val_features["val/image"], tf.compat.v1.uint8)
    val_label = tf.compat.v1.decode_raw(val_features["val/label"], tf.compat.v1.float32)
    # Reshape image data into the original shape
    val_image = tf.compat.v1.reshape(val_image, [224, 224, 3])
    val_label = tf.compat.v1.reshape(val_label, [5])
    val_images, val_labels = tf.compat.v1.train.shuffle_batch(
        [val_image, val_label],
        batch_size=BATCH_SIZE,
        capacity=100,
        min_after_dequeue=BATCH_SIZE,
        allow_smaller_final_batch=True,
    )

    init_op = tf.compat.v1.group(
        tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()
    )
    sess.run(init_op)

    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(coord=coord)

    model.initialize_with_vggface("vgg-face.mat", sess)
    loss_list = []
    param_num = 1
    for epoch in range(N_EPOCHS):
        tr_acc_list = []
        val_acc_list = []
        sess.run(tf.compat.v1.local_variables_initializer())
        i = 0
        error = 0
        stime = time.time()

        while i < NUM_IMAGES:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([tr_images, tr_labels])
            except:
                print(error, ": Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            _, c = sess.run(
                [optimizer, cost],
                feed_dict={imgs: epoch_x.astype(np.float32), values: epoch_y},
            )
            loss_list.append(np.power(c, 0.5))

            x = 100 / PER
            if not i % 2000:
                per = float(i) / NUM_IMAGES * 100
                print(
                    "Epoch:"
                    + str(round(per, 2))
                    + "% Of "
                    + str(epoch + 1)
                    + "/"
                    + str(N_EPOCHS)
                    + ", Batch loss:"
                    + str(round(c, 4))
                )
                ftime = time.time()
                remtime = (ftime - stime) * ((NUM_IMAGES - i) / (NUM_IMAGES / x))
                stime = ftime
                printTime(remtime)
            if not i % 20000:
                with open("param" + str(param_num) + ".pkl", "wb") as pfile:
                    pickle.dump(
                        sess.run(model.parameters), pfile, pickle.HIGHEST_PROTOCOL
                    )
                print(str(param_num) + " weights Saved!!")
                param_num += 1

        with open("param" + str(param_num) + ".pkl", "wb") as pfile:
            pickle.dump(sess.run(model.parameters), pfile, pickle.HIGHEST_PROTOCOL)
            print(str(param_num) + " weights Saved!!")
            param_num += 1

        sess.run(tf.compat.v1.local_variables_initializer())
        print("Computing Training Accuracy..")
        i = 0
        error = 0
        while i < NUM_IMAGES:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([tr_images, tr_labels])
            except:
                print("Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
            )
            tr_mean_acc = np.mean(1 - np.absolute(output - epoch_y))
            tr_acc_list.append(tr_mean_acc)

        tr_mean_acc = np.mean(tr_acc_list)

        print("Computing Validation Accuracy..")
        i = 0
        error = 0
        while i < NUM_TEST_IMAGES:
            i += BATCH_SIZE
            try:
                epoch_x, epoch_y = sess.run([val_images, val_labels])
            except:
                print("Error in reading this batch")
                error += 1
                if error > 10:
                    break
                continue
            output = sess.run(
                [model.output], feed_dict={imgs: epoch_x.astype(np.float32)}
            )
            val_mean_acc = np.mean(1 - np.absolute(output - epoch_y))
            val_acc_list.append(val_mean_acc)

        val_mean_acc = np.mean(val_acc_list)
        print("Epoch" + str(epoch + 1) + " completed out of " + str(N_EPOCHS))
        print("Tr. Mean Acc:" + str(round(tr_mean_acc, 4)))
        print("Val. Mean Acc:" + str(round(val_mean_acc, 4)))

    coord.request_stop()
    # Wait for threads to stop
    coord.join(threads)

    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "model_full")
    print("Session Saved!!")
    with open("loss_full.pkl", "wb") as pfile:
        pickle.dump(loss_list, pfile, pickle.HIGHEST_PROTOCOL)
    print("Loss List Saved!!")
