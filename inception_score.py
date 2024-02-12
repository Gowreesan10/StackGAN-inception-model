import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image

# Define flags
FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_string('checkpoint_dir',
                                 './inception_finetuned_models/birds_valid299/model.ckpt-5000',
                                 """Path where to read model checkpoints.""")
tf.compat.v1.flags.DEFINE_string('image_folder',
                                 '/Users/han/Documents/CUB_200_2011/CUB_200_2011/images',
                                 """Path where to load the images """)
tf.compat.v1.flags.DEFINE_integer('num_classes', 50,      # 20 for flowers
                                   """Number of classes """)
tf.compat.v1.flags.DEFINE_integer('splits', 10,
                                   """Number of splits """)
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.compat.v1.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")

# Batch normalization constants
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999


def preprocess_image(img):
    img = img.resize((299, 299))
    img = np.array(img)
    img = preprocess_input(img)
    return img


def get_inception_score(images, pred_model):
    splits = FLAGS.splits
    assert(isinstance(images, list))
    assert(isinstance(images[0], np.ndarray))
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)

    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(math.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)

    for i in range(n_batches):
        inp = []
        for j in range(bs):
            if (i*bs + j) == num_examples:
                break
            img = images[indices[i*bs + j]]
            img = preprocess_image(img)
            inp.append(img)
        inp = np.concatenate(inp, 0)
        pred = pred_model.predict(inp)
        preds.append(pred)

    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0))))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores))
    return np.mean(scores), np.std(scores)


def load_images(fullpath):
    print(fullpath)
    images = []
    for root, _, files in os.walk(fullpath):
        for name in files:
            if name.endswith(('jpg', 'png')):
                filename = os.path.join(root, name)
                img = Image.open(filename)
                images.append(img)
    print('images', len(images), images[0].size)
    return images


def main(unused_argv=None):
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Load InceptionV3 model
    inception_model = InceptionV3(weights='imagenet', include_top=True)

    # Load images
    images = load_images(FLAGS.image_folder)

    # Preprocess images
    inception_input = [preprocess_image(img) for img in images]

    # Get Inception score
    inception_score = get_inception_score(inception_input, inception_model)
    print("Inception Score:", inception_score)


if __name__ == '__main__':
    tf.compat.v1.app.run()
