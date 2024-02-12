import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image

# Define your flags using tf.compat.v1.flags
FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('checkpoint_dir',
                               './inception_finetuned_models/birds_valid299/model.ckpt-5000',
                               """Path where to read model checkpoints.""")

tf.compat.v1.flags.DEFINE_string('image_folder',
                               '/Users/han/Documents/CUB_200_2011/CUB_200_2011/images',
                               """Path where to load the images """)

tf.compat.v1.flags.DEFINE_integer('num_classes', 50,
                              """Number of classes """)
tf.compat.v1.flags.DEFINE_integer('splits', 10,
                              """Number of splits """)
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.compat.v1.flags.DEFINE_integer('gpu', 1, "The ID of GPU to use")

# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

fullpath = FLAGS.image_folder
print(fullpath)


def preprocess(img):
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = np.array(Image.fromarray(img).resize((299, 299), resample=Image.BILINEAR))
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return np.expand_dims(img, 0)


def get_inception_score(images, pred_op):
    splits = FLAGS.splits
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    bs = FLAGS.batch_size
    preds = []
    num_examples = len(images)
    n_batches = int(np.floor(float(num_examples) / float(bs)))
    indices = list(np.arange(num_examples))
    np.random.shuffle(indices)
    for i in range(n_batches):
        inp = []
        for j in range(bs):
            if (i * bs + j) == num_examples:
                break
            img = images[indices[i * bs + j]]
            img = preprocess(img)
            inp.append(img)
        inp = np.concatenate(inp, 0)
        pred = pred_op(inp)
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        istart = i * preds.shape[0] // splits
        iend = (i + 1) * preds.shape[0] // splits
        part = preds[istart:iend, :]
        kl = (part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
              )
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    print('mean:', "%.2f" % np.mean(scores), 'std:', "%.2f" % np.std(scores)
          )
    return np.mean(scores), np.std(scores)


def load_data(fullpath):
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.endswith('jpg') or name.endswith('png'):
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    img = np.array(Image.open(filename))
                    images.append(img)
    print('images', len(images), images[0].shape)
    return images


def inference(images, num_classes, for_training=False, restore_logits=True, scope=None):
    batch_norm_params = {
        'momentum': BATCHNORM_MOVING_AVERAGE_DECAY,  # 'decay' renamed to 'momentum'
        'epsilon': 0.001,
    }

    # Model Construction
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='valid', 
                      activation='relu', kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(0.00004))(inputs)
    x = layers.BatchNormalization(**batch_norm_params)(x)
    # ... (Add other Inception v3 blocks in a similar manner)

    # Final logits (assuming Inception v3 structure)
    x = layers.GlobalAveragePooling2D()(x)
    logits = layers.Dense(num_classes)(x)

    # Auxiliary logits (if applicable)
    if 'aux_logits' in endpoints:  # Adjust the key if necessary 
        auxiliary_logits = endpoints['aux_logits']
    else:
        auxiliary_logits = None

    return logits, auxiliary_logits


def main(unused_argv=None):
    # Evaluate the model on the dataset

    with tf.device("/GPU:" + str(FLAGS.gpu)):
        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = FLAGS.num_classes + 1

        # Build a Graph that computes the logits predictions from the
        # inference model.
        inputs = tf.keras.layers.Input(shape=(299, 299, 3))
        logits, _ = inference(inputs, num_classes)

        # Calculate softmax after removing the class for BG
        known_logits = tf.slice(logits, [0, 1], [FLAGS.batch_size, num_classes - 1])
        pred_op = tf.nn.softmax(known_logits)

        # Restore the model
        variables_to_restore = tf.train.list_variables(FLAGS.checkpoint_dir)
        variable_dict = {}
        for name, shape in variables_to_restore:
            variable_dict[name] = tf.Variable(initial_value=tf.train.load_variable(FLAGS.checkpoint_dir, name))
        tf.train.Checkpoint(**variable_dict).restore(FLAGS.checkpoint_dir)
        print('Restore the model from %s).' % FLAGS.checkpoint_dir)

        images = load_data(fullpath)
        get_inception_score(images, pred_op)


if __name__ == '__main__':
    tf.compat.v1.app.run()
