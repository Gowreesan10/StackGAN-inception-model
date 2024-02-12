import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
import scipy.misc
import os

# Model and configuration
FLAGS = {
    'checkpoint_dir': '/content/StackGAN-inception-model/inception_finetuned_models/birds_valid299/model.ckpt',
    'image_folder': '/Users/han/Documents/CUB_200_2011/CUB_200_2011/images',
    'num_classes': 50,  # 20 for flowers
    'splits': 10,
    'batch_size': 64,
    'gpu': 1  # Let's assume you want to use GPU 1
}

BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997
MOVING_AVERAGE_DECAY = 0.9999


def preprocess(img):
    """Preprocesses a single image for Inception v3.

    Args:
        img: Numpy array of the image (H, W, 3).

    Returns:
        Preprocessed image as a Numpy array.
    """
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)  # Handle grayscale
    img = scipy.misc.imresize(img, (299, 299, 3))
    img = preprocess_input(img)  # InceptionV3 preprocessing
    return img


def load_data(fullpath):
    """Loads images from a directory.

    Args:
        fullpath: Path to the image directory.

    Returns:
        List of loaded images as Numpy arrays.
    """
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.lower().endswith('.jpg') or name.lower().endswith('.png'):
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
    return images


def build_model():
    """Builds the Inception v3 model with custom output.

    Returns:
         Keras Model with the Inception v3 architecture, loaded from
         checkpoint if specified.
    """
    current_directory = os.getcwd()
    print("Current directory:", current_directory)
    base_model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3)
    )

    # Extract the mixed_10 layer output
    x = base_model.get_layer('mixed10').output

    # Add your custom top layers: Adjust this if your fine-tuned model is different
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(FLAGS['num_classes'], activation='softmax')(x)

    # Create a new model with the desired output from 'x'
    model = Model(inputs=base_model.input, outputs=x)

    # Load weights from checkpoint
    if FLAGS['checkpoint_dir']:
        model.load_weights(FLAGS['checkpoint_dir'])

    return model


def calculate_inception_score(images, model):
    """Calculates the Inception Score for a list of images."""
    splits = FLAGS['splits']
    batch_size = FLAGS['batch_size']

    preds = []
    num_examples = len(images)
    n_batches = int(np.ceil(float(num_examples) / float(batch_size)))

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_examples)
        batch_images = images[start_idx: end_idx]

        # Preprocess batch
        batch_images = [preprocess(img) for img in batch_images]
        batch_images = np.stack(batch_images, axis=0)

        # Get model predictions
        batch_preds = model.predict(batch_images)
        preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)

    # Inception Score calculation 
    scores = []
    for i in range(splits):
        part = preds[i * (preds.shape[0] // splits): (i + 1) * (preds.shape[0] // splits)]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


# Main execution
if __name__ == '__main__':
    name = input("1: ")
    images = load_data(FLAGS['image_folder'])
    name = input("2: ")
    model = build_model()
    
    name = input("3: ")
    mean_score, std_score = calculate_inception_score(images, model)
    print('Inception Score: mean:', mean_score, 'std:', std_score)
