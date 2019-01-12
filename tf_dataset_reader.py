# https://www.tensorflow.org/guide/datasets
import glob

import matplotlib.pyplot as  plt
import numpy as np
import tensorflow as tf


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [300, 300])
    return image_resized


def main():
    path = 'data/pascal_voc_images/**/*.jpg'
    image_paths = []
    for file_path in glob.glob(path):
        image_paths.append(file_path)

    file_names = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(9)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        while True:
            img_list = sess.run(next_element)

            fig = plt.figure(figsize=(8, 8))
            for i, img in enumerate(img_list):
                sub_img = fig.add_subplot(331 + i)
                sub_img.imshow(img.astype(np.uint8))

            plt.show()

            ans = input("Continue? (Y/n)")
            if ans and ans[0].lower() == 'n':
                break


if __name__ == '__main__':
    main()
