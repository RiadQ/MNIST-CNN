import numpy as np
import struct

def load_images(filepath):

    with open(filepath, 'rb') as f:
        magic_num, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic_num == 2051, f'Invalid magic num: {magic_num}'

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)

        return images


def load_labels(filepath):
    with open(filepath, 'rb') as f:
        magic_num, num_label = struct.unpack('>II', f.read(8))
        assert magic_num == 2049, f'Invalid magic num: {magic_num}'

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


def create_mini_batches(data, labels, size=128):
    samples = data.shape[0]

    batches = []
    for i in range(0, samples, size):
        data_batch = data[i:i+size]
        label_batch = labels[i:i+size]
        batches.append((data_batch, label_batch))
    
    return batches