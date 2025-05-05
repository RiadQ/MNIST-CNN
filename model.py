import numpy as np
from dataset import load_images, load_labels

np.set_printoptions(suppress=True)

train_images = load_images('train-images-idx3-ubyte') / 255.0
train_labels = load_labels('train-labels-idx1-ubyte')

test_images = load_images('t10k-images-idx3-ubyte') / 255.0
test_labels = load_labels('t10k-labels-idx1-ubyte')


def full_conv(grad, kernel):
    kernel = np.rot90(kernel, 2)
    grad_h, grad_w = grad.shape
    k_h, k_w = kernel.shape

    pad_h = k_h - 1
    pad_w = k_w - 1

    grad = np.pad(grad, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    output_h, output_w = (grad_h + k_h - 1, grad_w + k_w - 1)
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = grad[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    
    return output


def valid_conv(signal, grad):

    in_h, in_w = signal.shape
    grad_h, grad_w = grad.shape
    kernel_h = in_h - grad_h +1
    kernel_w = in_w - grad_w +1

    output = np.zeros((kernel_h, kernel_w))

    for i in range(kernel_h):
        for j in range(kernel_w):
            output[i, j] = np.sum(signal[i:i+grad_h, j:j+grad_w] * grad)

    return output


def relu(x):
    return np.maximum(x, 0.01 * x)


def softmax(x):
    shifted_x = x - np.max(x)
    exps = np.exp(shifted_x)
    return exps / np.sum(exps)


def deriv_relu(x):
    return np.where(x>0, 1, 0.01)


def cross_entropy_loss(p, q, epsilon=1e-15):
    q = np.clip(q, epsilon, 1-epsilon)
    return -np.sum(p*np.log(q), axis=0)


def clip_grad(grad, max_norm=50):
    return np.clip(grad, -max_norm, max_norm)


def run_test_cases(data, labels, model):
    test_case = 1
    correct = 0
    for i, image in enumerate(data):
        out = model.feedforward(image)
        pred = np.argmax(out)
        true = labels[i]
        print(f'Case {test_case} True Value: {true} Netwrok Output: {pred}')
        test_case += 1

        if pred == true:
            correct += 1

    print(f'Model accuracy: {round((correct/len(data))*100, 2)}%')


class ConvLayer:
    def __init__(self):
        self.kernel = np.random.randn(3, 3) * np.sqrt(2/(3*3))
        self.b = 0

    def convole2d(self, image):

        image_height, image_witdh = image.shape
        kernel_height, kernel_width = self.kernel.shape

        output_height = image_height - kernel_height + 1
        output_width = image_witdh - kernel_width + 1

        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * self.kernel) + self.b

        mask = deriv_relu(output)
        output = relu(output)

        return output, mask
        

class PoolLayer:
    def max_pool(self, conv_out):
        
        input_height, input_width = conv_out.shape
        kernel_height, kernel_width = (2, 2)

        pad_h = 0 if input_height % 2 == 0 else 1
        pad_w = 0 if input_width % 2 == 0 else 1

        if pad_h or pad_w:
            conv_out = np.pad(conv_out, ((0, pad_h), (0, pad_w)), mode='constant')

        output_height = conv_out.shape[0] // 2
        output_width = conv_out.shape[1] // 2

        output = np.zeros((output_height, output_width))

        max_row_indices = np.zeros((output_height, output_width), dtype=int)  
        max_col_indices = np.zeros((output_height, output_width), dtype=int)

        for i in range(output_height):
            for j in range(output_width):
                input_region = conv_out[
                    i*kernel_height : (i+1)*kernel_height,
                    j*kernel_width : (j+1)*kernel_width
                ]

                output[i, j] = np.max(input_region)

                output[i, j] = np.max(input_region)

                flat_argmax = np.argmax(input_region) 
                row_in_window = flat_argmax // kernel_width
                col_in_window = flat_argmax % kernel_width

                max_row_indices[i, j] = i * kernel_height + row_in_window
                max_col_indices[i, j] = j * kernel_width + col_in_window

        return output, (max_row_indices, max_col_indices)


class HiddenLayer:
    def __init__(self, neurons, input_dim):
        self.W = np.random.randn(neurons, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(neurons)

    def compute(self, x):
        return relu(np.dot(self.W, x) + self.b)


class OutputLayer:
    def __init__(self, input_dim):
        self.W = np.random.randn(10, input_dim) * np.sqrt(1 / input_dim)
        self.b = np.zeros(10)

    def compute(self, x):
        return softmax(np.dot(self.W, x) + self.b)


class NeuralNetwork:
    def __init__(self, data, labels, epochs):
        self.c1 = ConvLayer()
        self.c2 = ConvLayer()
        self.l1 = PoolLayer()
        self.l2 = PoolLayer()
        self.h1 = HiddenLayer(20, 36)
        self.o1 = OutputLayer(20)

        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.learn_rate = 0.01
        self.lambda_reg = 0.001

    def feedforward(self, x):
        self.c1_out, self.c1_mask = self.c1.convole2d(x)
        self.l1_out, self.max_l1 = self.l1.max_pool(self.c1_out)
        self.c2_out, self.c2_mask = self.c2.convole2d(self.l1_out)
        self.l2_out, self.max_l2 = self.l2.max_pool(self.c2_out)
        self.l2_out = self.l2_out.flatten()
        self.h1_out = self.h1.compute(self.l2_out)
        
        return self.o1.compute(self.h1_out)

    def train(self):
        for epoch in range(self.epochs):
            for i, data in enumerate(self.data):
                p = np.zeros(10)
                p[self.labels[i]] = 1
                q = self.feedforward(data)

                loss = cross_entropy_loss(p, q)

                dL_do1 = q - p
                dL_dw2 = np.outer(dL_do1, self.h1_out)
                dL_db4 = dL_do1

                do1_dx2 = self.o1.W
                dx2_dh1 = deriv_relu(np.dot(self.h1.W, self.l2_out) + self.h1.b)

                dL_dx2 = dL_do1 @ do1_dx2
                dL_dh1 = dL_dx2 * dx2_dh1

                dL_dw1 = np.outer(dL_dh1, self.l2_out)
                dL_db3 = dL_dh1

                dh1_dx1 = self.h1.W
                dL_dx1 = dL_dh1 @ dh1_dx1
            
                dL_dx1 = dL_dx1.reshape(6, 6)

                dL_dl2 = np.zeros((12, 12))
                
                for h in range(6):
                    for w in range(6):
                        dL_dl2[self.max_l2[0][h, w], self.max_l2[1][h, w]] = dL_dx1[h, w]

                dL_dz2 = dL_dl2[:11, :11] * self.c2_mask

                dL_dk2 = valid_conv(self.l1_out, dL_dz2)
                dL_db2 = sum(dL_dz2.flatten())
                dL_dc2 = full_conv(dL_dz2, self.c2.kernel)
                
                dL_dl1 = np.zeros((26, 26))

                for h in range(13):
                    for w in range(13):
                        dL_dl1[self.max_l1[0][h, w], self.max_l1[1][h, w]] = dL_dc2[h, w]

                dL_dz1 = dL_dl1 * self.c1_mask

                dL_dk1 = valid_conv(data, dL_dz1)
                dL_db1 = sum(dL_dz1.flatten())
                    
                print(f'Epoch: {epoch+1} Example: {i+1} Loss: {round(loss, 5)}')


                self.o1.W -= self.learn_rate * (dL_dw2 + self.lambda_reg * self.o1.W)
                self.h1.W -= self.learn_rate * (dL_dw1 + self.lambda_reg * self.h1.W)
                self.c2.kernel -= self.learn_rate * (dL_dk2 + self.lambda_reg * self.c2.kernel)
                self.c1.kernel -= self.learn_rate * (dL_dk1 + self.lambda_reg * self.c1.kernel)

                self.o1.b -= self.learn_rate * dL_db4
                self.h1.b -= self.learn_rate * dL_db3
                self.c2.b -= self.learn_rate * dL_db2
                self.c1.b -= self.learn_rate * dL_db1
                

model = NeuralNetwork(train_images, train_labels, 5)

model.train()

run_test_cases(test_images, test_labels, model)