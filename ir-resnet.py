import mxnet as mx
import sys
import os
import random
import cv2
import numpy as np
import multiprocessing
import logging
import fnmatch
import argparse
import cPickle as pickle
from matplotlib import pyplot as plt

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, "ResNet"))
from symbol_resnet import resnet

class DataBath(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label

class DataIter(mx.io.DataIter):
    def __init__(self, images, batch_size, height, width, process_num):
        assert process_num <= 40
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.conut = len(images)
        self.height = height
        self.width = width
        self.images = images
        self.cursor = -self.batch_size
        self.provide_data = [("positive", (self.batch_size, 3, height, width)),
                             ("negative", (self.batch_size, 3, height, width)),
                             ("one", (self.batch_size, ))]
        self.provide_label = [("anchor", (self.batch_size, 3, height, width))]
        self.queue = multiprocessing.Queue(maxsize=4)
        self.started = True
        self.processes = [multiprocessing.Process(target=self.write) for i in range(process_num)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def augment(self, mat):
        # bright = random.randint(60, 100)/100.0
        # mat = cv2.convertScaleAbs(mat, None, bright, 0)
        # mat = cv2.GaussianBlur(mat, (3, 3), 0, 0, borderType=cv2.BORDER_REPLICATE)
        rows, cols, _ = mat.shape
        # print rows, cols
        x_scale = random.randint(-12, 12) / 100.0
        y_scale = random.randint(-12, 12) / 100.0
        # x_scale = 0.1
        # y_scale = -0.1
        x_resize_scale = cols / (cols + abs(x_scale) * rows)
        y_resize_scale = rows / (rows + abs(y_scale) * cols)
        if x_scale >= 0:
            if y_scale >= 0:
                affine_matrix = np.float32([[x_resize_scale, x_resize_scale * x_scale, 0],
                                            [y_resize_scale * y_scale, y_resize_scale, 0]])
            else:
                affine_matrix = np.float32([[x_resize_scale, x_resize_scale * x_scale, 0],
                                            [y_resize_scale * y_scale, y_resize_scale,
                                             y_resize_scale * abs(y_scale) * cols]])
        else:
            if y_scale >= 0:
                affine_matrix = np.float32(
                    [[x_resize_scale, x_resize_scale * x_scale, x_resize_scale * abs(x_scale) * rows],
                     [y_resize_scale * y_scale, y_resize_scale, 0]])
            else:
                affine_matrix = np.float32(
                    [[x_resize_scale, x_resize_scale * x_scale, x_resize_scale * abs(x_scale) * rows],
                     [y_resize_scale * y_scale, y_resize_scale, y_resize_scale * abs(y_scale) * cols]])
        affine_mat = cv2.warpAffine(mat, affine_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        return affine_mat

    def generate_batch(self):
        ret = []
        while len(ret) < self.batch_size:
            a_idx, n_idx = random.sample(range(self.conut), 2)
            if a_idx == n_idx:
                continue
            a_mat = cv2.imread(self.images[a_idx])
            a_mat = cv2.resize(a_mat, (self.height, self.width))
            p_mat = self.augment(a_mat)
            p_mat = cv2.resize(p_mat, (self.height, self.width))
            n_mat = cv2.imread(self.images[n_idx])
            n_mat = cv2.resize(n_mat, (self.height, self.width))
            threshold = 250
            if np.mean(a_mat) > threshold or np.mean(p_mat) > threshold or np.mean(n_mat) > threshold:
                continue
            ret.append((a_mat, p_mat, n_mat))
        return ret

    def write(self):
        while True:
            if not self.started:
                break
            batch = self.generate_batch()
            a_batch = [x[0].transpose(2, 0, 1) for x in batch]
            p_batch = [x[1].transpose(2, 0, 1) for x in batch]
            n_batch = [x[2].transpose(2, 0, 1) for x in batch]
            one_batch = np.ones(self.batch_size)
            data_all = [mx.nd.array(p_batch),
                        mx.nd.array(n_batch),
                        mx.nd.array(one_batch)]
            label_all = [mx.nd.array(a_batch)]
            data_batch = DataBath(data_all, label_all)
            self.queue.put(data_batch)

    def __del__(self):
        self.started = False
        for process in self.processes:
            process.join()
            while not self.queue.empty():
                self.queue.get(block=False)

    def next(self):
        if self.queue.empty():
            logging.debug("waitting for data......")
        if self.iter_next():
            return self.queue.get(block=True)
        else:
            raise StopIteration

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.conut

    def reset(self):
        self.cursor = -self.batch_size

def get_network(batch_size):
    anchor = mx.symbol.Variable("anchor")
    positive = mx.symbol.Variable("positive")
    negative = mx.symbol.Variable("negative")
    concat = mx.symbol.Concat(
        *[anchor, positive, negative], dim=0, name="concat")
    share_net = resnet(
        data=concat,
        units=[2, 2, 2, 2],
        num_stage=4,
        filter_list=[64, 64, 128, 256, 512],
        num_class=128,
        data_type="imagenet",
        bottle_neck=False,
        bn_mom=0.9,
        workspace=512)
    one = mx.symbol.Variable("one")
    one = mx.symbol.Reshape(data=one, shape=(-1, 1))
    fa = mx.symbol.slice_axis(share_net, axis=0, begin=0, end=batch_size)
    fp = mx.symbol.slice_axis(share_net, axis=0, begin=batch_size, end=2 * batch_size)
    fn = mx.symbol.slice_axis(share_net, axis=0, begin=2 * batch_size, end=3 * batch_size)
    fs = fa-fp
    fd = fa-fn
    fs = fs*fs
    fd = fd*fd
    fs = mx.symbol.sum(fs, axis=1, keepdims=1)
    fd = mx.symbol.sum(fd, axis=1, keepdims=1)
    loss = fd - fs
    loss = one - loss
    loss = mx.symbol.Activation(data=loss, act_type='relu')
    return mx.symbol.MakeLoss(loss)

class Search(object):
    def __init__(self, model_path, epoch, height, width, imgs=None, codebook="./index.pkl"):
        symbol, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
        input_shape = dict([('data', (1, 3, height, width))])
        network = self.get_predict_net()
        # self.executor = mx.model.FeedForward(symbol=network, ctx=mx.gpu(), arg_params=arg_params, aux_params=aux_params, allow_extra_params=True)
        self.executor = network.simple_bind(ctx=mx.gpu(), **input_shape)
        self.executor.copy_params_from(arg_params, aux_params, allow_extra_params=True)
        self.args = dict(zip(network.list_arguments(), self.executor.arg_arrays))
        self.data = self.args["data"]
        self.height = height
        self.width = width
        if codebook is None:
            self.imgs = imgs
            assert self.imgs is not None
            self.build_index(self.imgs)
        else:
            self.imgs, self.codebook = pickle.load(open(codebook))

    def build_index(self, imgs):
        self.codebook = np.empty(shape=(len(imgs), 128))
        for idx, img in enumerate(imgs):
            if idx % 100 == 0:
                print idx
            mat = cv2.imread(img)
            mat = self.preprocess(mat)
            mat = np.transpose(mat, (2, 0, 1))
            mat = np.array([mat])
            self.codebook[idx, :] = self.get_feature(mat)

    def save(self, path):
        pickle.dump((self.imgs, self.codebook), open(path, 'w'), pickle.HIGHEST_PROTOCOL)

    def preprocess(self, mat):
        mat = cv2.resize(mat, (self.width, self.height))

        def enhance(mat):
            rows, cols, channel = mat.shape
            # mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            norm = cv2.normalize(mat, mat, 0, 255, cv2.NORM_MINMAX)
            scale = cv2.convertScaleAbs(norm, None, 1.2, 0)
            return scale

        def need_enhance(mat):
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2HSV)[:, :, 2]
            hist, _ = np.histogram(mat, 256, (0, 256))
            if np.argmax(hist) < 200:
                return True
            else:
                return False

        if need_enhance(mat):
            return enhance(mat)
        else:
            return mat

    def get_feature(self, mat, search=False):
        self.data[:] = mx.nd.array([mat])
        return self.executor.forward(is_train=False)[0].asnumpy()[0]

    def search(self, mat, top_k=5):
        assert self.codebook is not None and self.imgs is not None
        mat = cv2.GaussianBlur(mat, (3, 3), 0, 0, borderType=cv2.BORDER_REPLICATE)
        mat = cv2.resize(mat, (self.height, self.width))
        # mat = self.preprocess(mat)
        mat = np.transpose(mat, (2, 0, 1))
        code = self.get_feature(mat, search=True)[0]
        distance = np.linalg.norm(code - self.codebook, axis=1)
        print distance
        arg_result = np.argsort(distance)
        print arg_result
        result = []
        for idx in arg_result[:top_k]:
            result.append(self.imgs[idx])
        return mat, result

    def get_predict_net(self):
        data = mx.symbol.Variable("data")
        network = resnet(
            data=data,
            units=[2, 2, 2, 2],
            num_stage=4,
            filter_list=[64, 64, 128, 256, 512],
            num_class=128,
            data_type="imagenet",
            bottle_neck=False,
            bn_mom=0.9,
            workspace=512)
        return network

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

def train():
    parser = argparse.ArgumentParser(description="Image Search Using CNN")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--process_num", type=int, default=4)
    parser.add_argument("--root", type=str, default="")
    args = parser.parse_args()
    batch_size = args.batch_size
    dev = args.gpus
    network = get_network(batch_size=batch_size)
    # symbol, arg_params, aux_params = mx.model.load_checkpoint("resnet-18", 0)

    # shape = {"anchor": (batch_size, 3, 224, 224),
    #          "positive": (batch_size, 3, 224, 224),
    #          "negative": (batch_size, 3, 224, 224),
    #          "one": (batch_size, )}
    # mx.visualization.plot_network(network, shape=shape).render("ir-resnet", cleanup=True)
    images = []
    root_dir = args.root
    for root, dirnames, filenames in os.walk(root_dir):
        for img in fnmatch.filter(filenames, "*.jpg"):
            images.append(os.path.abspath(os.path.join(root, img)))

    train_set = DataIter(images=images, batch_size=batch_size, height=224, width=224, process_num=args.process_num)
    # lr_scheduler = mx.lr_scheduler.FactorScheduler(step=10, factor=0.1)
    optimizer = mx.optimizer.SGD(momentum=0.99)  # lr_scheduler=lr_scheduler)
    model = mx.model.FeedForward(
        allow_extra_params=True,
        ctx=mx.gpu(dev),
        symbol=network,
        num_epoch=200,
        learning_rate=0.1*1e-2,
        wd=0.0001,
        initializer=mx.init.Load("resnet-18-0000.params", default_init=mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2)),
        optimizer=optimizer)
    model.fit(X=train_set,
              eval_metric=Auc(),
              kvstore='local_allreduce_device',
              batch_end_callback=mx.callback.Speedometer(batch_size, 10),
              epoch_end_callback=mx.callback.do_checkpoint("models/ir-blur"))

def test():
    paser = argparse.ArgumentParser(description="Test Search")
    paser.add_argument("--model_path", type=str, default="models/ir")
    paser.add_argument("--epoch", type=int, default=20)
    paser.add_argument("--root", type=str, default="")
    paser.add_argument("--test_dir", type=str, default="test/")
    args = paser.parse_args()
    images = []
    root_dir = args.root
    for root, dirnames, filenames in os.walk(root_dir):
        for img in fnmatch.filter(filenames, "*.jpg"):
            images.append(os.path.abspath(os.path.join(root, img)))
    s = Search(model_path=args.model_path, epoch=args.epoch, height=224, width=224, imgs=images, codebook="./index.pkl")
    # s.save("index.pkl")
    # -----build index ------
    # search
    test_imgs = []
    root_dir = args.test_dir
    for root, dirnames, filenames in os.walk(root_dir):
        for img in fnmatch.filter(filenames, "*.jpg"):
            test_imgs.append(os.path.abspath(os.path.join(root, img)))

    def display(mat, processed_mat, imgs):
        plt.figure(figsize=(16, 12), dpi=240)
        aix1 = plt.subplot(2, 6, 1)
        plt.sca(aix1)
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        plt.imshow(mat)
        plt.xticks([]), plt.yticks([])
        aix1 = plt.subplot(2, 6, 2)
        plt.sca(aix1)
        processed_mat = cv2.cvtColor(processed_mat, cv2.COLOR_BGR2RGB)
        plt.imshow(processed_mat)
        plt.xticks([]), plt.yticks([])
        for j, img in enumerate(imgs):
            mat = cv2.imread(img)
            aix1 = plt.subplot(2, 6, j + 3)
            plt.sca(aix1)
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            plt.imshow(mat)
            plt.xticks([]), plt.yticks([])
        plt.show()
    for img in test_imgs:
        mat = cv2.imread(img)
        processed_mat, sorted_imgs = s.search(mat, top_k=10)
        processed_mat = np.transpose(processed_mat, (1, 2, 0))
        display(mat, processed_mat, sorted_imgs)

def predict():
    pass
if __name__ == "__main__":
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    train()
    # test()
