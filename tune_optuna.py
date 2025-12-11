# 文件名: tune_optuna_cpu.py
import json
import math
import mindspore as ms
from mindspore import nn, ops, Tensor, context, dtype as mstype
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as T
import mindspore.dataset.vision as V
import numpy as np
import random, os, cv2
import optuna
import transformer
import utils
from vgg import VGG19_Feature

# ------------------ Config ------------------
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "mscoco_sampled_100"  # 小数据集快速调参
STYLE_IMAGE_PATH = "images/mosaic.jpg"
SEED = 35
GRAD_CLIP_NORM = 1e5

# ---- CPU 环境 ----
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
ms.common.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------ Dataset ------------------
class CustomDataset:
    def __init__(self, folder_path, transform=None):
        self.img_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png'))]
        self.transform = transform
    def __len__(self): return len(self.img_paths)
    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        if img is None: img = np.zeros((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3), dtype=np.uint8)
        if self.transform: img = self.transform(img)
        img = img.astype(np.float32)/255.0
        img = img.transpose(2,0,1)
        return img

# ------------------ Loss ------------------
class StyleTransferLoss(nn.Cell):
    def __init__(self, transformer, content_weight, style_weight):
        super().__init__()
        self.transformer = transformer
        self.vgg = VGG19_Feature()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.l2_loss = nn.MSELoss()

        style_image = cv2.imread(STYLE_IMAGE_PATH)
        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
        style_image = cv2.resize(style_image, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
        style_tensor = Tensor(np.expand_dims(style_image.transpose(2,0,1),0), dtype=mstype.float32)/255.0
        feats = self.vgg(style_tensor)
        self.style_gram_features = {k: utils.gram(v)[0] for k, v in feats.items()}

        # 只计算存在的 feature layer
        self.style_layers = list(self.style_gram_features.keys())

    def construct(self, content_image):
        generated = self.transformer(content_image)
        content_feats = self.vgg(content_image)
        gen_feats = self.vgg(generated)

        # 对齐 feature map 尺寸
        for k in gen_feats.keys():
            if k in content_feats and gen_feats[k].shape != content_feats[k].shape:
                gen_feats[k] = ops.interpolate(
                    gen_feats[k],
                    size=(content_feats[k].shape[2], content_feats[k].shape[3]),
                    mode='bilinear', align_corners=False
                )

        content_loss = self.content_weight * self.l2_loss(
            gen_feats['relu2_2'], content_feats['relu2_2']
        )

        style_loss = Tensor(0.0, mstype.float32)
        for layer in self.style_layers:
            gen_gram = utils.gram(gen_feats[layer])[0]
            style_gram = self.style_gram_features[layer]
            if gen_gram.shape != style_gram.shape:
                gen_gram = ops.interpolate(
                    gen_gram,
                    size=(style_gram.shape[2], style_gram.shape[3]),
                    mode='bilinear', align_corners=False
                )
            style_loss += self.l2_loss(gen_gram, style_gram)

        style_loss *= self.style_weight
        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss, generated

# ------------------ Train Step ------------------
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__(auto_prefix=False)
        self.network = network
        self.weights = network.transformer.trainable_params()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.network, None, self.weights, has_aux=True)
        self.sqrt = ops.Sqrt()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.square = ops.Square()
        self.minimum = ops.Minimum()
        self.zeros_like = ops.ZerosLike()
        self.cast = ops.Cast()

    def construct(self, content_image):
        (total_loss, content_loss, style_loss, generated), grads = self.grad_fn(content_image)
        total_sq = Tensor(0.0, mstype.float32)
        for g in grads:
            if g is None: continue
            total_sq += self.reduce_sum(self.square(self.cast(g, mstype.float32)))
        global_norm = self.sqrt(total_sq + Tensor(1e-12, mstype.float32))
        scale = self.minimum(Tensor(1.0, mstype.float32), Tensor(GRAD_CLIP_NORM, mstype.float32)/(global_norm+Tensor(1e-6, mstype.float32)))
        clipped = [self.cast(g if g is not None else self.zeros_like(w), mstype.float32) * scale for g,w in zip(grads, self.weights)]
        self.optimizer(tuple(clipped))
        return total_loss, content_loss, style_loss, generated, global_norm, scale

# ------------------ Optuna Objective ------------------
def objective(trial):
    content_weight = trial.suggest_float("content_weight", 5.0, 20.0)
    style_weight = trial.suggest_float("style_weight", 1.0, 50.0)
    adam_lr = trial.suggest_float("adam_lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1,2,4])  # CPU上小一点

    transform = T.Compose([V.Resize((TRAIN_IMAGE_SIZE,TRAIN_IMAGE_SIZE)), V.RandomHorizontalFlip(0.5)])
    dataset = CustomDataset(DATASET_PATH, transform)
    data_loader = GeneratorDataset(dataset, column_names=["image"], num_parallel_workers=1, shuffle=True)
    data_loader = data_loader.map(operations=T.TypeCast(mstype.float32), input_columns=["image"])
    data_loader = data_loader.batch(batch_size)

    transformer_net = transformer.TransformerNet()
    loss_net = StyleTransferLoss(transformer_net, content_weight, style_weight)
    optimizer = nn.Adam(transformer_net.trainable_params(), learning_rate=adam_lr)
    train_net = TrainOneStepCell(loss_net, optimizer)
    train_net.set_train()

    total_loss_accum = 0.0
    for epoch in range(1):
        for batch_in_epoch, data in enumerate(data_loader.create_tuple_iterator(), start=1):
            content_batch = data[0]
            total_loss, _, _, _, _, _ = train_net(content_batch)
            total_loss_accum += float(total_loss.asnumpy())
            if batch_in_epoch >= 5:  # 小样本快速测试
                break
        break

    return total_loss_accum / 5

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\n===== Optuna 最佳结果 =====")
    trial = study.best_trial
    print(f"Loss: {trial.value}")
    print(f"Params: {trial.params}")

    with open("best_params.json", "w") as f:
        json.dump(trial.params, f)
    print("最优参数已保存到 best_params.json")
