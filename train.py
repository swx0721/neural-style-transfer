# train_cpu_safe_full_fixed.py
import mindspore as ms
from mindspore import nn, ops, Tensor, context, dtype as mstype
from mindspore.dataset import GeneratorDataset
import mindspore.dataset.transforms as T
import mindspore.dataset.vision as V
import numpy as np
import random
import time
import os
import cv2
import transformer
import vgg
import utils

# ------------------ GLOBAL SETTINGS ------------------
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "mscoco_sampled_1000"
NUM_EPOCHS = 1
STYLE_IMAGE_PATH = "images/mosaic.jpg"
BATCH_SIZE = 4
# CONTENT_WEIGHT = 19.09585
# STYLE_WEIGHT = 8.919890
# ADAM_LR = 9.681906077988061e-05
CONTENT_WEIGHT = 2.0
STYLE_WEIGHT = 2.5e2
ADAM_LR = 1e-4
SAVE_MODEL_PATH = "transforms/"
SAVE_IMAGE_PATH = "images/results/"
SAVE_MODEL_EVERY = 500
SEED = 35
PLOT_LOSS = 1
GRAD_CLIP_VALUE = 1.0

# ------------------ Device Setting（修复废弃API提示） ------------------
context.set_context(mode=context.PYNATIVE_MODE)
ms.set_device("CPU")  # 使用新API设置设备，替代deprecated的device_target

# ------------------ Seed ------------------
ms.common.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------ Dataset ------------------
class CustomDataset:
    def __init__(self, folder_path, transform=None):
        self.img_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                          if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.clip(img, 0.0, 1.0)
        if self.transform:
            img = self.transform(img)
        img = img.transpose(2, 0, 1)
        return img

# ------------------ Loss Function（保留梯度链路修复） ------------------
class StyleTransferLoss(nn.Cell):
    def __init__(self, transformer_net, content_weight=CONTENT_WEIGHT, style_weight=STYLE_WEIGHT):
        super().__init__()
        self.transformer = transformer_net
        self.vgg = vgg.VGG19_Feature()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.l2_loss = nn.MSELoss()

        # 载入风格图并提取风格特征
        style_image = utils.load_image(STYLE_IMAGE_PATH)
        if style_image is None:
            raise RuntimeError(f"Style image not found: {STYLE_IMAGE_PATH}")
        style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
        style_image = cv2.resize(style_image, (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
        style_image = style_image.astype(np.float32) / 255.0
        style_image = np.clip(style_image, 0.0, 1.0)
        style_tensor = Tensor(np.expand_dims(style_image.transpose(2,0,1),0), mstype.float32)
        
        style_feats = self.vgg(style_tensor)
        self.style_gram_features = {k: utils.gram(v) for k,v in style_feats.items()}

    def _fix_size(self, gen_feat, con_feat):
        ch, cw = con_feat.shape[2], con_feat.shape[3]
        return ops.interpolate(gen_feat, size=(ch, cw), mode='bilinear', align_corners=False)

    def construct(self, content_image):
        generated_image = self.transformer(content_image)
        content_feats = self.vgg(content_image)
        gen_feats = self.vgg(generated_image)

        # 内容损失
        gen_c = self._fix_size(gen_feats['relu2_2'], content_feats['relu2_2'])
        content_loss = self.content_weight * self.l2_loss(gen_c, content_feats['relu2_2'])

        # 风格损失（适配批次维度）
        style_loss = Tensor(0.0, mstype.float32)
        for layer in ['relu1_2','relu2_2','relu3_4','relu4_4','relu5_4']:
            gen_f = gen_feats[layer]
            gen_gram = utils.gram(gen_f)
            style_gram = self.style_gram_features[layer]
            style_gram_batched = ops.broadcast_to(style_gram, (gen_gram.shape[0], -1, -1))
            style_loss += self.l2_loss(gen_gram, style_gram_batched)
        style_loss *= self.style_weight

        total_loss = content_loss + style_loss
        return total_loss, content_loss, style_loss, generated_image

# ------------------ Train Step ------------------
class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer, clip_value=GRAD_CLIP_VALUE):
        super().__init__(auto_prefix=False)
        self.network = network
        self.weights = network.transformer.trainable_params()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.network, None, self.weights, has_aux=True)
        self.zeros_like = ops.ZerosLike()
        self.cast = ops.Cast()
        self.clip_value = clip_value

    def construct(self, content_image):
        (total_loss, content_loss, style_loss, generated_image), grads = self.grad_fn(content_image)
        clipped_grads = []
        for g, w in zip(grads, self.weights):
            if g is None:
                g = self.zeros_like(w)
            g = ops.clip_by_value(g, -self.clip_value, self.clip_value)
            g = self.cast(g, mstype.float32)
            clipped_grads.append(g)
        self.optimizer(tuple(clipped_grads))
        return total_loss, content_loss, style_loss, generated_image

# ------------------ Train Function ------------------
def train():
    transform = T.Compose([V.Resize((TRAIN_IMAGE_SIZE,TRAIN_IMAGE_SIZE)), V.RandomHorizontalFlip(0.5)])
    TransformerNetwork = transformer.TransformerNet()
    LossNetwork = StyleTransferLoss(TransformerNetwork, CONTENT_WEIGHT, STYLE_WEIGHT)
    optimizer = nn.Adam(TransformerNetwork.trainable_params(), learning_rate=ADAM_LR)

    dataset = CustomDataset(DATASET_PATH, transform)
    total_data_size = len(dataset)
    print(f"Dataset Size: {total_data_size}")
    data_loader = GeneratorDataset(dataset, column_names=["image"], num_parallel_workers=1, shuffle=True)
    data_loader = data_loader.map(operations=T.TypeCast(mstype.float32), input_columns=["image"])
    data_loader = data_loader.batch(BATCH_SIZE)

    import math
    steps_per_epoch = math.ceil(total_data_size / BATCH_SIZE)
    train_net = TrainOneStepCell(LossNetwork, optimizer)
    train_net.set_train()

    start_time = time.time()
    total_loss_history, content_loss_history, style_loss_history = [], [], []
    for epoch in range(NUM_EPOCHS):
        for step, data in enumerate(data_loader.create_tuple_iterator(), 1):
            content_batch = data[0]
            total_loss, content_loss, style_loss, generated_batch = train_net(content_batch)
            
            current_total = total_loss.asnumpy().item()
            current_content = content_loss.asnumpy().item()
            current_style = style_loss.asnumpy().item()
            total_loss_history.append(current_total)
            content_loss_history.append(current_content)
            style_loss_history.append(current_style)

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step}/{steps_per_epoch}] "
                  f"Loss: {current_total:.2f} (Content: {current_content:.2f}, Style: {current_style:.2f})")

            # 保存模型和生成图片
            if step % SAVE_MODEL_EVERY == 0 or step == steps_per_epoch:
                os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
                os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
                checkpoint_path = os.path.join(SAVE_MODEL_PATH, f"checkpoint_{step}.ckpt")
                ms.save_checkpoint(TransformerNetwork, checkpoint_path)
                
                sample_tensor = generated_batch[0:1]
                sample_image = utils.ttoi(sample_tensor)
                utils.saveimg(sample_image, os.path.join(SAVE_IMAGE_PATH, f"sample0_{step}.png"))

    stop_time = time.time()
    print(f"Done Training! Time elapsed: {stop_time - start_time:.2f} seconds")
    TransformerNetwork.set_train(False)
    final_path = os.path.join(SAVE_MODEL_PATH, f"final_{os.path.basename(STYLE_IMAGE_PATH).split('.')[0]}.ckpt")
    ms.save_checkpoint(TransformerNetwork, final_path)
    print(f"Final model saved to {final_path}")

    if PLOT_LOSS:
        print("Plotting losses...")
        utils.plot_losses(total_loss_history, content_loss_history, style_loss_history)

if __name__ == "__main__":
    train()