import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import _LRScheduler
# ハイパーパラメータ
peak_lr = 3e-4
warmup_epochs = 20
total_epochs = 1000


class CustomLRScheduler(_LRScheduler):
    def __init__(self, peak_lr, warmup_epochs, total_epochs, optimizer, last_epoch=-1):
        # 先にget_lrで必要となる独自の属性を定義する
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # optimizerのパラメータグループの数に合わせてbase_lrsを定義すると、より堅牢になります
        self.base_lrs = [peak_lr] * len(optimizer.param_groups)

        # 最後に親クラスの__init__を呼び出す
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]  # ウォームアップ期間中は線形増加
        else:
            # 元のコードではself.base_lrsの要素ごとに計算していたので、それに合わせます
            decay_ratio = 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.total_epochs - self.warmup_epochs)
                )
            )
            return [max(1e-5, base_lr * decay_ratio) for base_lr in self.base_lrs]

# ダミーパラメータ／オプティマイザ
dummy_param = torch.nn.Parameter(torch.zeros(1))
optimizer = optim.Adam([dummy_param], lr=peak_lr)

# スケジューラ
scheduler = CustomLRScheduler(peak_lr, warmup_epochs, total_epochs, optimizer)

# エポックごとにLRを記録
lrs = []
for epoch in range(total_epochs):
    scheduler.step()
    # get_lr() はグループ数分のリストを返すので，先頭要素を取得
    lr = scheduler.get_lr()[0]
    lrs.append(lr)

# プロット
plt.plot(range(1, total_epochs+1), lrs, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("CustomLRScheduler Schedule")
plt.grid(True)
plt.savefig('lr_schedule.png')  # 学習率スケジュールを画像として保存
plt.show()