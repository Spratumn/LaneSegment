import matplotlib.pyplot as plt
import pandas as pd
import os

from config import Config


cfg = Config()
train_data = pd.read_csv(os.path.join(cfg.LOG_DIR, 'train_log_deeplabv3+.csv'))
eval_data = pd.read_csv(os.path.join(cfg.LOG_DIR, 'eval_log_deeplabv3+.csv'))

train_loss = train_data['average_loss'].values[1:]
eval_loss = eval_data['average_loss'].values[1:]
eval_miou = eval_data['mean_iou'].values[1:]
eval_precision = eval_data['mean_precision'].values[1:]
eval_recall = eval_data['mean_recall'].values[1:]

plt.figure()
plt.title('eval result:')
plt.plot(range(len(eval_precision)), eval_precision, eval_recall)
plt.legend(["mean_precision", "mean_recall"])
plt.show()
