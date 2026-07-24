import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
df = pd.read_csv('data.csv', sep=';')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_ds = ChallengeDataset(train_df.reset_index(drop=True), 'train')
val_ds = ChallengeDataset(val_df.reset_index(drop=True), 'val')
train_dl = t.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = t.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

# create an instance of our ResNet model
# TODO
resnet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
crit = t.nn.BCELoss()
optim = t.optim.Adam(resnet.parameters(), lr=1e-4)
trainer = Trainer(resnet, crit, optim=optim, train_dl=train_dl, val_test_dl=val_dl,
                  cuda=t.cuda.is_available(), early_stopping_patience=10)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')