import pandas as pd
import torch
from envs.Classification_n_points_Env import FirstN
from supervised_learning.models.body import RNN, FeedForward, LSTM, CNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

len_stroke = 10
norm = True
plot_results = None
train_epochs = 1000
test_epochs = 200

to_plot = {}

# FeedForward
df = pd.read_csv("../datasets/stroke_gesture_df_remapped.csv", index_col=0)
env = FirstN(df, len_stroke, plot_results=plot_results, norm=norm)
criterion = torch.nn.BCELoss()
print("\n||FF||")
net = FeedForward(len_stroke, 20)
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
to_plot["ff"] = []

# train
net.train()
obs = env.reset()
for epoch in range(train_epochs):
    obs = torch.from_numpy(obs.flatten()).float()
    optimizer.zero_grad()
    y_pred = net(obs)
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()
    if done:
        obs = env.reset()

# eval
net.eval()
env.do_test = True
env.do_train = False
test_loss = 0
obs = env.reset()
for epoch in range(test_epochs):
    obs = torch.from_numpy(obs.flatten()).float()
    optimizer.zero_grad()
    y_pred = net(obs)
    to_plot["ff"].append((obs.detach().numpy().reshape(len_stroke, 2), env.label, y_pred.detach().numpy()))
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    loss.backward()
    test_loss += loss.item()
    optimizer.step()
    if done:
        obs = env.reset()
print('Avg FF Test loss: {}'.format(test_loss/test_epochs))


# RNN
env = FirstN(df, len_stroke, plot_results=plot_results, norm=norm)
print("\n||RNN||")
net = RNN(2, 20, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
to_plot["rnn"] = []

# train
net.train()
obs = env.reset()
for epoch in range(train_epochs):
    net.hidden = None
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0][0]
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()
    if done:
        obs = env.reset()

# eval
net.eval()
env.do_test = True
env.do_train = False
test_loss = 0
obs = env.reset()
for epoch in range(test_epochs):
    net.hidden = None
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0][0]
    to_plot["rnn"].append((obs.detach().numpy().reshape(len_stroke, 2), env.label, y_pred.detach().numpy()))
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    loss.backward()
    test_loss += loss.item()
    optimizer.step()
    if done:
        obs = env.reset()
print('Avg RNN Test loss: {}'.format(test_loss/test_epochs))


# LSTM
env = FirstN(df, len_stroke, plot_results=plot_results, norm=norm)
print("\n||LSTM||")
net = LSTM(2, 20, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.005)
to_plot["lstm"] = []

# train
net.train()
obs = env.reset()
for epoch in range(train_epochs):
    net.hidden = None
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0][0]
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()
    if done:
        obs = env.reset()

# eval
net.eval()
env.do_test = True
env.do_train = False
test_loss = 0
obs = env.reset()
for epoch in range(test_epochs):
    net.hidden = None
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0][0]
    to_plot["lstm"].append((obs.detach().numpy().reshape(len_stroke, 2), env.label, y_pred.detach().numpy()))
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    loss.backward()
    test_loss += loss.item()
    optimizer.step()
    if done:
        obs = env.reset()
print('Avg LSTM Test loss: {}'.format(test_loss/test_epochs))

# CNN
env = FirstN(df, len_stroke, plot_results=plot_results, norm=norm)

print("\n||CNN||")
net = CNN()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
to_plot["cnn"] = []

# train
net.train()
obs = env.reset()
for epoch in range(train_epochs):
    obs = torch.from_numpy(obs).reshape(1, 1, obs.shape[0], obs.shape[1]).float()
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0]
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    if epoch % 100 == 0:
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    loss.backward()
    optimizer.step()
    if done:
        obs = env.reset()

# eval
net.eval()
env.do_test = True
env.do_train = False
test_loss = 0
obs = env.reset()
for epoch in range(test_epochs):
    obs = torch.from_numpy(obs).reshape(1, 1, obs.shape[0], obs.shape[1]).float()
    optimizer.zero_grad()
    y_pred = net(obs)
    y_pred = y_pred[0]
    to_plot["cnn"].append((obs.detach().numpy().reshape(len_stroke, 2), env.label, y_pred.detach().numpy()))
    obs, rewards, done, info = env.step(y_pred)
    y_train = torch.from_numpy(np.array([env.label])).float()
    loss = criterion(y_pred, y_train)
    loss.backward()
    test_loss += loss.item()
    optimizer.step()
    if done:
        obs = env.reset()
print('Avg CNN Test loss: {}'.format(test_loss/test_epochs))

print("\n|| Results ||\n")
# calculating results
for key in to_plot.keys():
    truth = []
    preds = []
    print(key)
    for i in to_plot[key]:
        obs, label, classification = i
        truth.append(label)
        preds.append(int(np.round(classification)))
    print(confusion_matrix(truth, preds))


# plotting results
for key in to_plot.keys():
    for j, i in enumerate(to_plot[key]):
        if j % 10 == 0:
            obs, label, classification = i
            plt.figure(figsize=(10, 5))
            plt.scatter(obs[:, 0], obs[:, 1])
            plt.title("{}:- Actual: {} and Classification: {} ".format(key, str(label), str(classification)))
            plt.show()




