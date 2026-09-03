import torch
import numpy 

X = torch.from_numpy(np.array(x_list, dtype=np.float32))
Y = torch.from_numpy(np.array(y_list, dtype=np.float32)).unsqueeze(1)


net = torch.nn.Sequential(
    torch.nn.Linear(768, 256), torch.nn.ReLU(),
    torch.nn.Linear(256, 32),  torch.nn.ReLU(),
    torch.nn.Linear(32, 1),    torch.nn.Tanh(),
)


loss_fn = torch.nn.functional.mse_loss     # regression: predict a number
opt = torch.optim.Adam(net.parameters(), lr=1e-3)


for epoch in range(EPOCHS):
    for xb, yb in loader:
        opt.zero_grad()          # clear last step's gradients
        pred = net(xb)           # forward
        loss = loss_fn(pred, yb)
        loss.backward()          # compute gradients
        opt.step()

for i, layer in enumerate([net[0], net[2], net[4]]):
    w[f"W{i}"] = layer.weight.detach().numpy().T.copy()
    w[f"b{i}"] = layer.bias.detach().numpy()
np.savez("eval_net.npz", **w)