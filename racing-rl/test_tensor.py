import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

x = torch.tensor([[0,0,0], [0,-1, 0], [0,0.4, 0], [0,-0.4,0]], device=device, dtype=torch.float32)
print(x)
x_np = x.cpu().numpy()
negative_val_mask = x_np[:,1] < 0
x_np[negative_val_mask,2] = -x_np[negative_val_mask,1]
x_np[negative_val_mask,1] = 0
x = torch.tensor(x_np, device=device, dtype=torch.float32)
print(x)