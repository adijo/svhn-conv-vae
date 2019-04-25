import torch
from ignite.engine import Engine
from ignite.utils import convert_tensor


def create_vae_trainer(model, optimizer, loss_fn, device):
    def _update(engine, data):
        model.train()
        optimizer.zero_grad()
        data = convert_tensor(data[0], device=device)
        reconstructed_data, mu, log_var = model(data)
        loss, mse, kld = loss_fn(reconstructed_data, data, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item() / len(data), mse.item() / len(data), kld.item() / len(data)

    return Engine(_update)
