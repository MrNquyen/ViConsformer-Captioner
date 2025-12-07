from bisect import bisect
from icecream import ic
def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    return parameters


def lr_lambda_update(i_iter, cfg):
    if (
        cfg["use_warmup"] is True
        and i_iter <= cfg["warmup_iterations"]
    ):
        alpha = float(i_iter) / float(cfg["warmup_iterations"])
        return cfg["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg["lr_steps"], i_iter)
        return pow(cfg["lr_ratio"], idx)


def lr_lambda_update_epoch(current_epoch, cfg):
    warmup_epochs = cfg["warmup_epochs"]
    decay_factor = cfg["decay_factor"]
    lr_epoch_step_size = cfg["lr_epoch_step_size"]

    if current_epoch < warmup_epochs:
        return float(current_epoch) / float(max(1, warmup_epochs))
    else:
        num_decays = (current_epoch - warmup_epochs) // lr_epoch_step_size
        return decay_factor ** (num_decays)

