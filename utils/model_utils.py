from bisect import bisect
from icecream import ic
def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    return parameters


def lr_lambda_update(i_iter, cfg):
    ic(cfg)
    if (
        cfg["use_warmup"] is True
        and i_iter <= cfg["warmup_iterations"]
    ):
        alpha = float(i_iter) / float(cfg["warmup_iterations"])
        return cfg["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg["lr_steps"], i_iter)
        return pow(cfg["lr_ratio"], idx)
