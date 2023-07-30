
def clip_grad(grad, norm=50):
    """
    Clips gradients in a list of gradients
    :param grad: list of gradients.
    :param norm: gradient norm.
    :return: list of clipped gradients
    """
    grad_clipped = []
    for g in grad:
        g = (g * (g < norm).float()) + ((g > norm).float()) * norm
        g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
        grad_clipped.append(g)

    return grad_clipped