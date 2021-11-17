from torch.autograd import Function


class GradReverse(Function):
    """
    梯度反转层GRL
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(self, constant):
        return GradReverse.apply(self, constant)
