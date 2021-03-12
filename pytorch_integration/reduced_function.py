import torch
import torch.autograd as autograd

from pyadjoint.enlisting import Enlist
from pyadjoint.reduced_function import ReducedFunction
from pyadjoint.overloaded_type import create_overloaded_object


def ReducedFunctionTorch(outputs, controls):
    rf = ReducedFunction(outputs, controls)

    class TorchRF(autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            for c, arr in zip(rf.controls, args):
                c.update_numpy(arr.data.numpy(), 0)
            y = rf.replay()
            y = Enlist(y)
            out = []
            for v in y:
                out.append(torch.tensor(v._ad_to_list(v)))
            p = rf.save_checkpoints()
            ctx.checkpoints = p
            return rf.outputs.delist(out)

        @staticmethod
        def backward(ctx, *grad_output):
            p = ctx.checkpoints
            rf.set_checkpoints(p)
            dy = []
            for o, g in zip(rf.outputs, grad_output):
                grad = create_overloaded_object(o.saved_output)._ad_copy()
                grad._ad_assign_numpy(grad, g.data.numpy(), 0)
                dy.append(grad._ad_to_adj_value(grad))
            dydc = rf.adj_jac_action(rf.outputs.delist(dy))
            dydc = Enlist(dydc)

            return rf.controls.delist([torch.tensor(dc._ad_to_list(dc)) for dc in dydc])
    return TorchRF


