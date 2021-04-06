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
            offsets = [0] * len(rf.controls)
            checkpoints = []
            more_data = True
            outs = []
            batched_single_data = False
            shapes = []
            for arg in args:
                shapes.append(arg.shape)
            ctx.inp_shapes = shapes

            while more_data:
                for i, (c, arr) in enumerate(zip(rf.controls, args)):
                    batched_single_data = (len(arr.shape) > 0 and arr.shape[0] == 1)
                    data = arr.data.numpy().flatten()
                    offsets[i] = c.update_numpy(data, offsets[i])
                    more_data = offsets[i] < data.shape[0]
                y = rf.replay()
                y = Enlist(y)
                for i, v in enumerate(y):
                    if len(outs) <= i:
                        outs.append([])
                    outs[i].append(torch.tensor(v._ad_to_list(v)))
                checkpoints.append(rf.save_checkpoints())
            ctx.checkpoints = checkpoints
            y = []
            for out in outs:
                if len(out) > 1 or batched_single_data:
                    out = torch.stack(out)
                    ctx.batched = True
                else:
                    out = out[0]
                    ctx.batched = False
                y.append(out)
            return rf.outputs.delist(y)

        @staticmethod
        def backward(ctx, *grad_output):
            outs = []
            for i, checkpoint in enumerate(ctx.checkpoints):
                rf.set_checkpoints(checkpoint)
                dy = []
                for o, g in zip(rf.outputs, grad_output):
                    if ctx.batched:
                        g = g[i]
                    grad = create_overloaded_object(o.saved_output)._ad_copy()
                    grad._ad_assign_numpy(grad, g.data.numpy(), 0)
                    dy.append(grad._ad_to_adj_value(grad))
                dydc = rf.adj_jac_action(rf.outputs.delist(dy))
                dydc = Enlist(dydc)

                for i, dc in enumerate(dydc):
                    if len(outs) <= i:
                        outs.append([])
                    outs[i].append(torch.tensor(dc._ad_to_list(dc)))
            y = []
            for i, out in enumerate(outs):
                out = torch.stack(out)
                out = out.reshape(ctx.inp_shapes[i])
                y.append(out)
            return rf.controls.delist(y)
    return TorchRF


import multiprocessing as mp
from operator import itemgetter
mpctx = mp.get_context("fork")
import time


def ReducedFunctionTorchMulti(outputs, controls):
    rf = ReducedFunction(outputs, controls)

    class TorchRF(autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            offsets = [0] * len(rf.controls)
            checkpoints = []
            more_data = True
            outs = []
            batched_single_data = False
            shapes = []
            for arg in args:
                shapes.append(arg.shape)
            ctx.inp_shapes = shapes
            manager = mp.Manager()
            j = 0
            processes = []
            queues = []
            q = manager.Queue()

            while more_data:
                for i, (c, arr) in enumerate(zip(rf.controls, args)):
                    batched_single_data = (len(arr.shape) > 0 and arr.shape[0] == 1)
                    data = arr.data.numpy().flatten()
                    offsets[i] = c.update_numpy(data, offsets[i])
                    more_data = offsets[i] < data.shape[0]

                def replay_tape(j, q, r):
                    y = rf.replay()
                    q.put([j, y.vector().get_local()])

                    v = r.get(timeout=1000)
                    dy = []
                    for o, g in zip(rf.outputs, v):
                        g = g[i]
                        grad = create_overloaded_object(o.saved_output)._ad_copy()
                        grad._ad_assign_numpy(grad, g.data.numpy(), 0)
                        dy.append(grad._ad_to_adj_value(grad))
                    dydc = rf.adj_jac_action(rf.outputs.delist(dy))
                    q.put([j, dydc.vector().get_local()])

                r = manager.Queue()
                p = mpctx.Process(target=replay_tape, args=(j, q, r))
                p.start()
                j += 1
                processes.append(p)
                queues.append(r)

            o = []
            for _ in range(j):
                v = q.get(timeout=100)
                o.append(v)
            #[p.join() for p in processes]
            o = sorted(o, key=itemgetter(0))

            for _, y in o:
                y = Enlist(y)
                for i, v in enumerate(y):
                    if len(outs) <= i:
                        outs.append([])
                    outs[i].append(torch.tensor(v))
                checkpoints.append(rf.save_checkpoints())

            ctx.checkpoints = checkpoints
            ctx.processes = processes
            ctx.queues = queues
            ctx.queue = q
            y = []
            for out in outs:
                if len(out) > 1 or batched_single_data:
                    out = torch.stack(out)
                    ctx.batched = True
                else:
                    out = out[0]
                    ctx.batched = False
                y.append(out)
            return rf.outputs.delist(y)

        @staticmethod
        def backward(ctx, *grad_output):
            outs = []
            for i, checkpoint in enumerate(ctx.checkpoints):
                ctx.queues[i].put(grad_output)

            o = []
            for i, p in enumerate(ctx.processes):
                v = ctx.queue.get(timeout=1000)
                o.append(v)
            o = sorted(o, key=itemgetter(0))

            for _, y in o:
                y = Enlist(y)
                for i, v in enumerate(y):
                    if len(outs) <= i:
                        outs.append([])
                    outs[i].append(torch.tensor(v))

            y = []
            for i, out in enumerate(outs):
                out = torch.stack(out)
                out = out.reshape(ctx.inp_shapes[i])
                y.append(out)
            return rf.controls.delist(y)
    return TorchRF


