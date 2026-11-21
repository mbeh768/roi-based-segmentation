"""Weight probe: is the network actually using the ROI channel?

The ROI reaches the network at exactly one place. model/model.py concatenates a
1-channel ROI crop onto the bottleneck and feeds it to up5:

    x = torch.cat([x, roi_at(*x.shape[2:])], dim=1)   # (B, 512+1, 16, 16)
    ... self.up5(x)

up5 is ConvTranspose2d(513, 256, 2, 2), and ConvTranspose2d stores its weight as
(in_channels, out_channels, kH, kW) -- NOT (out, in, ...) like Conv2d. The ROI is
concatenated last, so the ROI's kernel is

    up5.weight[512]     shape (256, 2, 2), 1024 parameters

and up5.weight[:512] are the 512 bottleneck feature channels, which give a
same-layer, same-scale reference for what "a channel this layer attends to"
looks like. That reference is the point: an absolute weight norm means nothing
on its own, but roi_w_l2 / feat_w_l2_mean is scale-free -- 1.0 says the ROI
channel is weighted like an average feature channel, and a collapse toward 0
says the layer has learned to gate it off.

WEIGHT NORM ALONE IS NOT PROOF. The contribution a channel makes scales with
weight norm TIMES input scale, and the two arms differ in input scale: a real
ROI is binary {0,1} averaging ~0.10, while the ablation feeds N(0,1). A noise
arm could show a smaller weight norm purely because its input is ~10x larger in
scale, with the product unchanged. So the probe also records the std of the ROI
channel as it actually enters up5, and their product, which is the number that
is comparable across arms.

Gradient accumulation means the weights are identical across the micro-batches
of one optimiser step, so with accum_steps > 1 consecutive points repeat. That
is harmless -- the trace is still correct, just oversampled.

Usage, from a training driver:

    from roi_weight_probe import attach
    handle = attach(model, trainer.writer)      # before trainer.train()
"""
import torch


def _resolve(model):
    """Find the layer the ROI is concatenated into, and the ROI's channel index."""
    if not getattr(model, 'use_roi', False):
        raise ValueError("model has use_roi=False -- there is no ROI channel to probe")
    conv = model.up5
    roi_ch = getattr(model, 'roi_ch', 1)
    # ROI channels are concatenated last, so they are the final roi_ch input channels.
    return conv, conv.in_channels - roi_ch


class ROIWeightProbe:
    def __init__(self, model, writer, every=1, hist_every=50, tag='roi_probe'):
        self.conv, self.roi_start = _resolve(model)
        # TensorboardWriter wraps a SummaryWriter and owns the trainer's step
        # counter. Write to the raw writer instead: borrowing the wrapper's step
        # would lag one batch (set_step runs after the forward) and calling
        # set_step here would clobber the trainer's own bookkeeping.
        self.writer = getattr(writer, 'writer', writer)
        self.every = every
        self.hist_every = hist_every
        self.tag = tag
        self.step = 0
        self.handle = None

    def __call__(self, module, inputs, output):
        if not module.training:
            return                      # validation forwards would double-count
        n = self.step
        self.step += 1
        if self.writer is None or n % self.every:
            return

        with torch.no_grad():
            w = module.weight.detach().float()
            roi_w = w[self.roi_start:]                       # (roi_ch, out, kH, kW)
            feat_w = w[:self.roi_start]                      # (512, out, kH, kW)
            roi_l2 = roi_w.flatten(1).norm(dim=1).mean()
            feat_l2 = feat_w.flatten(1).norm(dim=1).mean()
            # std of the ROI channel as it actually arrives at this layer
            x = inputs[0]
            roi_in = x[:, self.roi_start:].detach().float()
            in_std = roi_in.std()

            s = self.writer.add_scalar
            s(f'{self.tag}/roi_w_l2', roi_l2.item(), n)
            s(f'{self.tag}/feat_w_l2_mean', feat_l2.item(), n)
            s(f'{self.tag}/roi_rel', (roi_l2 / feat_l2.clamp_min(1e-12)).item(), n)
            s(f'{self.tag}/roi_w_absmax', roi_w.abs().max().item(), n)
            s(f'{self.tag}/roi_in_std', in_std.item(), n)
            # weight norm x input scale: the cross-arm comparable contribution
            s(f'{self.tag}/roi_effective', (roi_l2 * in_std).item(), n)

            if self.hist_every and n % self.hist_every == 0:
                self.writer.add_histogram(f'{self.tag}/roi_kernel', roi_w.reshape(-1), n)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def attach(model, writer, every=1, hist_every=50, tag='roi_probe'):
    """Register the probe as a forward hook on up5. Returns the probe."""
    probe = ROIWeightProbe(model, writer, every=every, hist_every=hist_every, tag=tag)
    probe.handle = probe.conv.register_forward_hook(probe)
    return probe
