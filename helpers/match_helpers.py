import torch
import torch as ch
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import helpers.context_helpers as coh
import torch.nn.functional as F

def downscale_mask(mask, tgt_size, threshold=None):
    src_size = mask.shape[-1]
    factor = src_size // tgt_size
    assert src_size == factor * tgt_size
    pooled = F.avg_pool2d(mask, factor, stride=factor)
    if threshold is not None:
        return pooled > threshold
    else:
        return pooled

def target_weights(target_model):
    return [p for n, p in target_model.named_parameters()
            if 'weight' in n][0]
    
def edit_classifier_weights(target_model, key, val, context, 
                           niter=2001, piter=10, lr=0.05, 
                           low_rank_insert=True, low_rank_gradient=False,
                           unfold=False, mask=None):
    
    def update_callback(it, loss, pbar=None):
        if it % 50 == 0 or it == niter - 1:
            loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                         f'\tloss {loss.item():.4f}')
            if pbar:
                pbar.set_description(str(loss))
            else:
                print(loss_info)
    try:
        key, val = [d.detach() for d in [key, val]]
    except:
        val = val.detach()

    def compute_loss(mask=None):
        reps = val, target_model(key)
        if mask is not None:
            mask = downscale_mask(mask, val.shape[-1], None)
            mask = mask.sqrt()
            reps = [r * mask for r in reps]
        return torch.nn.functional.l1_loss(*reps) / len(val)

    # set up optimizer
    weight = target_weights(target_model)
    weight_orig = weight.clone()
    params = [weight]
    if low_rank_insert or low_rank_gradient:
        with torch.no_grad():
            ortho_weight = weight - projected_conv(weight, context, unfold=unfold)
            
    optimizer = torch.optim.Adam(params, lr=lr)

    pbar = tqdm(range(niter))
    for it in pbar:
        with torch.enable_grad():
            loss = compute_loss(mask)
            optimizer.zero_grad()
            loss.backward()

            if it == 0: loss_orig = loss.item()

            if low_rank_gradient:
                weight.grad[...] = projected_conv(weight.grad, context, unfold=unfold)
            optimizer.step()
            if update_callback is not None:
                update_callback(it, loss, pbar=pbar)
            if low_rank_insert and (it % piter == 0 or it == niter - 1):
                with torch.no_grad():
                    weight[...] = (
                        ortho_weight + projected_conv(weight, context, unfold=unfold))

    print("Loss (orig, final):", loss_orig, loss.item())
    print("L2 norm of weight change:", ch.norm(weight_orig - weight).item())
    
def projected_conv(weight, direction, unfold=False):
    if len(weight.shape) == 5:
        cosine_map = torch.einsum('goiyx, di -> godyx', weight, direction)
        result = torch.einsum('godyx, di -> goiyx', cosine_map, direction)
    else:
        if unfold:
            direction_r = direction.unsqueeze(2).unsqueeze(3)
            direction_r = direction_r.reshape(direction_r.shape[0],
                                               weight.shape[1],
                                               weight.shape[2],
                                               weight.shape[3]).transpose(2, 3)
            cosine_map = torch.einsum('oiyx, diyx -> od', weight, direction_r)
            result = torch.einsum('od, diyx -> oiyx', cosine_map, direction_r)
        else:
            cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
            result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    return result
