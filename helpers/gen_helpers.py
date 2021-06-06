import os, torch
import torch as ch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from robustness.attack_steps import L2Step
import helpers.math_helpers as math

def get_all_images(loader):
    all_ims, all_targs = [], []
    for _, (im, targ) in tqdm(enumerate(loader), total=len(loader)):
        all_ims.append(im)
        all_targs.append(targ)
    return ch.cat(all_ims), ch.cat(all_targs)

def get_preds(model, imgs, BS=50, use_tqdm=False):
    top_preds = []
    
    batch_count = int(np.ceil(len(imgs) / BS))
    with ch.no_grad():
        it = range(batch_count)
        if use_tqdm:
            it = tqdm(it)
        for bc in it:
            op = model(imgs[bc*BS:(bc+1)*BS].cuda())
            pred = ch.argmax(op, dim=1)
            top_preds.append(pred.detach().cpu())
            
    return ch.cat(top_preds)
    
def eval_accuracy(model, loader, workers=10, batch_size=128, cache_file=None):
    if cache_file and os.path.exists(cache_file):
        df = ch.load(cache_file)
        targets, preds = df['targets'], df['preds']
        
    else:
        targets, preds = [], []
        for _, (im, targ) in tqdm(enumerate(loader), total=len(loader)):
            with ch.no_grad():
                op = model(im.cuda())
                pred = op.argmax(dim=1)
                targets.append(targ)
                preds.append(pred.cpu())

        targets, preds = ch.cat(targets), ch.cat(preds)
        
        if cache_file:
            ch.save({'targets': targets, 'preds': preds}, cache_file)
    
    acc = 100 * ch.mean(targets.eq(preds).float()).item()
    print(f"Accuracy: {acc}")
    
    return targets.cpu(), preds, acc