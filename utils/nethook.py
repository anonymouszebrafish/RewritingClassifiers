import torch, numpy, types, copy, inspect, types, contextlib
from collections import OrderedDict, defaultdict


def subsequence(sequential, first_layer=None, last_layer=None,
        after_layer=None, upto_layer=None, single_layer=None,
        share_weights=False):
    assert ((single_layer is None) or
            (first_layer is last_layer is after_layer is upto_layer is None))
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [None if d is None else d.split('.')
            for d in [first_layer, last_layer, after_layer, upto_layer]]
    return hierarchical_subsequence(sequential, first=first, last=last,
            after=after, upto=upto, share_weights=share_weights)

def hierarchical_subsequence(sequential, first, last, after, upto,
        share_weights=False, depth=0):
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    (F, FN), (L, LN), (A, AN), (U, UN) = [
            (d[depth], (None if len(d) == depth+1 else d))
            if d is not None else (None, None)
            for d in [first, last, after, upto]]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            FR, LR, AR, UR = [n if n is None or n[depth] == name else None
                    for n in [FN, LN, AN, UN]]
            chosen = hierarchical_subsequence(layer,
                    first=FR, last=LR, after=AR, upto=UR,
                    share_weights=share_weights, depth=depth+1)
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError('Layer %s not found' % '.'.join(name))
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result