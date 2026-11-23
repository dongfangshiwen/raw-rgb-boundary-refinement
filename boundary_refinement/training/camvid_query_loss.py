"""Known-pixel set-loss adaptation for a future author-recipe control.

Uses 2/5/5 classification/BCE/Dice, 0.1 no-object weight, and the author
point-count/oversampling defaults. Known pixel centers replace continuous
point sampling so void cannot silently count as background. This module is
not used by the frozen V17 common-loss comparison.
"""
import numpy as np
import torch
import torch.nn.functional as F


def unique_query_assignment(cost):
    """Minimum-cost full matching from targets to distinct queries, O(n²q)."""
    if cost.ndim != 2 or cost.shape[0] < cost.shape[1]:
        raise ValueError('Need [queries >= targets, targets] costs')
    costs = cost.detach().cpu().double().numpy().T
    if not np.isfinite(costs).all():
        raise ValueError('Assignment costs must be finite')
    n, q = costs.shape
    # Rectangular shortest augmenting path with dual row/column potentials.
    row_dual = np.zeros(n+1); col_dual = np.zeros(q+1)
    matching = np.zeros(q+1, dtype=np.int64)
    previous = np.zeros(q+1, dtype=np.int64)
    for row in range(1,n+1):
        matching[0] = row
        j = 0
        seen = np.zeros(q+1,dtype=bool)
        slack = np.full(q+1,np.inf)
        while True:
            seen[j] = True
            root = matching[j]
            cols = np.flatnonzero(~seen)
            reduced = costs[root-1,cols-1]-row_dual[root]-col_dual[cols]
            update = reduced < slack[cols]
            changed = cols[update]
            slack[changed] = reduced[update]
            previous[changed] = j
            next_j = cols[np.argmin(slack[cols])]
            delta = slack[next_j]
            row_dual[matching[seen]] += delta
            col_dual[seen] -= delta
            slack[cols] -= delta
            j = next_j
            if matching[j]==0:
                break
        while j:
            k=previous[j];matching[j]=matching[k];j=k
    queries = np.empty(n,dtype=np.int64)
    for j,row in enumerate(matching[1:]):
        if row:queries[row-1]=j
    return torch.as_tensor(queries,device=cost.device)


def read_points(mask_logits, flat_idx, height, width):
    y,x = flat_idx//width,flat_idx%width
    coordinates = torch.stack(((x.float()+.5)*(2/width)-1, (y.float()+.5)*(2/height)-1), dim=-1)
    return F.grid_sample(mask_logits[:,None],coordinates[:,:,None],align_corners=False)[:,0,:,0]


def known_pixel_query_loss(classes, masks, truth, *, points=12544, generator=None):
    """One Mask2Former decoder stage: class+known-pixel BCE and Dice."""
    if classes.ndim!=3 or masks.ndim!=4 or truth.ndim!=3 or classes.shape[:2]!=masks.shape[:2] or len(truth)!=len(masks):
        raise ValueError('Expected BQC, BQHW, BHW')
    n_classes=classes.shape[-1]-1
    if classes.shape[1]<n_classes or n_classes<2 or points<4:
        raise ValueError('Unsuitable query count or sampler')
    h,w=truth.shape[-2:]
    class_labels=[];bce=[];dice=[];items=[]
    with torch.autocast(classes.device.type,enabled=False):
        cls=classes.float();mask=masks.float()
        def sample(valid,shape):
            return valid[torch.randint(len(valid),shape,device=valid.device,generator=generator)]
        for i in range(len(truth)):
            flat=truth[i].reshape(-1);known=flat!=255;valid=known.nonzero()[:,0]
            if not len(valid):continue
            if torch.any((flat[valid]<0)|(flat[valid]>=n_classes)):
                raise ValueError('Target outside declared ontology')
            unique=torch.unique(flat[valid],sorted=True)
            with torch.no_grad():
                coord=sample(valid,(points,))
                logits=read_points(mask[i],coord[None].expand(cls.shape[1],-1),h,w)
                targets=(flat[coord][None]==unique[:,None]).float()
                pair_bce=F.softplus(logits).mean(-1,keepdim=True)-logits@targets.T/points
                prob=logits.sigmoid()
                pair_dice=1-(2*prob@targets.T+1)/(prob.sum(-1,keepdim=True)+targets.sum(-1)[None]+1)
                score=-2*cls[i].softmax(-1)[:,unique]+5*pair_bce+5*pair_dice
                assigned=unique_query_assignment(score)
                targets_cls=torch.full((cls.shape[1],),n_classes,dtype=torch.long,device=truth.device)
                targets_cls[assigned]=unique
            chosen=mask[i,assigned]
            with torch.no_grad():
                candidates=sample(valid,(len(unique),points*3))
                top=(-read_points(chosen,candidates,h,w).abs()).topk(int(points*.75),dim=-1).indices
                selected=torch.gather(candidates,1,top)
                selected=torch.cat((selected,sample(valid,(len(unique),points-selected.shape[-1]))),-1)
                expected=(flat[selected]==unique[:,None]).float()
            logits=read_points(chosen,selected,h,w)
            bce.append(F.binary_cross_entropy_with_logits(logits,expected,reduction='none').mean(-1))
            prob=logits.sigmoid()
            dice.append(1-(2*(prob*expected).sum(-1)+1)/(prob.sum(-1)+expected.sum(-1)+1))
            items.append(i);class_labels.append(targets_cls)
        if not items:return cls.sum()*0+mask.sum()*0
        weight=cls.new_ones(n_classes+1);weight[-1]=.1
        sce=F.cross_entropy(cls[items].transpose(1,2),torch.stack(class_labels),weight=weight)
        return 2*sce+5*torch.cat(bce).mean()+5*torch.cat(dice).mean()
