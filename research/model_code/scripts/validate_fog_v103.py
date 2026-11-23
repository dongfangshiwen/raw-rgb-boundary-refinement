"""Frozen inference on BDD fog clips, WeatherProof pairs and the historical IDD controls."""
from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
from PIL import Image
import torch
from scripts import directional_veto_v103 as route
from scripts import eval_native_guide_v102 as base
from scripts import native_gaussian_v102 as gaussian
from scripts import train_directional_v103 as fit
from scripts.native_guide_v102 import PixelGrid
from scripts.restoration_affinity_v75 import boundary_band
from scripts.train_regional_guidance_v18 import configure_cuda
from scripts.projective_ledger_v12 import digest, now
from scripts.weatherproof_release_v103 import OUT, WORK, read, write

ARMS = ("released","paper","segfix","gaussian","gaussian_segfix","paper_segfix",*route.MODES)
LAST = WORK/"experiment_exports/native-guide-segfix-2026-11-20"
PREVIOUS = ("released","paper_512","segfix","matched_gaussian_fixed","gaussian_then_segfix","paper_then_segfix")


def open_frame(row):
    if row["cohort"]=="fog":
        rgb, truth = base.load_image(row)
        return rgb, [truth.cpu().numpy().astype(np.uint8)]
    paths = [WORK / p for p in row["paths"]]
    assert list(map(digest,paths)) == row["sha256"]
    with Image.open(paths[0]) as im, Image.open(paths[-1]) as mask:
        rgb, labels = np.array(im.convert("RGB")), np.array(mask)
    if labels.ndim==3:
        assert np.array_equal(labels[:,:,0],labels[:,:,1]) and np.array_equal(labels[:,:,0],labels[:,:,2])
        labels=labels[:,:,0]
    assert rgb.shape[:2]==labels.shape
    if row["cohort"]=="bdd_dense":
        assert set(np.unique(labels))<=set(range(19))|{255}
        return rgb,[labels]
    lut = np.full(256,255,np.uint8)
    if row["cohort"]=="bdd":
        lut[[0,1,2]]=[0,1,1]
        return rgb,[lut[labels]]
    lut[[1,3,4]]=[2,0,10]
    strict=lut[labels]
    lut[9]=8
    return rgb,[strict,lut[labels]]


def native_predictions(model, small, raw, lookup, offset, old, decisions):
    shape = raw.shape[-2:]
    z, released = model(small, shape)
    ref = released.argmax(1)[0].byte()
    del released
    native = base.resize(z,shape)
    g = native.argmax(1)[0].byte()
    grid=PixelGrid(z,small,raw)
    for ix in grid.active().nonzero().flatten().split(32768):
        if len(ix):
            g.view(-1)[ix]=gaussian.scores(grid.four(ix,include_distance=True)).argmax(-1).byte()
    warp=lambda m:m.flatten()[lookup].reshape(shape)
    masks={"released":[ref],"segfix":[warp(ref)],"gaussian":[g],"gaussian_segfix":[warp(g)]}
    stencil=route.paper_stencil(z,small)
    colors=route.DirectionalColors(raw)
    for s in route.SEEDS:
        scores=route.paper_scores(z,stencil,old["dual_core",s],native)
        paper, moved, ids, features=route.conflicts(scores,lookup,offset,colors)
        del scores
        masks.setdefault("paper",[]).append(paper)
        masks.setdefault("paper_segfix",[]).append(moved)
        for mode in route.MODES:
            masks.setdefault(mode,[]).append(decisions[mode,s].decide(paper,moved,ids,features))
    return {name:[v.cpu().numpy() for v in values] for name,values in masks.items()}


class SparseCounts:
    def __init__(self, truth):
        self.truth=truth.reshape(-1).copy()
        self.band=boundary_band(torch.from_numpy(truth.copy()).cuda().long()).cpu().numpy().ravel()
        self.valid=self.truth<19
        self.rows=self.truth.astype(np.int32)*19 + self.band.astype(np.int32)*361

    def cm(self, pred):
        a=np.bincount(self.rows[self.valid]+pred[self.valid],minlength=722).reshape(2,19,19)
        return np.stack((a.sum(0),a[1]))

    def change(self, paper, candidate, baseline):
        ix=np.flatnonzero(paper!=candidate)
        ix=ix[self.valid[ix]]
        old, new, y=paper[ix],candidate[ix],self.truth[ix]
        diff=(np.bincount(self.rows[ix]+new,minlength=722)-np.bincount(self.rows[ix]+old,minlength=722)).reshape(2,19,19)
        scores=baseline+np.stack((diff.sum(0),diff[1]))
        flags=np.where(new==y,0,np.where(old==y,1,2))
        val=np.bincount(self.band[ix].astype(np.int32)*3+flags,minlength=6).reshape(2,3)
        delta=np.stack((val.sum(0),val[1]))
        assert np.array_equal(delta[:,0]-delta[:,1],np.trace(scores-baseline,axis1=-2,axis2=-1))
        return scores,delta


def packet(masks, scorer, binary):
    def view(m):
        flat=m.ravel()
        return (flat==0).astype(np.uint8) if binary else flat
    mapped={k:[view(m) for m in v] for k,v in masks.items()}
    parts=[]; corrections=[]
    for k in range(len(route.SEEDS)):
        paper=mapped["paper"][k]
        cm=scorer.cm(paper)
        result=[scorer.change(paper,mapped[a][k%len(mapped[a])],cm) for a in ARMS]
        counts, edits=zip(*result)
        parts.append(np.stack(counts,axis=1)); corrections.append(np.stack(edits,axis=1))
    counts=np.stack(parts,axis=2).astype(np.int32)
    assert (counts>=0).all() and (counts[1]<=counts[0]).all() and (counts.sum(-1)==counts[:,:1,:1].sum(-1)).all()
    return counts,np.stack(corrections,axis=2).astype(np.int32)


def save_examples(row,model,masks,rgb,truth,folder):
    d=OUT/"examples"/folder/model
    d.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(d/"input.png")
    Image.fromarray(truth).save(d/"truth-ids.png")
    for method in ARMS:
        Image.fromarray(masks[method][0]).save(d/(method+"-city-ids.png"))
    write(d/"source.json",dict(id=row["id"],group=row["group"],cohort=row["cohort"],seed=route.SEEDS[0],
         label_mapping="For BDD, predicted Cityscapes ID 0 corresponds to truth foreground 1; for other sets truth and prediction use Cityscapes IDs. Void is 255."))


@torch.inference_mode()
def run():
    configure_cuda()
    plans=["protocol.json","training.json","bdd-manifest.json","bdd-visual-lock.json","wp-manifest.json","wp-weather-lock.json","mapping-correction.json"]
    audit=read(OUT/"mapping-correction.json")
    assert len(audit["files"])==135 and audit["total_pixels"]==135*1280*720
    lock=sha256(("".join(digest(OUT/n) for n in plans)+digest(__file__)+digest(route.__file__)+base.SHA).encode()).hexdigest()
    models={t:base.prior.Target(t) for t in base.TARGETS}
    segfix=base.SegFix();old=base.original.load();decisions=fit.load()
    visible=set(read(OUT/"bdd-visual-lock.json")["ids"])
    prior=read(LAST/"composition/evaluation.json")
    rows=[*read(OUT/"bdd-manifest.json")["rows"],*read(OUT/"wp-manifest.json")["rows"],
          *[r for r in read(LAST/"manifest.json")["rows"] if r["cohort"]=="fog"]]
    examples={}
    for r in rows:
        if r["cohort"]=="bdd" and r["id"] in visible: examples.setdefault("bdd",r["id"])
        if r["cohort"]=="fog": examples.setdefault("idd",r["id"])
        if r["cohort"]=="wp" and r["weather"]=="fog" and "wp" not in examples:
            _, masks=open_frame(r)
            if SparseCounts(masks[0]).band.any():examples["wp"]=r["id"]
    tic=monotonic()
    for i,row in enumerate(rows):
        cohort=row["cohort"]; key=row["id"].replace("/","_")+".npz"
        dest=OUT/"counts"/cohort/key
        if dest.exists():
            with np.load(dest,allow_pickle=False) as d:
                assert d["id"].item()==row["id"]
                if d["lock"].item()==lock:continue
                assert d["lock"].item()==audit["invalid_previous_evaluation"]
        rgb,truths=open_frame(row)
        raw=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255
        resized,(h,w)=base.peer_input(rgb)
        small=resized[...,:h,:w].contiguous()
        if cohort=="wp":
            # Both target backbones use an identical stride-aligned analysis raster for arbitrary archive sizes.
            target=tuple(max(32,min(512,32*round(d/32))) for d in (h,w))
            small=base.resize(raw,target,antialias=True)
        scorers=[SparseCounts(truth) for truth in truths]
        lookup,offset=segfix.lookup(rgb,True)
        mats=[];changes=[]
        for model_name,model in models.items():
            masks=native_predictions(model,small,raw,lookup,offset,old,decisions)
            scores,edits=zip(*(packet(masks,sc,cohort=="bdd") for sc in scorers))
            mats.append(np.stack(scores));changes.append(np.stack(edits))
            for folder,chosen in examples.items():
                if row["id"]==chosen:save_examples(row,model_name,masks,rgb,truths[0],folder)
        counts=np.stack(mats); edits=np.stack(changes)
        if dest.exists() and cohort!="bdd":
            with np.load(dest,allow_pickle=False) as old_packet:
                np.testing.assert_array_equal(old_packet["counts"],counts)
                np.testing.assert_array_equal(old_packet["edits"],edits)
        if cohort=="fog":
            with np.load(LAST/"composition/counts/fog"/key,allow_pickle=False) as previous:
                assert previous["lock"].item()==prior["lock"]
                np.testing.assert_array_equal(counts[:,0,:,:6],np.take(previous["counts"],[prior["arms"].index(k) for k in PREVIOUS],axis=2))
        dest.parent.mkdir(parents=True,exist_ok=True)
        tmp=dest.with_suffix(".part")
        with tmp.open("wb") as fp:
            np.savez_compressed(fp,id=row["id"],group=row["group"],counts=counts,edits=edits,lock=lock,
                                arms=ARMS,targets=base.TARGETS,seeds=route.SEEDS,native_size=rgb.shape[:2],analysis_size=small.shape[-2:])
        tmp.replace(dest)
        if i==0 or (i+1)%25==0 or i+1==len(rows):
            print("Validated",i+1,len(rows),cohort,round(monotonic()-tic,1),flush=True)
    write(OUT/"evaluation.json",dict(lock=lock,arms=ARMS,targets=base.TARGETS,seeds=route.SEEDS,files=len(rows),
          example_ids=examples,historical_idd_baselines_verified=6,inputs={n:digest(OUT/n) for n in plans},
          time=now(),seconds=monotonic()-tic))

if __name__=="__main__":
    run()
