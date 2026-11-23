"""Recompute paired claims from the ledgers, validate archived controls, and plot the measured effects."""
from collections import defaultdict
import gzip
import json
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from boundary_refinement.models.restoration import bounded_tone as color
from boundary_refinement.training import luminance_blend as calibrated
from boundary_refinement.evaluation.semantic import evaluate_prototype_risk as evaluate
from boundary_refinement.evaluation.paired import muses_transmission as joint
from boundary_refinement.data.cohorts import mechanism_holdouts as held
from boundary_refinement.training import prototype_risk as proto
from boundary_refinement.models.operators import sapa as published
from boundary_refinement.models.restoration import road_tone_control as road
from boundary_refinement.models.operators import donor_colors as method
from boundary_refinement.data.staging import stage_vkitti_fog as vkitti
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.training.train_regional_guidance import atomic_write


ROOT = calibrated.ROOT
PARTS = ("new","external","rain","snow","virtual")


def compressed(path):
    with gzip.open(path,"rt",encoding="utf8") as stream:
        return json.load(stream)


def close(first,second):
    if isinstance(first,dict):
        if first.keys()!=second.keys():
            raise ValueError("Evidence keys differ")
        for key in first:
            close(first[key],second[key])
    elif isinstance(first,list):
        if len(first)!=len(second):
            raise ValueError("Evidence lists differ")
        for x,y in zip(first,second,strict=True):
            close(x,y)
    elif isinstance(first,(int,float)):
        np.testing.assert_allclose(first,second,rtol=1e-10,atol=1e-10)
    elif first!=second:
        raise ValueError("Evidence value differs")


def permutation_color(rows, comparator):
    data=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r["ds"]=="rrshid_new" and r["arm"] in ("luma",comparator):
            data[r["group"]][r["arm"]].append(r["deltaE"])
    x=np.array([mean(g["luma"])-mean(g[comparator]) for g in data.values()])
    generator=np.random.default_rng(99031)
    trials, extreme=65536,0
    for _ in range(trials//512):
        signs=2*generator.integers(2,size=(512,len(x)))-1
        extreme+=int((np.abs((signs*x).mean(-1)) >= abs(x.mean())-1e-12).sum())
    return (extreme+1)/(trials+1)


def permutation_sem(rows, comparator):
    parts=defaultdict(lambda:np.zeros((2,3,19,19),np.int64))
    for r in rows:
        if r["method"] in ("fixed",comparator):
            parts[r["group"]][int(r["method"]==comparator),proto.SEEDS.index(r["seed"])]+=r["edge"]
    n=len(parts)
    if n!=13:
        raise ValueError("Prespecified recording inference requires the 13 reserved groups")
    a,b=np.stack(list(parts.values())).transpose(1,0,2,3,4)
    def iou(cm):
        cm=cm.reshape(-1,3,19,19)
        d=cm.diagonal(axis1=-2,axis2=-1)
        denom=cm.sum(-1)+cm.sum(-2)-d
        return np.divide(d,denom,out=np.zeros_like(d,dtype=float),where=denom>0).mean((1,2))
    x,y=a.sum(0).ravel(),b.sum(0).ravel()
    observed=abs(float((iou(x)-iou(y))[0]))
    diff=(a-b).reshape(n,-1)
    extreme=0
    for i in range(0,2**n,128):
        flip=(np.arange(i,i+128)[:,None]>>np.arange(n))&1
        change=flip@diff
        extreme+=int((np.abs(iou(x-change)-iou(y+change))>=observed-1e-12).sum())
    return extreme/(2**n)


def figure(tone, road_data, mechanism):
    fig, axes=plt.subplots(1,2,figsize=(10.8,5.3),layout="constrained")
    def forest(ax, rows, xlabel):
        for i,(label,center,ci,kind) in enumerate(rows):
            y=len(rows)-1-i
            ink="#186d71" if kind=="rgb" else "#6f4d97"
            if ci is not None:
                ax.plot(ci,[y,y],color=ink,lw=2)
            ax.plot(center,y,"o",color=ink,ms=5)
        ax.axvline(0,color="#555555",lw=.8)
        ax.set_yticks(range(len(rows)),[r[0] for r in rows][::-1])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x",color="#dddddd",lw=.6)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    rows=[]
    for key,label in (("rrshid_new","RRSHID, new"),("hrsd_dhid","HRSD-DHID"),("hrsd_lhid","HRSD-LHID"),
                      ("kitti_light","KITTI light"),("kitti_dense","KITTI dense"),("nh_holdout","NH-HAZE"),
                      ("dense","Dense-Haze"),("ohaze","O-HAZE"),("i_haze","I-HAZE")):
        card=(tone if key in tone else road_data)[key]["effects"]["deltaE"]["luma-affine"]
        rows.append((label,card["equal_group_mean"],card["ci95"],"rgb"))
    forest(axes[0],rows,"Window minus affine (CIEDE2000; lower is better)")
    axes[0].set_title("Photometric effect by domain",loc="left",weight="bold")
    rows=[]
    names=(("foggy_driving_fine","Foggy Driving Fine"),("foggy_zurich_testv2","Foggy Zurich (one video)"),
           ("muses_rain","MUSES rain"),("muses_snow","MUSES snow"),("vkitti_fog","Virtual KITTI fog"),
           ("new_muses_clear","MUSES clear, new"))
    for key,label in names:
        card=mechanism[key]["effects"]["fixed-mean"]
        rows.append((label,card["diff"],card["ci95"],"sem"))
    forest(axes[1],rows,"Class-weighted minus mean (boundary mIoU, pp)")
    axes[1].set_title("Donor color effect with frozen decoder",loc="left",weight="bold")
    fig.savefig(ROOT/"mechanism-evidence.png",dpi=185,bbox_inches="tight")
    plt.close(fig)


def run():
    rgbsets={}
    for label,folder,func,name in (("mcaf",color.ROOT,color.summarize,""),("road",road.ROOT,color.summarize,""),
                                  ("calibrated",ROOT,calibrated.summary,""),("external",ROOT,calibrated.summary,"external-")):
        doc=read(folder/(name+"test.json"))
        file=folder/(name+"individual.json")
        if digest(file)!=doc["ledger"]:
            raise ValueError("Photometric ledger changed")
        rows=read(file)
        recomputed=func(rows)
        close(recomputed,doc["sets"])
        rgbsets[label]=recomputed
        if label=="mcaf":
            prob={"new_RRSHID_deltaE_luma_vs_"+b:permutation_color(rows,b) for b in ("affine","rolled")}
    checkpoints=0
    for folder,array in ((color.ROOT,"records"),(road.ROOT,"runs"),(proto.ROOT,"runs")):
        notes=read(folder/"completed.json")
        for r in notes[array]:
            arm=r.get("arm",r.get("method"))
            if r["weights"]!=digest(folder/f"{arm}-{r['seed']}.pth") or r["protocol"]!=digest(folder/"protocol.json"):
                raise ValueError("Checkpoint or protocol mismatch")
            checkpoints+=1
    edges, replay_totals={},{}
    for part in PARTS:
        doc=read(proto.ROOT/f"{part}-test.json")
        file=proto.ROOT/f"{part}-individual.json.gz"
        if digest(file)!=doc["ledger"] or doc["evaluator"]!=digest(evaluate.__file__):
            raise ValueError("Semantic ledger changed")
        rows=compressed(file)
        ids=vkitti.CLASSES if part=="virtual" else range(19)
        close(evaluate.scores(rows,ids,"counts"),doc["full"])
        close(evaluate.scores(rows,ids,"edge"),doc["edge"])
        close(evaluate.diagnostics(rows),doc["diagnostics"])
        edges.update(doc["edge"])
        if part=="new":
            for b in ("mean","rolled"):
                prob["new_MUSES_clear_boundary_fixed_vs_"+b]=permutation_sem(rows,b)
        else:
            newer={(r["ds"],r["id"],r["seed"],r["method"]):r for r in rows}
            names={"dual_core":"fixed","bilinear":"bilinear","dysample":"dysample","sapa_context":"sapa_context"}
            matches=0
            for path in (method.ROOT/f"{part}-individual.json.gz",published.ROOT/f"{part}-additional.json.gz"):
                for old in compressed(path):
                    if old["method"] not in names:
                        continue
                    row=newer[old["dataset"],old["id"],old["seed"],names[old["method"]]]
                    if row["counts"]!=old["counts"] or row.get("edge")!=old.get("edge"):
                        raise ValueError(f"Frozen prediction differed from archived {old['method']}")
                    matches+=1
            if matches!=sum(doc["counts"].values())*len(proto.SEEDS)*len(names):
                raise ValueError("Missing archived native confusion matrices")
            replay_totals[part]=matches
    factorial=read(ROOT/"joint-test.json")
    joint_rgb,joint_seg=ROOT/"joint-rgb.json",ROOT/"joint-seg.json.gz"
    if digest(joint_rgb)!=factorial["rgb_table"] or digest(joint_seg)!=factorial["cm_table"]:
        raise ValueError("Dual-output ledger changed")
    close(calibrated.summary(read(joint_rgb)),factorial["photos"])
    cm=compressed(joint_seg)
    for k in ("edge","full"):
        close(joint.aggregate(cm,"counts" if k=="full" else "edge"),factorial[k])
    p_order=sorted(prob,key=prob.get)
    corrected=np.maximum.accumulate([(len(prob)-i)*prob[k] for i,k in enumerate(p_order)]).clip(max=1.)
    example=method.ROOT/"examples/foggy_driving_fine"
    replay=ROOT/"replay-luma"
    for a,b in (("mask_train_ids.png","dual_core.png"),("restored.png","restored.png")):
        with Image.open(replay/a) as x, Image.open(example/b) as y:
            np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    with Image.open(ROOT/"real-fog-example/mask_train_ids.png") as seg, Image.open(ROOT/"real-fog-example/restored.png") as rgb:
        if seg.size!=rgb.size or int(np.max(np.asarray(seg)))>=19:
            raise ValueError("New native output has unexpected size or classes")
    figure(rgbsets["mcaf"],rgbsets["road"],edges)
    atomic_write(ROOT/"verification.json",dict(completed_network_fits=checkpoints,
        optimizer_updates=sum(read(r/"completed.json")["steps"] for r in (proto.ROOT,road.ROOT,color.ROOT)),
        constrained_calibrations=len(calibrated.load()),replayed_archived_predictions=replay_totals,
        fresh_selection=[digest(held.TILES),digest(held.SCENES)],joint_predictions=len(cm),example_size=list(seg.size),
        monte_carlo_and_exact_two_sided_p=prob,holm_four_mechanism_tests=dict(zip(p_order,corrected.tolist())),
        interpretation="Paired randomization sensitivity on the new RRSHID tiles and MUSES clear recordings. "
                       "RRSHID tile locations are unavailable. No regional/site independence is asserted. "
                       "Seeds averaged within sampling units; four-comparison Holm adjustment. Legacy cohorts descriptive.",
        label_ties="Prototype targets now use torch.argmax, the identical tie convention as the rendering weights. "
                   "All nine gate runs and every reported V97 split replayed after the correction.",
        protocol=[digest(p/"protocol.json") for p in (color.ROOT,road.ROOT,proto.ROOT,ROOT)],utc=now()))


if __name__=="__main__":
    run()
