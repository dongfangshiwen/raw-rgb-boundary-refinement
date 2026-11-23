"""Measure the RGB-to-native-mask paths used in Table 3b on one fog frame per drive."""
import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch


ARMS = ("bilinear","fixed","original")
TARGETS = ("mask2former","segformer")
ROUNDS = 6
WARMUP = 12
NATIVE = (1536,2048)
SIDE = (384,512)


def write(path, obj):
    part=path.with_suffix(".tmp")
    part.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n","utf8")
    part.replace(path)


def coarse(net,image,ev):
    x=(image-net.mean)/net.std
    with torch.autocast("cuda",dtype=torch.bfloat16):
        if net.name=="segformer":
            return net.net(pixel_values=x).logits.float()
        pm=torch.ones((1,*image.shape[-2:]),device=image.device,dtype=torch.long)
        q=net.net(pixel_values=x,pixel_mask=pm)
    coarse_size=q.masks_queries_logits.shape[-2:]
    return ev.setup.peer.query_to_scores(q.class_queries_logits,q.masks_queries_logits,coarse_size,coarse_size)


def native_mask(net,image,arm,fixed,original,ev):
    scores=coarse(net,image,ev)
    base=ev.resize(scores,NATIVE)
    if arm=="bilinear":
        return base.argmax(1)[0].byte()
    guide=image.clamp(0,1).mul(255).round().byte()
    if arm=="fixed":
        delta=fixed.correction(scores,ev.prepare(scores,guide))
    else:
        stencil=ev.legacy.guide_stencil(guide[[0,0]],scores.argmax(1)[0])
        delta=ev.legacy.prediction("dual_free",original,scores,stencil)
    return (base+ev.resize(delta,NATIVE)).argmax(1)[0].byte()


def working_image(rgb,ev):
    batch=torch.from_numpy(rgb).cuda().permute(2,0,1)[None].float()/255.
    return ev.resize(batch,SIDE,antialias=True).contiguous()


def check_and_prepare(net,rows,indices,fixed,original,ev,baseline_path):
    host=[]
    with np.load(baseline_path,allow_pickle=False) as packet:
        reference=packet["counts"]
        assert int(packet["complete"])==146 and int(packet["seeds"][0])==101
        assert list(packet["arms"][[1,2,6]])==["no_correction","fixed","paper"]
        for row,i in zip(rows,indices):
            rgb,truth=ev.scene(row,"idd")
            img=working_image(rgb,ev)
            metric=ev.ev.Counts(truth)
            for k,arm in zip((1,2,6),ARMS):
                pred=native_mask(net,img,arm,fixed,original,ev)
                np.testing.assert_array_equal(np.stack(metric.score(pred)),reference[i,:,k,0],
                                              err_msg=net.name+" "+row["id"]+" "+arm)
            host.append(img.cpu())
    return host


def timed(net,images,fixed,original,ev):
    for step in range(WARMUP):
        image=images[step%len(images)].cuda()
        for arm in ARMS:
            del_prediction=native_mask(net,image,arm,fixed,original,ev)
            del del_prediction
        del image
    torch.cuda.synchronize()
    records={a:{"elapsed_ms":[],"peak_bytes":[],"resident_bytes":[]} for a in ARMS}
    orders=tuple(itertools.permutations(ARMS))
    for i,host in enumerate(images):
        image=host.cuda()
        for rep in range(ROUNDS):
            for arm in orders[(i+rep)%len(orders)]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident=torch.cuda.memory_allocated()
                start=perf_counter()
                pred=native_mask(net,image,arm,fixed,original,ev)
                torch.cuda.synchronize()
                elapsed=(perf_counter()-start)*1000
                peak=torch.cuda.max_memory_allocated()
                assert pred.dtype==torch.uint8 and pred.shape==NATIVE and peak>=resident
                records[arm]["elapsed_ms"].append(elapsed)
                records[arm]["peak_bytes"].append(int(peak))
                records[arm]["resident_bytes"].append(int(resident))
                del pred
        del image
    return records


def aggregate(raw):
    result={}
    for arm,data in raw.items():
        ms=np.asarray(data["elapsed_ms"]); peak=np.asarray(data["peak_bytes"])
        assert len(ms)==len(peak)==24*ROUNDS and np.isfinite(ms).all() and np.all(ms>0)
        result[arm]={"mean_ms":float(ms.mean()),"median_ms":float(np.median(ms)),
                     "p95_ms":float(np.percentile(ms,95)),"serial_fps":float(1000/ms.mean()),
                     "peak_mib":float(peak.max()/2**20)}
    for row in result.values():
        row["additional_peak_mib_vs_bilinear"]=row["peak_mib"]-result["bilinear"]["peak_mib"]
        row["additional_ms_vs_bilinear"]=row["mean_ms"]-result["bilinear"]["mean_ms"]
    return result


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from scripts import eval_range_v101 as ev
    ev.setup.configure_cuda()
    args.out.mkdir(parents=True,exist_ok=True)
    rows_by_id={r["id"]:r for r in ev.read(ev.IDD/"evaluation.json")["rows"] if "fog" in r["cohorts"]}
    with np.load(args.audit/"counts-mask2former.npz",allow_pickle=False) as pkt:
        ids=pkt["ids"].tolist()
        groups=pkt["groups"].tolist()
    selected=[min((j for j in range(len(ids)) if groups[j]==g),key=ids.__getitem__) for g in sorted(set(groups))]
    rows=[rows_by_id[ids[j]] for j in selected]
    assert len(rows)==24 and len(set(r["group"] for r in rows))==24
    protocol={"choices":"First frame ID in lexicographic order per previously evaluated IDD-AW fog recording. Same 24 images, all three methods, both backbones, final MUSES seed 101. Six permutations balance per-image execution order.",
              "measurement":"Batch one, single-stream synchronized wall time from already resized GPU RGB through backbone normalization, backbone forward, decoder and uint8 native mask on GPU. Excludes disk access, host transfer, initial resize, photographic branch and mask transfer to CPU. No simultaneous branch execution.",
              "size":{"preloaded_rgb":list(SIDE),"output_mask":list(NATIVE)},"scope":"Measured semantic branch only. Same coarse score bilinear is the common comparator.",
              "memory":"Maximum PyTorch allocated GPU bytes across warm timing trials, including model, inputs and live activations; excluding allocator reservations. Extra peak equals the maximum for the method minus the bilinear maximum on the same backbone.",
              "summaries":"Arithmetic mean ms per individual frame and 1000 / mean ms as serial FPS; median and 95th percentile included. Descriptive measurement on this installation.",
              "warmups_per_arm_per_target":WARMUP,"repeats_per_frame":ROUNDS,"batch":1,
              "frames":[{"id":r["id"],"group":r["group"]} for r in rows],"torch":str(torch.__version__),
              "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"script_sha256":sha256(Path(__file__).read_bytes()).hexdigest()}
    write(args.out/"timing-protocol.json",protocol)
    all_fits=ev.original.load()
    original=all_fits["dual_core",101]
    del all_fits
    fixed=ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    targets={}
    for target in TARGETS:
        net=ev.Target(target)
        images=check_and_prepare(net,rows,selected,fixed,original,ev,args.audit/f"counts-{target}.npz")
        torch.cuda.empty_cache()
        raw=timed(net,images,fixed,original,ev)
        targets[target]={"measurements":raw,"summary":aggregate(raw),"validated_native_masks":len(rows)*len(ARMS)}
        del net,images
        torch.cuda.empty_cache()
        print(target,targets[target]["summary"],flush=True)
    write(args.out/"timing-results.json",{"protocol_sha256":sha256((args.out/"timing-protocol.json").read_bytes()).hexdigest(),
         "targets":targets})


if __name__=="__main__":
    main()
