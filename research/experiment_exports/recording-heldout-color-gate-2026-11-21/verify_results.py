"""Independent metric replay and direct native-prediction check on a registered image."""
from hashlib import sha256
import argparse
import numpy as np
import torch
import gate
from fresh_clips import HERE, put, read


def pooled(cm):
    bank=cm.sum(axis=0,dtype=np.int64)
    inter=np.diagonal(bank,axis1=-2,axis2=-1)
    u=bank.sum(-2)+bank.sum(-1)-inter
    return 100*np.divide(inter,u,out=np.zeros_like(inter,dtype=float),where=u!=0).mean(-1)


def reports():
    src=read(HERE/"muses-results.json")["panels"]
    n=0
    for model,card in src.items():
        with np.load(HERE/f"counts-{model}.npz",allow_pickle=False) as f:
            assert f["finished"].item()==len(f["ids"])
            results=pooled(f["counts"])
        for s,scale in enumerate(gate.SCALES):
            for b,region in enumerate(("full","boundary")):
                for a,name in enumerate(gate.ARMS):
                    expected=card["statistics"][str(scale)][region]["arms"][name]["miou"]
                    np.testing.assert_allclose(results[s,b,a],expected,atol=1e-10);n+=1
    return n


def bdd():
    import fresh_scores
    cards=read(HERE/"bdd-results.json")["results"]
    n=0
    for target,card in cards.items():
        with np.load(HERE/f"bdd-counts-{target}.npz",allow_pickle=False) as f:
            counts,panels=f["counts"],f["panels"]
        for p in np.unique(panels):
            vals=pooled(counts[panels==p]).mean(-1)
            for s,scale in enumerate(gate.SCALES):
                for b,reg in enumerate(("full","boundary")):
                    for m,arm in enumerate(fresh_scores.ARMS):
                        expected=card["panels"][p]["scales"][str(scale)][reg]["scores"][arm]
                        np.testing.assert_allclose(vals[s,b,m],expected,atol=1e-10);n+=1
    return n


@torch.inference_mode()
def native_replay():
    gate.ev.setup.configure_cuda()
    rows=gate.source_rows()
    item=min(rows["calibrate"],key=lambda r:sha256(("native-audit-112/"+r["id"]).encode()).digest())
    model=gate.ev.Target("mask2former")
    kernel=gate.ev.RangeDecoder("bilinear_bounded4").cuda().eval().requires_grad_(False)
    rgb,y=gate.open_muses(item)
    metric=gate.ev.ev.Counts(y)
    with np.load(HERE/"counts-mask2former.npz",allow_pickle=False) as f:
        pos=f["ids"].tolist().index(item["id"]); stored=f["counts"][pos]
    w=read(HERE/"weights.json")["fits"]["all"]["aligned"]["weights"]
    for j,scale in enumerate(gate.SCALES):
        image=gate.image_tensor(rgb,scale)
        z,official=model(image,y.shape)
        guide=image.clamp(0,1).mul(255).round().byte()
        corrected=official+gate.ev.resize(kernel.correction(z,gate.ev.prepare(z,guide)),y.shape)
        a=official.argmax(1)[0]; b=corrected.argmax(1)[0]
        ids=(a!=b).flatten().nonzero().flatten()
        idx1=a.flatten()[ids];idx2=b.flatten()[ids]
        v,colour,_=gate.donor_cues(z,guide.float()/255,ids,idx1,idx2,y.shape)
        prob=torch.sigmoid(w[0]+w[1]*(-1 if scale==512 else 1)+w[2]*v.double()+w[3]*colour.double())
        m=official[0].reshape(19,-1);r=corrected[0].reshape(19,-1)
        old=(m[idx1,ids]-m[idx2,ids]).clamp_min(0)
        new=(r[idx2,ids]-r[idx1,ids]).clamp_min(0)
        cut=(old/(old+new).clamp_min(1e-20)).clamp(1e-5,1-1e-5)
        out=a.clone()
        accept=prob>cut
        out.flatten()[ids[accept]]=idx2[accept]
        for arm,map_ in (("official",a),("fixed",b),("aligned",out)):
            np.testing.assert_array_equal(np.stack(metric.score(map_)),stored[j,:,gate.ARMS.index(arm)])
    return dict(id=item["id"],shape=list(y.shape),scales=list(gate.SCALES),target="mask2former",exact_confusions=12)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--external",action="store_true");args=ap.parse_args()
    if args.external:
        earlier=read(HERE/"verification.json")
        earlier["external_miou_entries"]=bdd()
        put(HERE/"verification.json",earlier)
    else:
        report=dict(muses_miou_entries=reports(),native=native_replay())
        put(HERE/"verification.json",report)
    print(read(HERE/"verification.json"),flush=True)
