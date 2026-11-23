"""Run after extracting Supplementary Data S1. Requires Python 3 and NumPy."""
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import gzip
import json
import sys
import zipfile
import numpy as np

# A ZIP path is optional so the distributed archive can also be checked without extracting it.
archive=zipfile.ZipFile(sys.argv[1]) if len(sys.argv)>1 else None
root=Path(__file__).resolve().parent
def raw(path):
    return archive.read(path) if archive else (root/path).read_bytes()
def read(path):
    b=raw(path)
    return json.loads(gzip.decompress(b) if path.endswith(".gz") else b)
def same(x,y):
    np.testing.assert_allclose(x,y,rtol=0,atol=1e-8)

for path,digest in read("SHA256.json").items():
    assert sha256(raw(path)).hexdigest()==digest,path
utility=read("data/road/utility-by-drive.json")
road=read("data/road/kitti-rgb-individual.json")
for ds,ref in utility.items():
    by=defaultdict(lambda:defaultdict(list))
    for row in road:
        if row["dataset"]==ds:
            by[row["group"]][row["arm"]].append(row["psnr"])
    vectors={a:np.array([np.mean(v[a]) for v in by.values()]) for a in ref["scores"]}
    for arm,vec in vectors.items():same(vec.mean(),ref["scores"][arm])
    contrast=vectors["utility"]-vectors["direct"]
    same(contrast.mean(),ref["delta"])
    draws=np.random.default_rng(31041).integers(6,size=(12000,6))
    same(np.percentile(contrast[draws].mean(1),[2.5,97.5]),ref["ci"])
for cohort in ("photographic","external"):
    photos=read(f"data/rgb/{cohort}-individual.json")
    target=read(f"data/rgb/{cohort}-test.json")["sets"]
    vals=defaultdict(lambda:defaultdict(list))
    for row in photos:
        for metric in ("psnr","deltaE","ab_error"):
            vals[row["ds"],row["arm"],metric][row["group"]].append(row[metric])
    for (ds,arm,metric),by in vals.items():
        same(np.mean([np.mean(g) for g in by.values()]),target[ds]["scores"][metric][arm])

def miou(mat,classes=range(19)):
    h=np.stack([d for _,d in sorted(mat.items())])
    hits=h.diagonal(axis1=-2,axis2=-1)
    total=h.sum(-1)+h.sum(-2)-hits
    return np.divide(hits*100.,total,out=np.zeros_like(total,dtype=float),where=total>0)[:,list(classes)].mean()
real=read("data/semantics/real-joint-individual.json.gz")
targets=read("data/semantics/real-joint.json")
assert len(real)==targets["predictions"]==2115
acc=defaultdict(dict)
for r in real:
    ds=r["dataset"]
    for field in ("counts","edge"):
        if field not in r:continue
        scopes=(ds,"foggy_driving_101") if field=="counts" and ds.startswith("foggy_driving_") else (ds,)
        for s in scopes:
            by=acc[field,s,r["arm"]]
            by[r["seed"]]=by.get(r["seed"],0)+np.asarray(r[field])
for (field,ds,arm),v in acc.items():
    same(miou(v),targets["metrics"][field][ds]["scores"][arm])
for cohort in ("external","new","rain","snow","virtual"):
    frames=read(f"data/donors/{cohort}-individual.json.gz")
    target=read(f"data/donors/{cohort}-test.json")
    total=defaultdict(lambda:np.zeros(5))
    counts=defaultdict(dict)
    for r in frames:
        ds=r["ds"];arm=r["method"]
        if "donor" in r and arm in ("mean","fixed"):
            total[ds,arm]+=np.asarray(r["donor"][0])
        for field in ("counts","edge"):
            if field not in r:continue
            for scope in ((ds,"foggy_driving_101") if field=="counts" and ds.startswith("foggy_driving_") else (ds,)):
                by=counts[field,scope,arm]
                by[r["seed"]]=by.get(r["seed"],0)+np.asarray(r[field])
    for (ds,arm),v in total.items():
        d=target["diagnostics"][ds]["subsets"]["mixed_owner"]["scores"][arm]
        for j,k in enumerate(("owner_percent","raw_rgb_rmse","restored_rgb_rmse","effective_pixels"),1):same(v[j]/v[0],d[k])
    for (field,ds,arm),v in counts.items():
        classes=(0,2,5,6,7,8,9,10,13,14) if ds=="vkitti_fog" else range(19)
        same(miou(v,classes),target["full" if field=="counts" else field][ds]["scores"][arm])
joint=read("data/joint/joint-seg.json.gz")
target=read("data/joint/joint-test.json")
assert len(joint)==1980
bank=defaultdict(dict)
for r in joint:
    for field in ("counts","edge"):
        b=bank[field,r["ds"],r["rgb"]+"/"+r["sem"]]
        b[r["seed"]]=b.get(r["seed"],0)+np.asarray(r[field])
for (field,ds,arm),v in bank.items():same(miou(v),target["full" if field=="counts" else field][ds]["scores"][arm])
pics=read("data/joint/joint-rgb.json")
bygroup=defaultdict(lambda:defaultdict(list))
for r in pics:
    for metric in ("psnr","deltaE"):
        bygroup[r["ds"],r["arm"],metric][r["group"]].append(r[metric])
for (ds,arm,metric),gs in bygroup.items():same(np.mean([np.mean(g) for g in gs.values()]),target["photos"][ds]["scores"][metric][arm])
for t in [*read("data/training/rgb-traces.json").values(),read("data/training/seg-trace.json")]:
    assert len(t["minibatch"])==t["steps"] and t["probe"][0][0]==0 and t["probe"][-1][0]==t["steps"]
    same(t["minibatch"][-1],t["archived_last_loss"])
    same(t["checkpoint_max_difference"],0.)
assert targets["example_verified"]
assert all(v["confusion_verified"] for v in read("data/examples/paired-examples.json").values())
for r in read("data/examples/photo-examples.json"):
    for score in r["scores"].values():same(score["max_abs_ledger_difference"],0.)

trace=read("figures/provenance/verified-full-network.json")
assert trace["native_rgb_and_mask_pixel_identical"]
assert trace["backbones"]["MCT_stage_depths"]==[4,4,8,4,4]
assert sum(trace["trained"].values())==75872
stages=trace["feature_trace"]
for i,(ch,side) in enumerate(((24,256),(48,128),(96,64),(48,128),(24,256)),1):
    assert all(v["out"]==[1,ch,side,side] for v in stages[f"MCT.layer{i}"])
assert len(stages["MCT.native_RGB"])==3
assert [x["out"][1] for x in stages["LiDSeg.LGF.s8"]]==[256,256]
assert trace["tensors"]["cell_weights"]==[240*320,16]
assert trace["tensors"]["native_labels"]==trace["tensors"]["native_RGB"][:2]==[960,1280]
for stage,ledger,method,idfield,hashfield,arm,ids in (
    ("road-transition-v68","data/road/rgb-completed.json","utility","arm","SHA256","rgb-utility",(32003,32017,32029)),
    ("road-tone-control-v98","protocols/rgb/completed.json","affine","method","weights","affine",(32003,32017,32029)),
    ("road-tone-control-v98","protocols/rgb/completed.json","luma","method","weights","luma",(32003,32017,32029)),
    ("resolution-interaction-v84","protocols/detail/completed.json","raw_detail","method","weights_SHA256","raw_detail",(101,211,307)),
    ("semantic-subpixel-v94","protocols/donor/completed.json","dual_core","method","weights_SHA256","dual_core",(101,211,307))):
    runs=read(ledger)["runs"]
    for seed in ids:
        [fit]=[r for r in runs if r[idfield]==method and r["seed"]==seed]
        assert sha256(raw(f"fitted-heads/{stage}/{arm}-{seed}.pth")).hexdigest()==fit[hashfield]
photos=read("data/examples/expanded-rgb-records.json")
assert len(photos)==4
for r in photos:
    assert r["index_zero_based"]==(r["total"]-1)//2
    same(r["psnr_input_mct_curve"][2],r["matched_archived_curve"])
scene=next(r for r in photos if r["dataset"]=="ohaze")
with np.load(BytesIO(raw("data/examples/local-restoration-maps.npz")),allow_pickle=False) as maps:
    assert list(maps["step"].shape)==scene["grid"] and np.all((0<=maps["step"])&(maps["step"]<=1))
    same(np.bincount(maps["candidate"].ravel(),minlength=4),scene["candidate_cell_counts"])
    same((maps["alpha"]>.5).mean(),scene["alpha_fraction_over_half"])
cell=read("data/examples/cell-example.json")
z=np.array(cell["children"])
w=np.array(cell["percent_weights"])/100
colors=np.array(cell["native_grid_rgb_4x4"])
np.testing.assert_allclose(w.sum(),1,rtol=0,atol=1e-7)
np.testing.assert_allclose([z.mean()*100,(w*z).sum()*100],cell["correct_mass_percent"],rtol=0,atol=1e-5)
means=np.stack((colors.mean((0,1)),(colors*w[:,:,None]).sum((0,1)),colors[z==1].mean(0)))
np.testing.assert_allclose(means,cell["donor_rgb"],rtol=0,atol=2e-5)
np.testing.assert_allclose(np.sqrt(((means[:2]-means[2])**2).mean(1)),cell["raw_rmse"],rtol=0,atol=2e-5)
sem_examples=read("data/examples/expanded-semantic-records.json")
assert len(sem_examples)==4
for record in sem_examples:
    assert record["checked_older_confusions"]==["fixed","dysample","sapa_context"]
    x1,y1,x2,y2=record["crop_xyxy"]
    h,w=record["shape"]
    assert 0<=x1<x2<=w and 0<=y1<y2<=h
    if record["calibrated_confusion_also_archived"]:
        [prior]=[r for r in real if r["dataset"]==record["dataset"] and r["id"]==record["id"]
                                 and r["arm"]=="curve_fixed" and r["seed"]==101]
        same(record["calibrated_confusion"],prior["counts"])
print("Verified file digests, six tables' numerical sources, three optimization replays, 2,115 real-fog method/seed records, 1,980 stress-test predictions, the traced network, fifteen fitted heads and the selected figure checks.")


# Recompute the new local examples directly from their unrounded source records.
worked=read("data/examples/worked-examples.json")
q=np.array(worked["squared_magnitude"])
projection=np.array(worked["projection"])
steps=np.clip(np.divide(projection,q,out=np.zeros_like(q),where=q>0),0,1)
np.testing.assert_allclose(steps,worked["step"],rtol=0,atol=1e-6)
gain=2*steps*projection-steps**2*q
np.testing.assert_allclose(gain,worked["gain"],rtol=0,atol=1e-7)
k=int(gain.argmax())
assert k==worked["candidate_id"]
ref_gain=2*steps[k]*worked["reference_projection"][k]-steps[k]**2*q[k]
same(ref_gain,worked["reference_gain_at_selected_step"])
same(ref_gain,worked["direct_local_mse_reduction"])
alpha=np.interp(worked["luminance"],np.linspace(0,1,5),worked["knots"])
same(alpha,worked["window_fraction"])
pix=worked["pixel_rgb"]
same((1-alpha)*np.array(pix["affine"])+alpha*np.array(pix["windowed"]),pix["blended"])
with np.load(BytesIO(raw("data/examples/central-photo-crop.npz")),allow_pickle=False) as z:
    assert z["target"].shape==(96,96,3)
    for name,value in worked["central_crop"]["psnr"].items():
        same(-10*np.log10(np.square(z[name]-z["target"],dtype=np.float64).mean()),value)
coords=worked["central_crop"]["bounds_xyxy"]
height,width=worked["image_shape"]
assert coords==[width//2-48,height//2-48,width//2+48,height//2+48]
ex=worked["semantic"]
[r]=[r for r in real if r["seed"]==101 and r["dataset"]=="foggy_driving_fine" and
     r["arm"]=="curve_fixed" and r["id"]==ex["id"]]
for field,key,err in (("counts","valid","full_errors"),("edge","band","band_errors")):
    grid=np.array(r[field])
    assert grid.shape==(19,19) and grid.sum()==ex[key]
    assert grid.sum()-grid.trace()==ex[err]
with np.load(BytesIO(raw("data/examples/score-regions.npz")),allow_pickle=False) as masks:
    assert tuple(masks["valid"].shape)==tuple(ex["shape"])
    assert masks["valid"].sum()==ex["valid"] and masks["edge"].sum()==ex["band"]
    assert not np.any(masks["edge"] & ~masks["valid"])
same(ex["band"]/ex["valid"],ex["pixel_fraction"])
palette=read("data/examples/class-palette.json")
assert [v["id"] for v in palette]==[*range(19),255] and palette[-1]["rgb"]==[0,0,0]
bydrive=read("data/examples/utility-drive-points.json")
for ds,values in bydrive.items():
    assert len(values)==6
    for group,value in values.items():
        def group_mean(arm):
            return np.mean([r["psnr"] for r in road if r["dataset"]==ds and r["group"]==group and r["arm"]==arm])
        same(value,group_mean("utility")-group_mean("direct"))
refs=read("bibliography/verified-references.json")["references"]
assert [r["number"] for r in refs]==list(range(1,59)) and len({r["url"] for r in refs})==58
assert len(read("tables-as-printed.json"))==6 and len(read("context-table.json"))==7
print("Verified the A4 figure examples, local predicted and observed gains, three crop PSNRs, native scoring masks, twelve drive contrasts, and 58 linked references.")


# Architecture and equation edition.
eqs=read("mathematics/equation-map.json")
tex=raw("mathematics/Equations.tex").decode("utf8")
assert [e["number"] for e in eqs]==list(range(1,25))
for e in eqs:
    assert tex.count("\\label{eq:"+e["label"]+"}")==1
    assert e["latex"] in tex.splitlines() and e["implementation"].strip()
review=read("mathematics/equation-review.json")
assert review["native_output_pixel_identity"] and len(review["tests"])==27
assert np.all(np.isfinite(list(review["tests"].values()))) and max(review["tests"].values())<2e-5
assert review["device"] and "cu" in review["torch"]
assert sum(r["type"] in ("journal","proceedings") for r in refs)==55
assert len(read("bibliography/additional-crossref-records.json"))==14
assert len({r["url"] for r in refs if r["url"].startswith("https://doi.org/")})==54
bib=raw("bibliography/References.bib").decode("utf8")
assert len(bib.splitlines())>len(refs) and bib.count("\n}")==58
for r in refs:
    assert bib.count("{ref"+str(r["number"]).zfill(2)+",")==1
    assert "url = {"+r["url"]+"}" in bib
audit=read("figures/provenance/architecture-review.json")
assert audit["trace_pixel_identical"] and audit["published_equations"]==len(eqs)
for key,dig in audit["generated"].items():
    assert sha256(raw("figures/"+key+".png")).hexdigest()==dig
for k,in_channels,out_channels in (("raw-detail.conv0",76,40),("raw-detail.conv3",40,32),("raw-detail.conv6",32,38)):
    [v]=stages[k]
    assert v["in"][0]==[1,in_channels,240,320] and v["out"]==[1,out_channels,240,320]
assert trace["backbones"]["MCT"]==1233024 and trace["backbones"]["LiDSeg"]==39546330
s=read("data/examples/boundary-stencil.json")
beta=np.array(s["bilinear"]); pi=np.array(s["guided"])
assert s["native_row_column"]==[4*n+i for n,i in zip(s["row_column_zero_based"],s["child_row_column"])]==[473,613]
assert set(map(tuple,s["neighbor_offsets"]))=={(dy,dx) for dy in (-1,0,1) for dx in (-1,0,1)}
assert np.count_nonzero(beta)==4 and np.array_equal(beta>0,pi>0)
np.testing.assert_allclose([beta.sum(),pi.sum()],1,rtol=0,atol=1e-7)
center=s["neighbor_offsets"].index([0,0]);left=s["neighbor_offsets"].index([0,-1])
assert (round(100*beta[center],2),round(100*pi[center],2),round(100*(pi[left]-beta[left]),2))==(76.56,63.35,10.75)
print("Checked 24 equations and 27 numerical comparisons, the generated diagrams, frozen and fitted layer dimensions, the measured interpolation stencil, and all 58 bibliographic records.")



# Figure 2 measurements and Figure 8 study design (uses the same archive helpers).
provenance = "figures/provenance/framework/"
activation = read(provenance + "activation-review.json")
skip = read(provenance + "skip-review.json")
study = read(provenance + "protocol-review.json")
assert activation["frame"] == skip["frame"]
assert activation["input_sha256"] == trace["example_file"]
assert trace["native_rgb_and_mask_pixel_identical"] and study["native_both_outputs_equal"]
assert trace["shape"] == [960, 1280, 3]
assert trace["tensors"]["native_labels"] == [960, 1280]
assert trace["tensors"]["native_RGB"] == [960, 1280, 3]
expected_chw = {
    "MCT_layer1": [24, 256, 256], "MCT_layer3": [96, 64, 64],
    "PID_P3": [256, 120, 160], "PID_I3": [1024, 15, 20], "PID_D3": [256, 120, 160]
}
assert activation["tensors_CHW"] == expected_chw
with np.load(BytesIO(raw("data/examples/measured-activations.npz")), allow_pickle=False) as maps:
    assert set(maps.files) == set(expected_chw)
    for name, (channels, height, width) in expected_chw.items():
        values = maps[name]
        assert values.shape == (height, width) and values.dtype == np.float32
        assert np.isfinite(values).all() and values.max() > values.min()
        assert sha256(np.ascontiguousarray(values).tobytes()).hexdigest() == activation["array_sha256"][name]
        np.testing.assert_allclose(np.percentile(values, [2, 98]), activation["percentile_limits"][name],
                                   rtol=1e-5, atol=1e-7)
with np.load(BytesIO(raw("data/examples/measured-skip-weights.npz")), allow_pickle=False) as scales:
    assert set(scales.files) == {"stage2_to_4", "stage1_to_5"}
    for name, channels in (("stage2_to_4", 48), ("stage1_to_5", 24)):
        w = scales[name]
        assert w.shape == (2, channels) and w.dtype == np.float32
        assert (w >= 0).all() and (w <= 1).all()
        np.testing.assert_allclose(w.sum(0), np.ones(channels), rtol=0, atol=2e-7)
        np.testing.assert_allclose(w[1].mean(), skip["mean_skip"][name], rtol=0, atol=1e-7)
        assert skip["max_abs_mix_error"][name] < 2e-6
assert [stages[f"MCT.layer{i}"][0]["out"][1:] for i in range(1, 6)] == [
    [24, 256, 256], [48, 128, 128], [96, 64, 64], [48, 128, 128], [24, 256, 256]]
assert stages["MCT.patch_unembed"][0]["out"] == [1, 72, 256, 256]
assert [s["out"] for s in stages["PID.conv_seg"]] == [[1, 19, 96, 128], [1, 19, 120, 160]]
drives = []
for part, nframes, ndrives in (("fit", 87, 19), ("calibrate", 19, 4), ("held", 36, 6)):
    cohort = study["kitti"][part]
    assert cohort["frames"] == nframes and len(set(cohort["drives"])) == ndrives
    drives.extend(cohort["drives"])
assert len(drives) == len(set(drives)) == 29 and study["kitti_split_drive_overlap"] == 0
assert study["kitti_fog_views_per_frame"] == 2 and study["calibration_views"] == {"kitti_cal": 38, "nh_train": 36}
assert study["knots"] == 5 and study["rgb_head_updates"] == 250
fit = study["semantic_fitting"]
assert (fit["frames"], len(set(fit["recordings"])), fit["adapter_updates"], fit["decoder_updates"]) == (96, 9, 480, 320)
assert (fit["range_strengths"], fit["spatial_exponent"]) == (8, 1)
ct = study["controlled_test"]
assert (ct["frames"], len(set(ct["recordings"])), ct["views"]) == (33, 13, 66)
assert ct["paired_ids_match"] and ct["transmissions"] == [0.7, 0.4]
assert set(fit["recordings"]).isdisjoint(ct["recordings"])
assert len(audit["generated"]) == 5 and len(audit["current_revision"]) == 3
for name, item in audit["current_revision"].items():
    assert sha256(raw(provenance + item["draft_filename"])).hexdigest() == item["draft_sha256"]
    assert item["final_sha256"] == audit["generated"][name]
    assert raw(provenance + item["prompt"]).strip() and raw(provenance + item["revision_prompt"]).strip()
print("Verified five measured feature maps, 72 channel-wise skip weights across two fusions, native outputs, three disjoint KITTI groups, MUSES training and paired test cohorts, and revision records for three schematics.")


# Recompute the sign of the predicted and observed gains, the blend and the donor example.
fig3=read("data/examples/figure3-review.json")
photo=read("data/examples/worked-examples.json")
cell=read("data/examples/cell-example.json")
audit3=read("figures/provenance/fig03/review.json")
for n,record in audit3["illustrations"].items():
    original=raw("figures/provenance/fig03/"+n+".png")
    assert sha256(original).hexdigest()==record["sha256"]
    assert raw("figures/provenance/fig03/"+n+"-prompt.txt").strip()
assert sha256(raw("figures/Fig03-modules.png")).hexdigest()==audit3["printed_png_sha256"]
assert len(audit3["illustrations"])==3
for name in ("png","pdf","svg"):
    assert raw("figures/Fig03-modules."+name)
with np.load(BytesIO(raw("data/examples/figure3-examples.npz")),allow_pickle=False) as ex:
    for name,digest in fig3["arrays_sha256"].items():
        assert sha256(ex[name].tobytes(order="C")).hexdigest()==digest
    patches=ex["candidate_patches"]
    assert patches.shape==(4,72,72,3) and patches.dtype==np.float32
    assert np.isfinite(patches).all() and patches.min()>=0 and patches.max()<=1
    assert tuple(ex["step_map"].shape)==tuple(photo["grid"])
    r,c=photo["cell_row_col"]
    assert abs(ex["step_map"][r,c]-photo["step"][3])<1e-6
    square=np.s_[22:26,22:26]
    np.testing.assert_array_equal(ex["context_rgb"][square],np.array(cell["native_grid_rgb_4x4"]))
    labels=(ex["context_labels"][square]==cell["owner_id"])
    np.testing.assert_array_equal(labels,np.array(cell["children"]))
q,p=np.asarray(photo["squared_magnitude"]),np.asarray(photo["projection"])
step=np.clip(p/np.maximum(q,1e-8),0,1)
estimate=2*p*step-q*step**2
selected=int(estimate.argmax())
assert selected==photo["candidate_id"]==3
np.testing.assert_allclose(step,fig3["candidate_steps"],atol=1e-6)
np.testing.assert_allclose(estimate[selected],fig3["predicted_gain"],atol=1e-7)
real=2*photo["reference_projection"][selected]*step[selected]-q[selected]*step[selected]**2
np.testing.assert_allclose(real,fig3["observed_gain"],atol=1e-7)
assert fig3["predicted_gain"]>0>fig3["observed_gain"]
alpha=np.interp(fig3["luminance"],np.linspace(0,1,5),fig3["knots"])
same(alpha,fig3["alpha"])
np.testing.assert_allclose(photo["pixel_rgb"]["affine"],photo["pixel_rgb"]["blended"],atol=1e-7)
weights=np.asarray(cell["percent_weights"])
assert labels.sum()==10
np.testing.assert_allclose([100*labels.mean(),weights[labels].sum()],fig3["donor_mass_percent"],atol=2e-5)

# The previous unformatted analysis transcriptions provide an independent inverse check.
before=read("tables/source-transcriptions.json")
printed=read("tables/printed-structure.json")
assert len(before)==len(printed)==7
total_rows=0
for n,(p0,p1) in enumerate(zip(before,printed),1):
    assert p1["number"]==n
    head=p1["column_headers"]
    official=read("context-table.json") if n==1 else read("tables-as-printed.json")[n-2]
    assert official==[head,*p1["rows"]]
    assert len(p0)==len(p1["rows"])+1
    for oldrow,new in zip(p0[1:],p1["rows"]):
        if n==3:
            restored=[new[0],new[1]+" / "+new[2],new[3]+" "+new[4],new[5]+" "+new[6]]
        elif n==6:
            effect=new[6] if new[7]=="—" else new[6]+" "+new[7]
            restored=[new[0],new[1],new[2]+" → "+new[3],new[4]+" → "+new[5],effect]
        elif n==7:
            restored=[*new[:-2],new[-2]+" "+new[-1]]
        else:
            restored=new
        assert restored==oldrow,(n,oldrow,new)
        total_rows+=1
    if n in (4,5):
        winners=set()
        for k in range(1,len(head)):
            nums=[float(row[k]) for row in p1["rows"]]
            best=min(nums) if n==4 and k in (2,4) else max(nums)
            winners.update((i,k) for i,x in enumerate(nums) if x==best)
        assert winners==set(tuple(x) for x in p1["bold_best"])
assert total_rows==40
print("Verified Figure 3 measured crops, both utility signs, the five-knot blend, native vegetation weights and the reversible transcription of all 40 rows in seven editable tables.")

# Compare both native segmentation replays to the archived confusion matrices,
# and reproduce the recording-deletion analysis from the original 1,980 rows.
vis=read("data/examples/figure17-review.json")
original_examples=read("data/examples/paired-examples.json")
raw_joint=read("data/joint/joint-seg.json.gz")
registered=read("data/joint/joint-test.json")
sens=read("data/examples/recording-sensitivity.json")
for when, item in vis["selected"].items():
    assert item["id"]==original_examples[when]["id"] and item["recording"]==original_examples[when]["group"]
    for cohort,arms in item["levels"].items():
        for arm,measure in arms.items():
            h=np.asarray(measure["confusion"])
            assert h.shape==(19,19)
            assert (measure["valid"],measure["predicted_sky"],measure["reference_sky"],measure["correct"])==(
                h.sum(),h[:,10].sum(),h[10].sum(),h.diagonal().sum())
            if arm in ("fixed","mean"):
                [row]=[r for r in raw_joint if r["id"]==item["id"] and r["ds"]==cohort and r["seed"]==vis["seed_seg"] and r["rgb"]=="curve" and r["sem"]==arm]
                np.testing.assert_array_equal(row["counts"],h)
night=vis["selected"]["night"]["levels"]["MUSES_t40"]
assert (night["fixed"]["predicted_sky"],night["fixed"]["valid"],night["fixed"]["reference_sky"])==(1373146,1373151,0)
assert all(m["predicted_sky"]/m["valid"]>.9999 for m in night.values())
for ds,reference in sens.items():
    rs=[r for r in raw_joint if r["ds"]==ds and r["rgb"]=="curve"]
    recordings=sorted(set(r["group"] for r in rs))
    assert len(recordings)==13 and set(reference["omitted_recording_effects"])==set(recordings)
    def delta_excluding(omit):
        scores={}
        for mode in ("fixed","mean"):
            perseed=[]
            for seed in (101,211,307):
                h=sum(np.array(r["edge"]) for r in rs if r["seed"]==seed and r["sem"]==mode and r["group"]!=omit)
                union=h.sum(0)+h.sum(1)-h.diagonal()
                perseed.append(np.divide(100*h.diagonal(),union,out=np.zeros(19,dtype=float),where=union>0).mean())
            scores[mode]=np.mean(perseed)
        return scores["fixed"]-scores["mean"]
    same(delta_excluding(None),reference["pooled"])
    same(reference["pooled"],registered["edge"][ds]["effects"]["curve-fixed_minus_mean"]["diff"])
    vals=[delta_excluding(g) for g in recordings]
    np.testing.assert_allclose(vals,[reference["omitted_recording_effects"][g] for g in recordings],atol=1e-8)
    np.testing.assert_allclose([min(vals),max(vals)],reference["range"],atol=1e-8)
    assert all(v>0 for v in vals)
print("Verified Figure 17 native sky counts for the original day/night examples, the bilinear baseline receipt, and all 26 leave-one-recording-out paired effects.")
