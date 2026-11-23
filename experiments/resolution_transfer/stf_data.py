"""Lock real fog recordings and validate the reference LiDAR-to-camera projection."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
from PIL import Image
import requests
from scipy.ndimage import binary_propagation, maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from experiments.resolution_transfer.cohort import OUT, ROOT, digest, read, write
from experiments.resolution_transfer import stf_archive as archive

TO_CITY={1:13,2:18,3:17,4:14,6:11,9:0,11:1,13:2,14:4,15:8,17:9,18:5,19:7}
RAW_NAMES={1:"car",2:"bicycle",3:"motorcycle",4:"truck",6:"person",9:"road",11:"sidewalk",
           13:"building",14:"fence",15:"vegetation",17:"terrain",18:"pole",19:"traffic-sign"}
SOURCE_REV="bb57f8c5e29a8d647d5f65b0e277401bf332cbd0"
MIRROR=archive.DATA/"members/SemanticSTF"
CONTROL_URL=f"https://raw.githubusercontent.com/princeton-computational-imaging/SeeingThroughFog/{SOURCE_REV}/tools/ProjectionTools/Sweden2KittiCalib/kitti_stereo_velodynehdl_calib.txt"


def reserve_fog():
    archive.index()
    lines=(MIRROR/"val/val.txt").read_text(encoding="utf-8").splitlines()
    by_video=defaultdict(list)
    for row in lines:
        ident,weather=row.split(",")
        if weather in ("light_fog","dense_fog"):
            group=ident.rsplit("_",1)[0];date=ident[:10]
            by_video[group].append(dict(id=ident,weather=weather,recording=group,day=date))
    assert sum(map(len,by_video.values()))==156 and len(by_video)==46
    rank=lambda x:sha256(("stf-frozen-boundary-113/"+x["id"]).encode()).hexdigest()
    subset=[min(rows,key=rank) for _,rows in sorted(by_video.items())]
    assert len({r["day"] for r in subset})==11
    official_taxonomy=(MIRROR/"semanticstf.yaml").read_text()
    names={int(k):s for k,s in re.findall(r'(\d+):\s*"([^"]+)"',official_taxonomy)}
    assert all(names[k]==s for k,s in RAW_NAMES.items())
    spec={"task":"Prospective external replication on author-labelled real fog and pointwise human semantics. Defined after seeing the dense BDD results, before opening any SemanticSTF image, cloud, point class, or network output. All methods/weights already frozen. This is a separate secondary endpoint.",
          "selection":"Official validation split only. Among 156 fog frames, one hash-ranked frame per original acquisition timestamp (46 video prefixes over 11 days). Do not select according to severity, label prevalence, visual quality or model scores. Captures on the same date are the sampling cluster. Locations/roads can repeat across dates; GPS independence not assumed.",
          "labels":"13 directly corresponding Cityscapes/STF classes; predeclared table. Ignore unlabelled, invalid, other-vehicle, bike riders, parking, other-ground and trunk. Predictions outside the mapped set count as missed matched-class points; they are never reassigned. Cameras and native laser points are synchronized by original ID.",
          "projection":"Use rectified stereo P and inverse body-to-camera pose times body-to-roof HDL64 pose. Cross-check against Princeton's released KITTI-form transform and P. Project all finite forward returns at z>1m; keep the nearest hit per rounded image pixel before class filtering. Reject hidden-surface candidates more than max(1m, .05*z) behind the nearest return in a 5x5 pixel window. Use only points at least two pixels inside the frame. Inspect image-overlay calibration on the first timestamp of four dates picked solely by SHA before semantic inference.",
          "metrics":"All valid visible points and points within Euclidean radius 6,12,24 native image pixels of a visible point from a different mapped reference class. Primary radius=12. Evaluate semantic prediction at projected pixels only. Do not infer dense masks in between. Zero-union class contributes zero to fixed-13 mIoU. Report GT-present sensitivity and percent of image actually labelled. Compare at long side 768; three existing source seeds averaged after pooling. Report light/dense separately. Day-level paired cluster percentile CI, 5000 draws, plus equal-day and leave-one-day diagnostics. Test calibration sensitivity by querying same reference at x/y shifts of +/-2 pixels; no outcome-based correction.",
          "arms":["released","coarse","fixed","paper","shared","by_scale","shifted_paper","shifted_shared"],
          "primary_comparisons":"Original manuscript vs released and matched fixed; aligned paper vs shifted paper. Distinguish improvement over original backbone from advantage of fitted weights. Other arms exploratory; a new learned contribution needs support over the original and fixed, not just over backbone.",
          "calibration_source":CONTROL_URL,"release":archive.REV,"archive_size":archive.SIZE,"release_lfs_sha256":archive.ARCHIVE_OID,
          "protocol":digest(OUT/"protocol.json"),"frozen_training":digest(OUT/"training.json"),
          "source_files":{str(p.relative_to(MIRROR)):digest(p) for p in (MIRROR/"val/val.txt",MIRROR/"semanticstf.yaml",
                    MIRROR/"calib/calib_tf_tree_full.json",MIRROR/"calib/calib_cam_stereo_left.json")},
          "classes":{str(k):dict(stf=RAW_NAMES[k],city=v) for k,v in TO_CITY.items()},"rows":subset}
    write(OUT/"stf-protocol.json",spec,True)
    return subset


def calibrate():
    cam=read(MIRROR/"calib/calib_cam_stereo_left.json")
    tree=read(MIRROR/"calib/calib_tf_tree_full.json")
    def body(child):
        x=next(v for v in tree if v["child_frame_id"]==child)
        assert x["header"]["frame_id"]=="body"
        a=x["transform"];M=np.eye(4)
        M[:3,:3]=Rotation.from_quat([a["rotation"][n] for n in "xyzw"]).as_matrix()
        M[:3,3]=[a["translation"][n] for n in "xyz"]
        return M
    tr=np.linalg.inv(body("cam_stereo_left_optical"))@body("lidar_hdl64_s3_roof")
    P=np.asarray(cam["P"]).reshape(3,4)
    ref=archive.DATA/"kitti-stereo-hdl64.txt"
    if not ref.is_file():
        r=requests.get(CONTROL_URL,timeout=(8,24));r.raise_for_status()
        ref.write_bytes(r.content)
    content={k:np.fromstring(v,sep=" ") for k,v in (line.split(":",1) for line in ref.read_text().splitlines() if ":" in line)}
    extrinsic_error=float(np.max(np.abs(tr.flatten()-content["Tr_velo_to_cam"])))
    intrinsic_error=float(np.max(np.abs(P.flatten()-content["P2"])))
    assert extrinsic_error<1e-5 and intrinsic_error<1e-5,(extrinsic_error,intrinsic_error)
    return tr,P,(cam["height"],cam["width"]),dict(extrinsic_max_abs=extrinsic_error,intrinsic_max_abs=intrinsic_error,official_sha256=digest(ref))


def stage():
    rows=reserve_fog();bk=archive.index()
    _,_,shape,check=calibrate()
    def row(r):
        files=[archive.member(f"SemanticSTF/val/{folder}/{r['id']}.{ext}",bk)
               for folder,ext in (("images","png"),("velodyne","bin"),("labels","label"))]
        with Image.open(files[0]) as im:assert im.size==shape[::-1]
        assert files[1].stat().st_size//20==files[2].stat().st_size//4
        return dict(r,paths=[str(p.relative_to(ROOT)) for p in files],sha256=[digest(p) for p in files])
    with ThreadPoolExecutor(12) as pool:
        done=[]
        for i,r in enumerate(pool.map(row,rows),1):
            done.append(r)
            if i%10==0 or i==len(rows):print("Staged labelled fog recordings",i,"/",len(rows),flush=True)
    write(OUT/"stf-manifest.json",dict(rows=done,preselection=digest(OUT/"stf-protocol.json"),calibration=check,
                                       weather=dict(Counter(r["weather"] for r in done))),True)


def project(row, tr,P, expected_shape):
    image,cloud,annotation=[ROOT/p for p in row["paths"]]
    assert [digest(p) for p in (image,cloud,annotation)]==row["sha256"]
    with Image.open(image) as im:rgb=np.array(im.convert("RGB"))
    h,w=rgb.shape[:2];assert (h,w)==expected_shape
    xyz=np.fromfile(cloud,dtype="<f4").reshape(-1,5)[:,:3]
    raw=np.fromfile(annotation,dtype="<i4")
    assert len(xyz)==len(raw) and set(np.unique(raw))<=set(range(21))
    c=xyz.astype(np.float64)@tr[:3,:3].T+tr[:3,3]
    z=c[:,2];select=np.flatnonzero(np.isfinite(c).all(1)&(z>1.))
    p=c[select]@P[:,:3].T+P[:,3]
    pix=np.rint(p[:,:2]/p[:,2:3]).astype(np.int64)
    inside=(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    pix=pix[inside];select=select[inside]
    linear=pix[:,1]*w+pix[:,0]
    order=np.lexsort((select,z[select],linear))
    front=order[np.r_[True,np.diff(linear[order])!=0]]
    ix=select[front];pid=linear[front];xy=pix[front]
    depth=np.full(h*w,np.inf,np.float32);depth[pid]=z[ix];depth=depth.reshape(h,w)
    nearby=minimum_filter(depth,size=5,mode="constant",cval=np.inf).ravel()[pid]
    lut=np.full(21,255,np.uint8)
    for k,v in TO_CITY.items():lut[k]=v
    labels=lut[raw[ix]]
    valid=(labels<19)&(z[ix]<=nearby+np.maximum(1.,.05*z[ix]))
    valid&=(xy[:,0]>=2)&(xy[:,0]<w-2)&(xy[:,1]>=2)&(xy[:,1]<h-2)
    black=(rgb==0).all(-1);rim=np.zeros_like(black)
    rim[[0,-1],:]=black[[0,-1],:];rim[:,[0,-1]]=black[:,[0,-1]]
    padding=maximum_filter(binary_propagation(rim,mask=black),size=5).ravel()[pid]
    removed=int((valid&padding).sum());valid&=~padding
    xy=xy[valid];labels=labels[valid]
    assert len(np.unique(xy[:,1]*w+xy[:,0]))==len(xy) and len(xy)>0
    dist=np.full(len(xy),np.inf)
    for k in np.unique(labels):
        use=labels==k
        if (~use).any():dist[use]=cKDTree(xy[~use]).query(xy[use],k=1,distance_upper_bound=24.000001)[0]
    return rgb,xy,labels,dist,dict(points=len(raw),visible_before_class=len(ix),removed_in_rectification_padding=removed,
                                  mapped=len(labels),near12=int((dist<=12).sum()))


if __name__=="__main__":stage()
