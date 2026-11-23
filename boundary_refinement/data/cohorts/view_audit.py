"""Image-only overlap audit on the complete new and previous WeatherProof camera reservations."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from boundary_refinement.data.cohorts import reserve_anchor as repo
from boundary_refinement.data.cohorts import reserve_weatherproof as source
from boundary_refinement.data.staging.weatherproof_files import WORK,read,write

def run():
    out=repo.OUT
    plan={"source":"All 110 previously reserved and 161 new central clear reference crops; do not consult labels or prediction scores.",
          "retrieval":"RootSIFT local-neighbor vote, 14 nearest candidates per image, union nearest eight grayscale thumbnail signatures, exact file and decoded RGB hashes. Check mutual Lowe-ratio 0.72 matches with a robust similarity transform.",
          "threshold":"At least 12 geometric inliers, at least 55% of reciprocal matches, and convex hull spanning at least 6% of both images. Inspect flagged clear-image pairs. Sensitivity excludes new views connected to earlier views and clusters matching new views. The detector cannot establish geographical independence of disjoint crops."}
    if (out/"view-audit-plan.json").exists():assert read(out/"view-audit-plan.json")==plan
    else:write(out/"view-audit-plan.json",plan)
    sys.path.insert(0,str(WORK/"benchmark_data/segfix-author-v102/dependencies"));import cv2
    cv2.setNumThreads(2);cv2.setRNGSeed(10517)
    rows=[]
    for r in source.reservation()["rows"]:rows.append((r["code"],r["group"],r["triplets"][1]))
    for r in repo.reserve_wp():rows.append((r["code"],r["group"],r["id"]))
    feat=cv2.SIFT_create(nfeatures=420,contrastThreshold=.025)
    vectors=[];points=[];areas=[];grays=[];files=[];fingerprints=[];phash=[]
    for code,group,prefix in rows:
        p=source.DATA/"members"/(prefix+"_gt-clean.png");files.append(p)
        with Image.open(p) as im:
            rgb=np.array(im.convert("RGB"));fingerprints.append((sha256(p.read_bytes()).hexdigest(),sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()))
            im=im.convert("L");im.thumbnail((480,480));g=np.ascontiguousarray(np.array(im));grays.append(g)
        kp,desc=feat.detectAndCompute(g,None);areas.append(g.size)
        d=np.zeros((0,128),np.float32) if desc is None else np.sqrt(desc/(desc.sum(1,keepdims=True)+1e-10))
        points.append(np.array([k.pt for k in kp],np.float32));vectors.append(d)
        phash.append(cv2.dct(cv2.resize(g,(32,32)).astype(np.float32))[:8,:8].ravel()[1:])
    features=np.concatenate(vectors);owners=np.repeat(np.arange(len(rows)),[len(v) for v in vectors])
    index=cv2.flann_Index(features,dict(algorithm=1,trees=4));bits=np.array(phash);bits=bits>np.median(bits,axis=1,keepdims=True)
    pairs=set()
    for i,desc in enumerate(vectors):
        if not len(desc):continue
        near,d=index.knnSearch(desc,10,params=dict(checks=48))
        votes=np.bincount(owners[near].flatten(),minlength=len(rows));votes[i]=0
        options=set(np.argsort(-votes)[:15])|set(np.argsort(np.count_nonzero(bits[i]!=bits,axis=1))[:9])
        pairs.update(tuple(sorted((i,int(j)))) for j in options if j!=i)
    bf=cv2.BFMatcher(cv2.NORM_L2);flags=[]
    def accepted(v,w):
        return {m.queryIdx:m.trainIdx for m,n in bf.knnMatch(v,w,k=2) if m.distance<.72*n.distance}
    for i,j in sorted(pairs):
        if min(len(vectors[i]),len(vectors[j]))<12:continue
        a=accepted(vectors[i],vectors[j]);b=accepted(vectors[j],vectors[i])
        matches=[(s,t) for s,t in a.items() if b.get(t)==s]
        if len(matches)<12:continue
        l,r=np.array(matches).T;xx=points[i][l];yy=points[j][r]
        transform,mask=cv2.estimateAffinePartial2D(xx,yy,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=2000)
        if transform is None:continue
        use=mask.ravel().astype(bool);n=int(use.sum())
        if n<12 or n/len(matches)<.55:continue
        cover=min(cv2.contourArea(cv2.convexHull(pts[use]))/area for pts,area in ((xx,areas[i]),(yy,areas[j])))
        if cover>=.06:flags.append(dict(a=rows[i][0],b=rows[j][0],ga=rows[i][1],gb=rows[j][1],inliers=n,matched=len(matches),area=cover))
    for a,b in combinations(range(len(rows)),2):
        if fingerprints[a][1]==fingerprints[b][1] and not any(r["a"]==rows[a][0] and r["b"]==rows[b][0] for r in flags):
            flags.append(dict(a=rows[a][0],b=rows[b][0],ga=rows[a][1],gb=rows[b][1],exact=True))
    by_code={r[0]:i for i,r in enumerate(rows)}
    for at in range(0,len(flags),12):
        fs=flags[at:at+12];can=Image.new("RGB",(1122,((len(fs)+2)//3)*155),"white");draw=ImageDraw.Draw(can)
        for j,f in enumerate(fs):
            x=j%3*374;y=j//3*155
            draw.text((x+2,y+1),f["a"]+" / "+f["b"],fill="black")
            for v,k in enumerate(("a","b")):
                with Image.open(files[by_code[f[k]]]) as im:
                    im.thumbnail((184,136));can.paste(im.convert("RGB"),(x+186*v,y+16))
        can.save(out/f"overlap-{at//12+1:02d}.jpg",quality=92)
    note=dict(total_views=len(rows),retrieved_pairs=len(pairs),total_pairs=len(rows)*(len(rows)-1)//2,flags=flags,
              exact_decoded_duplicate_pairs=sum(n*(n-1)//2 for n in Counter(h[1] for h in fingerprints).values()),
              cv2=cv2.__version__,rule=plan)
    write(out/"view-audit.json",note);print("View audit",len(rows),len(pairs),"flagged",len(flags),flush=True)

if __name__=="__main__":run()
