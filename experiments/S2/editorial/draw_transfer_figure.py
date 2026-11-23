"""Plot the first reserved validation frame and native predictions with a label-defined crop."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter


PALETTE = np.asarray([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],
    [0,80,100],[0,0,230],[119,11,32],
], dtype=np.uint8)
EDIT_COLORS = np.asarray([[242,243,244],[0,114,178],[213,94,0],[143,101,168]],dtype=np.uint8)
TARGETS = {"mask2former":"Mask2Former", "segformer":"SegFormer-B1"}


def edge_and_band(labels):
    valid = labels < 19
    edge = np.zeros_like(valid)
    vertical = valid[1:] & valid[:-1] & (labels[1:] != labels[:-1])
    horizontal = valid[:,1:] & valid[:,:-1] & (labels[:,1:] != labels[:,:-1])
    edge[:-1] |= vertical
    edge[1:] |= vertical
    edge[:,:-1] |= horizontal
    edge[:,1:] |= horizontal
    return edge, maximum_filter(edge, size=13, mode="constant", cval=False) & valid


def colorize(labels):
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    good = labels < len(PALETTE)
    out[good] = PALETTE[labels[good]]
    return out


def array(path, expected_sha):
    assert sha256(path.read_bytes()).hexdigest() == expected_sha
    with Image.open(path) as f:
        return np.array(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Study directory or extracted Supplementary Data S2")
    parser.add_argument("data", type=Path, help="Parent of benchmark_data with the licensed BDD100K files")
    parser.add_argument("output", type=Path, help="Directory containing replayed masks and frame.json")
    args = parser.parse_args()
    folder = args.output
    record = json.loads((folder / "frame.json").read_text(encoding="utf-8"))
    row = record["data"]
    first = json.loads((args.study / "bddval-manifest.json").read_text(encoding="utf-8"))["rows"][0]
    assert row == first
    rgb = array(args.data / row["rgb_path"].replace(chr(92), "/"), row["rgb_sha"])
    truth = array(args.data / row["mask_path"].replace(chr(92), "/"), row["mask_sha"])
    h, w = truth.shape
    assert rgb.shape == (h,w,3)
    edge, band = edge_and_band(truth)
    ys, xs = np.nonzero(edge)
    nearest = np.argmin((ys - (h-1)/2)**2 + (xs - (w-1)/2)**2)
    cy, cx = int(ys[nearest]), int(xs[nearest])
    left = max(0,min(w-192,cx-96))
    top = max(0,min(h-128,cy-64))
    crop = np.s_[top:top+128,left:left+192]
    fig, axes = plt.subplots(3,3,figsize=(6.4,4.25))
    fig.subplots_adjust(top=.94,bottom=.105,left=.124,right=.997,hspace=.28,wspace=.045)
    plt.rcParams["font.family"] = "DejaVu Sans"
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for line in ax.spines.values():
            line.set_visible(False)
    def show(ax,image):
        ax.imshow(image,interpolation="nearest",aspect="equal")
    for ax,im,title in zip(axes[0],(rgb,rgb[crop],colorize(truth[crop])),("Full exposure","Native RGB crop","Reference labels")):
        show(ax,im)
        ax.set_title(title,size=8.7,pad=2.5,weight="semibold")
    axes[0,0].add_patch(Rectangle((left-.5,top-.5),192,128,fill=False,edgecolor="black",linewidth=1.6))
    axes[0,0].add_patch(Rectangle((left-.5,top-.5),192,128,fill=False,edgecolor="white",linewidth=.85))
    for ax,title in zip(axes[1],("Released decoder","Transferred decoder","Changes in boundary band")):
        ax.set_title(title,size=8.3,pad=2.5,weight="semibold")
    changes = {}
    for i,(backbone,label) in enumerate(TARGETS.items(),1):
        predicted = {}
        packet = args.study / f"counts-{backbone}-bddval" / (row["id"]+".npz")
        with np.load(packet,allow_pickle=False) as counts:
            for arm in ("released","paper_fit"):
                file=backbone+"-"+arm+".png"
                m=array(folder/file,record["mask_files_sha256"][file])
                for j,valid in enumerate((truth<19,band)):
                    result=np.bincount(19*truth[valid].astype(np.int32)+m[valid],minlength=361).reshape(19,19)
                    np.testing.assert_array_equal(result,counts["counts"][j,counts["arms"].tolist().index(arm),0])
                predicted[arm]=m
        baseline,new = predicted.values()
        edits = np.zeros_like(truth)
        edits[band & (new==truth) & (baseline!=truth)] = 1
        edits[band & (new!=truth) & (baseline==truth)] = 2
        edits[band & (new!=truth) & (baseline!=truth) & (new!=baseline)] = 3
        for ax, view in zip(axes[i],(colorize(baseline[crop]),colorize(new[crop]),EDIT_COLORS[edits[crop]])):
            show(ax,view)
        axes[i,0].set_ylabel(label,fontsize=8.25,weight="semibold",labelpad=3)
        changes[backbone] = {name:int(np.count_nonzero(edits[crop]==k)) for k,name in enumerate(
            ("other","corrected","newly_wrong","different_wrong_class")) if k}
    keys=["Corrected","Newly wrong","Different wrong class"]
    fig.legend([Patch(facecolor=EDIT_COLORS[k]/255) for k in (1,2,3)],keys,ncol=3,
               loc="lower center",bbox_to_anchor=(.54,.006),frameon=False,fontsize=8,
               handlelength=1,handletextpad=.4,columnspacing=1.45)
    for ext in ("png","pdf"):
        fig.savefig(folder/f"figure11c.{ext}",dpi=400,bbox_inches="tight",pad_inches=.042)
    plt.close(fig)
    data={"frame":row["id"],"crop_xywh":[left,top,192,128],"center_boundary_xy":[cx,cy],
          "valid_band_pixels_in_crop":int(band[crop].sum()),"changed_boundary_pixels_in_crop":changes,
          "selection":"First reserved validation frame. Fixed 192 by 128 crop centered on the annotated four-neighbor transition nearest the image center, before examining either prediction.",
          "checks":"Both native full and boundary matrices reproduced from each pair of saved PNGs. Crops are display only."}
    (folder/"figure11c-data.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
