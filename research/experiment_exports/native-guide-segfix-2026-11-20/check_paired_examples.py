"""Cross-check the exported paired RGB and mask examples against saved full-precision scores."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

HERE=Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text(encoding="utf-8"))
def pixels(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()

def main():
    note=read(HERE/"paired-evaluation.json")
    key=note["example"]
    row=next(r for r in read(HERE/"manifest.json")["rows"] if r["id"]==key)
    gt=pixels(HERE.parent.parent/row["mask_path"])
    edges=np.zeros_like(gt,dtype=bool)
    y=(gt[1:]!=gt[:-1])&(gt[1:]<19)&(gt[:-1]<19)
    x=(gt[:,1:]!=gt[:,:-1])&(gt[:,1:]<19)&(gt[:,:-1]<19)
    edges[:-1]|=y;edges[1:]|=y;edges[:,:-1]|=x;edges[:,1:]|=x
    band=maximum_filter(edges,size=13,mode="constant")
    cm_n=0; errors=[]
    with np.load(HERE/"paired-counts"/(key+".npz"),allow_pickle=False) as primary, \
         np.load(HERE/"paired-manuscript"/(key+".npz"),allow_pickle=False) as original:
        for k,level in enumerate(note["conditions"]):
            folder=HERE/"paired-examples"/str(level)
            ref=pixels(folder/"reference-rgb.png").astype(np.float64)/255.
            for i,name in enumerate(note["photo"]):
                image=pixels(folder/(name+"-rgb.png")).astype(np.float64)/255.
                psnr=-10*np.log10(np.mean((ref-image)**2))
                errors.append(abs(psnr-primary["psnr"][k,i]))
            for i,name in enumerate([*note["semantic"],"paper_512"]):
                pred=pixels(folder/(name+"-labels.png"))
                saved=original["counts"][k,:,0] if name=="paper_512" else primary["counts"][k,i,:,0]
                for j,mask in enumerate((gt<19,(gt<19)&band)):
                    ids=gt[mask].astype(np.int32)*19+pred[mask]
                    np.testing.assert_array_equal(np.bincount(ids,minlength=361).reshape(19,19),saved[j])
                    cm_n+=1
    assert max(errors)<.10
    result=dict(quantized_rgb_psnrs=len(errors),max_abs_rgb_png_error_db=max(errors),
                exact_mask_confusions=cm_n,example=key)
    (HERE/"paired-example-audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(result)

if __name__=="__main__":
    main()
