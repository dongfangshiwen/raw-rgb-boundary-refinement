"""Reproduce the two Fig. 1(b) crop counts from native class IDs; no training or CUDA is required.
Usage from this directory: python recheck_figure01.py PATH/TO/00000000_labellevel3Ids.png
Optional: --s1 "PATH/Supplementary Data S1.zip" checks full and radius-six native confusion counts.
"""
import argparse,hashlib,json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(path,s1):
 spec=json.loads((B/'figure-01-provenance.json').read_text(encoding='utf8'))
 assert sha(path)==spec['input']['mask_sha']
 predicted={}
 for k in ['released','raw_guided']:
  row=spec['masks'][k];assert sha(B/row['path'])==row['png_sha256'];predicted[k]=np.array(Image.open(B/row['path']))
 a,b=predicted['released'],predicted['raw_guided']
 lut=np.full(256,255,dtype=np.uint8)
 for k,v in spec['idd_level3_to_cityscapes'].items():lut[int(k)]=int(v)
 gt=lut[np.array(Image.open(path))];h,w=gt.shape;assert a.shape==b.shape==gt.shape
 valid=gt<19;edge=np.zeros_like(valid);dv=(gt[:,1:]!=gt[:,:-1])&valid[:,1:]&valid[:,:-1];edge[:,1:]|=dv;edge[:,:-1]|=dv;dv=(gt[1:]!=gt[:-1])&valid[1:]&valid[:-1];edge[1:]|=dv;edge[:-1]|=dv
 y,x=np.nonzero(edge);ind=np.argmin((y-(h-1)/2)**2+(x-(w-1)/2)**2);cx,cy=int(x[ind]),int(y[ind])
 x0=max(0,min(w-320,cx-160));y0=max(0,min(h-240,cy-120));assert [x0,y0,x0+320,y0+240]==spec['measurement']['coordinates_zero_based_xyxy_end_exclusive']
 ex=np.pad(edge,((0,0),(6,6)));wide=np.logical_or.reduce([ex[:,k:k+w] for k in range(13)]);ey=np.pad(wide,((6,6),(0,0)));band=np.logical_or.reduce([ey[k:k+h] for k in range(13)])&valid
 def tally(mask):
  return dict(band_pixels=int(mask.sum()),corrected=int(((a!=gt)&(b==gt)&mask).sum()),new_errors=int(((a==gt)&(b!=gt)&mask).sum()),changed_both_wrong=int(((a!=gt)&(b!=gt)&(a!=b)&mask).sum()),unchanged=int(((a==b)&mask).sum()))
 roi=np.zeros_like(band);roi[y0:y0+240,x0:x0+320]=True
 for name,mask in [('crop',band & roi),('whole_frame',band)]:
  counts=tally(mask);assert counts==spec['measurement'][name],(name,counts);print(name,counts)
 if s1:
  with ZipFile(s1) as z,np.load(BytesIO(z.read('idd/mixed-counts.npz')),allow_pickle=False) as m:
   fi=m['ids'].tolist().index(spec['input']['id']);si=m['seeds'].tolist().index(101)
   for arm,mask in [('author',a),('raw',b)]:
    for j,vv in enumerate([valid,band]):
     matrix=np.bincount(gt[vv].astype(np.int32)*19+mask[vv],minlength=361).reshape(19,19)
     np.testing.assert_array_equal(matrix,m['counts'][fi,j,m['arms'].tolist().index(arm),si])
   print('Verified both archived native confusion matrices for both methods.')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('label_path',type=Path);parser.add_argument('--s1',type=Path);args=parser.parse_args();run(args.label_path,args.s1)
