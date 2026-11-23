"""Exercise move identity, direction, score-scale invariance and displaced-guide controls."""
import torch
from boundary_refinement.models.operators.directional_veto import DirectionalColors, MoveDecision, conflicts
from boundary_refinement.data.staging.weatherproof_files import OUT,write

def run():
    torch.manual_seed(78)
    h,w=32,64
    z=torch.randn(1,19,h,w)
    raw=torch.rand(1,3,h,w)
    off=torch.zeros(h,w,2,dtype=torch.long);off[:,:,1]=2
    ids=torch.arange(h*w)
    lookup=ids//w*w+((ids%w)+2).clamp_max(w-1)
    colors=DirectionalColors(raw)
    p,m,active,x=conflicts(z,lookup,off,colors)
    assert torch.equal(active,(p.ravel()!=m.ravel()).nonzero().flatten())
    assert torch.equal(p.ravel()[lookup],m.ravel())
    assert torch.isfinite(x).all()
    constant=DirectionalColors(torch.full_like(raw,.25))
    assert torch.count_nonzero(conflicts(z,lookup,off,constant)[-1][:,2:])==0
    assert torch.allclose(conflicts(7*z+2,lookup,off,colors)[-1],x,rtol=2e-5,atol=2e-5)
    orig=DirectionalColors(raw.roll((17,37),(-2,-1)))
    assert torch.allclose(x[:,4],orig.pair(active,off,(0,0))[0])
    head=MoveDecision("aligned")
    with torch.no_grad():
        head.linear.bias.fill_(-100)
        assert torch.equal(head.decide(p,m,active,x),p)
        head.linear.bias.fill_(100)
        assert torch.equal(head.decide(p,m,active,x),m)
    _,unchanged,empty,empty_x=conflicts(z,ids,torch.zeros_like(off),colors)
    assert torch.equal(unchanged,p) and empty_x.shape==(0,6) and not len(empty)
    write(OUT/"operator-check.json",dict(device="CPU",checks=["conflict set","released displacement",
          "finite features","constant-color null","logit scale and offset invariance","wrong-location RGB control",
          "veto-all identity","accept-all identity","empty conflict identity"]))
    print("Directional operator invariants passed",flush=True)
if __name__=="__main__":run()
