"""CPU geometry and gradient checks independent of target images."""
import json
import torch
import torch.nn.functional as F
from boundary_refinement.models.operators import path_kernel as op
from boundary_refinement.models.operators import range_kernel as v101
from boundary_refinement.data.cohorts.holdout_path import OUT
from boundary_refinement.data.staging.weatherproof_files import write

def check():
    torch.manual_seed(104);z=torch.randn(1,19,7,9);image=torch.rand(1,3,28,36)
    st=op.Stencil(z,image);ids=st.active.nonzero().flatten();feat=st.features(ids)
    t=torch.linspace(0,1,6)
    ramp=t[:,None]*torch.tensor([.2,.8,-.3])[None]+.2
    assert op.excess_variation(torch.zeros(6,3))==0 and op.excess_variation(ramp)<1e-6
    loop=torch.tensor([0.,.4,1.,1.,.4,0.])[:,None].expand(-1,3)
    assert abs(op.excess_variation(loop).item()-2)<1e-5
    assert torch.all(feat[0]>0) and torch.allclose(feat[0].sum(-1),torch.ones(len(ids)))
    assert torch.all((feat[2]>=0)&(feat[2]<=1))
    quant=image.mul(255).round().byte()
    errors={}
    for mode,old in (("gaussian","bilinear_gaussian"),("ranges","bilinear_bounded4")):
        net=op.Decoder(mode);peer=v101.RangeDecoder(old)
        peer.load_state_dict(net.core.state_dict())
        expected=peer.correction(z,v101.prepare(z,quant))
        delta=z.new_zeros(28*36,19)
        delta[ids]=((net.weights(feat)-feat[0])[:,None]@feat[-1]).squeeze(1)
        delta=delta.reshape(28,36,19).permute(2,0,1)[None]
        # The published nine-slot implementation reduces block colors and donor sums in a
        # different float32 order. A narrow Gaussian amplifies color rounding in the logits.
        error=(expected-delta).abs().max().item();assert error<2e-5,(mode,error);errors[mode]=error
        a,b=(F.interpolate(d,(45,61),mode="bilinear",align_corners=False) for d in (delta,expected))
        reference=F.interpolate(z,(45,61),mode="bilinear",align_corners=False)
        assert torch.allclose(a,b,atol=2e-5) and torch.equal((a+reference).argmax(1),(b+reference).argmax(1))
    for mode in ("endpoint","path","shifted","permuted"):
        net=op.Decoder(mode)
        with torch.no_grad():net.additional.fill_(-100.)
        assert torch.equal(net.weights(feat),op.Decoder("gaussian").weights(feat))
        net=op.Decoder(mode);F.cross_entropy(net(feat),torch.arange(len(ids))%19).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    c=op.Stencil(torch.ones_like(z),image);assert not c.active.any()
    one=op.Stencil(z,torch.ones_like(image)).features(ids)
    assert one[1].max()==0 and one[2].max()<2e-5 and one[3].max()<2e-5
    record=dict(analytic_straight_constant=0,analytic_returning_path=2,prior_positive=True,
        max_abs_correction_vs_existing_9_slot_implementation=errors,conditional_identity_at_zero_strength=True,
        finite_gradients_four_controls=True,flat_logits_inactive=True,working_and_odd_native_raster_verified=True)
    write(OUT/"operator-checks.json",record);print(json.dumps(record))

if __name__=="__main__":check()
