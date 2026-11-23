"""Register the five published NH-HAZE2 test pairs for independent transfer checks."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

import requests
import torch

from boundary_refinement.data.staging.prepare_rw2ah import decode
from boundary_refinement.data.provenance import digest, now, read
from boundary_refinement.data.real_detail_data import image_tensor, long_edge
from boundary_refinement.data.real_haze_data import overlap_check
from boundary_refinement.data.staging.stage_fusion_holdouts import witness
from boundary_refinement.data.staging.stage_lmhaze import TRAIN_CARD, reference
from boundary_refinement.data.staging.stage_postfusion_holdout import observed_refs
from boundary_refinement.training.structure_transfer import ROOT
from boundary_refinement.training.train_regional_guidance import atomic_write, torch_save


CARD = ROOT/"nhhaze2_manifest.json"
PAPER = "https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Ancuti_NTIRE_2021_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2021_paper.html"
RELEASE = "https://github.com/heydarimo/Night-Time-Dehazing#reproducibility-of-paper-results"
FILES = {
    "061": ("1jhWIxT9t5TTzEtxhlUYajmTwcGNtHvSx","1HGZNoK4XPjQc3kxY9VqlB12vVpOdjee0"),
    "062": ("1zkXzn4fDzyq2yrAasjkOEOyYKzGnbOKu","1mWOgmC8k1JLD-fr5s3HZaHuIu3pHYuEf"),
    "063": ("11CGNcURndZlQqZIcsVlFNriJPXkwYyao","1hCIsQXYQ864NWHaEeWAebmEAeETRzu0_"),
    "064": ("15hmLs5buP2ZQWzUVd5NKySg58c7qKJy3","1av939IOTm5Xeb_k2AuIUURt4-rUAsbWP"),
    "065": ("100Mrc3cCc21kyhpHLKz7QDoiqdof-LW6","1oQocJWsoVQxKHN03ahTSHWQWWv2P7Zya"),
}


def fetch(identity):
    url = f"https://drive.usercontent.google.com/download?id={identity}&export=download&confirm=t"
    for attempt in range(5):
        try:
            response = requests.get(url, timeout=(12,25))
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            if attempt==4:
                raise
            print(dict(retry=identity,error=type(exc).__name__),flush=True)


def stage():
    ROOT.mkdir(exist_ok=True,parents=True)
    if CARD.exists():
        record = read(CARD)
        if digest(record["file"])!=record["SHA256"]:
            raise ValueError("NH-HAZE2 cache changed")
        return record
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {key: [pool.submit(fetch,file) for file in names] for key,names in FILES.items()}
        raw = {key: [job.result() for job in jobs] for key,jobs in futures.items()}
    candidates, pairs, provenance = [],[],{}
    for seq,(fog,clean) in raw.items():
        a,b = decode(fog),decode(clean)
        if a.size!=b.size:
            raise ValueError("NH-HAZE2 pair reference geometry differs")
        x,y = (image_tensor(long_edge(v,1024)) for v in (a,b))
        pairs.append(dict(id=seq,hazy=x,clean=y))
        candidates.append(witness(seq,b,sha256(clean).hexdigest()))
        provenance[seq] = dict(file_ids=FILES[seq], sha256=[sha256(v).hexdigest() for v in (fog,clean)],shape=list(a.size))
    lmhaze = [reference(row) for row in read(TRAIN_CARD)["scenes"] if row["role"]=="fit"]
    flags, nearest = overlap_check([*lmhaze,*observed_refs()],candidates)
    excluded = {r["id"] for r in flags}
    kept = [r for r in pairs if r["id"] not in excluded]
    if not kept:
        raise ValueError("No independent NH-HAZE2 image pair passed overlap screening")
    target = ROOT/"nhhaze2_1024.pth"
    torch_save(target,kept)
    card = dict(paper=PAPER, release=RELEASE, checked=provenance,count=len(kept),excluded=flags,nearest=nearest,
                file=str(target),SHA256=digest(target),scope="Published HistoFusionNet reproduction copy, NH-HAZE2 test 061–065. Exclude prior-project near duplicates. Neither challenge-server test nor known author pretraining exposure is inferred.",finished_utc=now())
    atomic_write(CARD,card)
    print(dict(paired=len(kept),excluded=list(excluded)),flush=True)
    return card


if __name__ == "__main__":
    stage()
