"""Verify the target comparisons from stored counts and present both audits in one report."""
from boundary_refinement.paths import WORKSPACE
from collections import defaultdict
from html import escape
from pathlib import Path
import argparse
import json

import numpy as np


ROOT = WORKSPACE / "experiment_exports"
FIXED = ROOT / "fixed-range-audit-v106"
PEER = ROOT / "featup-control-v107"
HISTORY = ROOT / "matched-color-decoding-v104/manuscript-extension"
LABELS = {"bdd":"BDD100K，不限天气", "idd":"IDD-AW，雾天",
          "wp_fog":"WeatherProof，纯雾", "wp_fog_mixed":"WeatherProof，雾及混合天气"}
NAMES = {"mask2former":"Mask2Former","segformer":"SegFormer-B1","ade5":"SegFormer-B0（ADE，5 个对应类）"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ci(record):
    return f"{record['pp']:+.3f} [{record['ci95'][0]:+.3f}, {record['ci95'][1]:+.3f}]"


def table(headers, rows):
    head="".join(f"<th>{escape(str(x))}</th>" for x in headers)
    body="".join("<tr>"+"".join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return '<div class="scroll"><table><thead><tr>'+head+"</tr></thead><tbody>"+body+"</tbody></table></div>"


def verify(fixed, peer):
    """Independent paired bootstrap, using group confusion totals and a new calculation path."""
    max_error=0.; endpoints=0; images_checked=0
    for panel in ("bdd","idd"):
        files=sorted((PEER/"counts"/panel).glob("*.npz"))
        for suite in ("mask2former","segformer"):
            bank=defaultdict(lambda:0)
            for file in files:
                with np.load(file,allow_pickle=False) as new, np.load(FIXED/"counts"/panel/file.name,allow_pickle=False) as prev:
                    assert new["id"].item()==prev["id"].item() and new["group"].item()==prev["group"].item()
                    cm=new[suite]; earlier=prev[suite]
                    for a,b in ((0,0),(1,1),(2,5)):
                        np.testing.assert_array_equal(cm[:,a],earlier[:,b])
                    bank[new["group"].item()]+=cm
            images_checked+=len(files)
            groups=sorted(bank)
            array=np.stack([bank[g] for g in groups]).astype(np.int64)
            columns=[i for i in range(19) if panel!="idd" or i not in (9,16)]
            # For speed retain only class-wise sufficient statistics before resampling.
            hit=array.diagonal(axis1=-2,axis2=-1)
            union=array.sum(-1)+array.sum(-2)-hit
            stats=np.stack([np.take(v,columns,axis=-1) for v in (hit,union)],axis=-1)
            for j,region in enumerate(("full","boundary")):
                endpoints_src=peer[panel][suite][region]
                names=endpoints_src["names"];assert names==groups
                stat=stats[:,j].astype(float)
                def miou(s):
                    return 100*np.divide(s[...,0],s[...,1],out=np.zeros_like(s[...,0]),where=s[...,1]>0).mean((-1,-2))
                full=miou(stat.sum(0))
                weights=np.random.default_rng(10517).multinomial(len(groups),np.full(len(groups),1/len(groups)),5000)
                trials=[]
                for chunk in np.array_split(weights,100):
                    pooled=np.einsum("ng,gaskt->naskt",chunk,stat,optimize=True)
                    trials.append(miou(pooled))
                boot=np.concatenate(trials)
                for other,label in ((0,"released"),(1,"fixed4"),(3,"featup_frozen"),(4,"featup_fit"),(5,"featup_fit_gated")):
                    actual=[full[2]-full[other],*np.percentile(boot[:,2]-boot[:,other],[2.5,97.5])]
                    r=endpoints_src["targets"][suite]["contrasts"]["paper - "+label]
                    error=float(np.max(np.abs(np.array(actual)-[r["pp"],*r["ci95"]])))
                    if other<2:
                        prev=fixed[panel][suite][region]["targets"][suite]["contrasts"]["paper - "+label]
                        np.testing.assert_allclose([r["pp"],*r["ci95"]],[prev["pp"],*prev["ci95"]],atol=1e-9,rtol=0)
                    assert error<1e-9,(panel,suite,region,label,error)
                    max_error=max(max_error,error);endpoints+=1
    return dict(paired_endpoints=endpoints,max_abs_error_pp=max_error,road_image_backbone_pairs=images_checked)


def main():
    raw=read(FIXED/"results.json");fixed=raw["panels"];peer=read(PEER/"results.json")["panels"]
    run=read(PEER/"training.json")
    check=verify(fixed,peer)
    # Match the primary reference means and intervals to the previous implementation on every cohort.
    history=read(HISTORY/"results-view-audited.json")["panels"];road=read(HISTORY/"road-recheck-results.json")["idd"]
    for tag,suites in fixed.items():
        for suite,regions in suites.items():
            src=road if tag=="idd" else history["bdd_unrestricted" if tag=="bdd" else tag]["suites"][suite]
            for reg in ("boundary","full"):
                a=regions[reg]["targets"][suite]["contrasts"]["paper - released"]
                b=src[reg]["targets"][suite]["contrasts"]["paper - released"]
                np.testing.assert_allclose([a["pp"],*a["ci95"]],[b["pp"],*b["ci95"]],atol=1e-9,rtol=0)
    assert all(len(r["loss"])==r["steps"]==320 and np.isfinite(r["loss"]).all() for r in run["runs"])
    check.update(v105_endpoint_reproduction=True,source_seeds=[r["seed"] for r in run["runs"]],
                 checkpoint_steps=[r["steps"] for r in run["runs"]])
    (PEER/"verification.json").write_text(json.dumps(check,indent=2)+"\n",encoding="utf-8")

    factorial=[];primary=[];full_rows=[]; ablation=[]
    for tag,suites in fixed.items():
        for suite,res in suites.items():
            r=res["boundary"]["targets"][suite]["contrasts"]
            factorial.append([LABELS[tag],NAMES[suite],f"{res['images']} / {res['groups']} / {res['boundary']['groups']}",
                              ci(r["paper - released"]),ci(r["paper - fixed4"]),
                              ci(res["full"]["targets"][suite]["contrasts"]["paper - fixed4"])])
            ablation.append([LABELS[tag],NAMES[suite],ci(r["both_v101 - fit_spatial"]),ci(r["both_v101 - fit_range"])])
    for tag,suites in peer.items():
        for suite,regions in suites.items():
            for reg,rows in (("boundary",primary),("full",full_rows)):
                s=regions[reg]["targets"][suite]
                rows.append([LABELS[tag],NAMES[suite],f"{s['scores']['paper']:.3f}",f"{s['scores']['featup_fit']:.3f}",
                             ci(s["contrasts"]["paper - featup_fit"]),ci(s["contrasts"]["paper - featup_fit_gated"]),
                             ci(s["contrasts"]["paper - featup_frozen"])])
    ext_heads=["数据","分割骨干","稿件方法（%）","微调 FeatUp（%）","稿件 − 微调 FeatUp","稿件 − 微调 FeatUp 加相同门控","稿件 − 冻结 FeatUp"]
    n=sum(v["frames"] for v in read(FIXED/"evaluation.json")["panels"].values())
    title="原图引导语义边界解码的比较与消融"
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{color:#18242c;background:white;font:16px/1.7 'Microsoft YaHei UI','Segoe UI',sans-serif}}body{{max-width:1120px;margin:42px auto;padding:0 24px 45px}}
h1,h2{{line-height:1.35;color:#153946}}h1{{font-size:32px}}h2{{font-size:22px;margin:34px 0 9px}}p{{max-width:96ch;margin:9px 0}}a{{color:#005f73}}
table{{border-collapse:collapse;font-size:14px;width:100%;font-variant-numeric:tabular-nums;text-align:left}}th,td{{padding:9px 10px;border-bottom:1px solid #dce4e6}}
th{{border-top:2px solid #315860;background:#f1f6f5}}td:nth-child(n+3){{white-space:nowrap}}.scroll{{overflow:auto}}.note{{font-size:14px;color:#44545d}}
@media print{{body{{margin:0;font-size:10pt;max-width:none}}h1{{font-size:19pt}}h2{{font-size:13pt}}table{{font-size:8pt}}th,td{{padding:3px}}tr{{break-inside:avoid}}.scroll{{overflow:visible}}}}</style>
<h1>{title}</h1><p>稿件解码器在两个道路骨干的真实雾录像上改善了语义边界。对齐的原图颜色足以解释相当一部分改善。各组在学习四个颜色强度与空间指数后，相对固定四尺度核的边界差值区间均跨零。外部学习算子在其中一个骨干与数据组合上反超。</p>
<h2>同一组原图和分数，学习颜色权重带来了多少改善？</h2>
<p>所有差值单位均为 mIoU 百分点，方括号为按采集组配对的 95% 百分位区间。BDD100K 使用文件名前缀分组，IDD-AW 使用行驶记录，WeatherProof 使用审核、合并后的相机视角。第三列最后一个数是具有有效边界标签的组数。这些目标场景此前均已评价，本轮用于同场对照。</p>
{table(["数据","分割骨干","图像 / 采集组 / 有效边界组","稿件 − 原模型（边界）","稿件 − 固定四尺度（边界）","稿件 − 固定四尺度（全图）"],factorial)}
<p class="note">BDD100K 采用本项目预留、来自官方训练划分的 128 个采集前缀，未按天气筛选。IDD-AW 使用 62 张图的 16 段录像，统计 17 个共有类。WeatherProof 在 150 类模型的输出上仅评价 5 个直接匹配的类。纯雾是雾及混合天气的子集。六像素边界带以语义参考标签计算。多组区间均为名义 95% 区间。</p>
<h2>与 FeatUp 的同条件边界比较</h2>
<p>采用作者 DINO16 预训练权重的前两个 2× 学习型双边上采样块，将相同类别分数放大四倍。两个块内部的学习层均保留。原版后两个放大阶段及特征通道专用输出层不适用当前输出尺寸与类别空间。比较对象为这一算子适配版本。<a href="https://mhamilton.net/featup.html">FeatUp 作者论文与项目</a> · <a href="https://github.com/mhamilton723/FeatUp/blob/{REVISION}/featup/upsamplers.py">固定版本作者源码</a></p>
<p>监督微调使用 MUSES 的 96 张、9 段源录像，种子为 101、211、307，每个种子 320 步。复用论文附录的源分数、图像裁剪及加权交叉熵，逐种子核对裁剪随机数终态。两个块共微调 {run['runs'][0]['parameters']:,} 个参数。所有目标推理使用预先指定的末步权重，并另报作者初始化，以及对微调结果施加稿件相同粗边界门控的结果。没有基于目标图像选权重。</p>
{table(ext_heads,primary)}
<h2>相同图像上的全图 mIoU</h2>
{table(ext_heads,full_rows)}
<h2>颜色与空间参数的分项消融</h2><p>前一列将空间指数锁定在学习值，测量进一步拟合四个颜色强度带来的变化。后一列将颜色强度锁定在学习值，测量进一步拟合空间指数带来的变化。</p>
{table(["数据","分割骨干","额外拟合颜色强度（边界）","额外拟合空间指数（边界）"],ablation)}
<h2>投稿可用的表述</h2>
<p>主贡献可写成在粗类别过渡区使用对齐原图颜色，对邻域类别分数重新加权，并保持源参数冻结向其他分割骨干迁移。正文可报告真实雾边界收益，以及 BDD100K 上 SegFormer-B1 的外部反例。参数学习的独特优势、全图普遍提升和真实雾双输出同步领先，现有对照尚未建立。若保留并行去雾，可用独立照片上的复原结果限定其适用范围。</p>
<h2>数据与校验</h2><p>固定核消融共推理 {n} 张既有图像。每张图原模型和稿件方法在两个评价区域、三个种子下的混淆矩阵均与前轮一致。对道路上的 {check['road_image_backbone_pairs']} 个图像–骨干组合，另用保存的计数独立复算了 {check['paired_endpoints']} 组差值和区间。两种稿件算子实现的少量逐像素舍入差异列在固定核结果的 checks 字段中。</p>
<p><a href="results.json">FeatUp 全部指标</a> · <a href="training.json">微调权重摘要与训练曲线数据</a> · <a href="protocol.json">FeatUp 协议</a> · <a href="../fixed-range-audit-v106/results.json">固定核消融全部指标</a> · <a href="../fixed-range-audit-v106/protocol.json">固定核消融协议</a> · <a href="verification.json">独立复算校验</a></p>
</html>"""
    (PEER/"index.html").write_text(html,encoding="utf-8")
    print(check)


REVISION="6b5a6c0e91f75e69194807128dcbc39c3084a30d"
if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
