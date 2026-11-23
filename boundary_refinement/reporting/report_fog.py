"""Write the human-readable supplement and a portable, NumPy-only scoring archive."""
from boundary_refinement.paths import SOURCE_ROOT
from boundary_refinement.reporting.source_bundle import source_files
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape
import json
from boundary_refinement.evaluation.semantic import fog_counts as stats
from boundary_refinement.data.staging.weatherproof_files import OUT, WORK, read, write, sha

NAMES={"released":"骨干原始输出","paper":"单用新稿解码器","segfix":"单用 SegFix","gaussian":"单用高斯插值",
       "gaussian_segfix":"高斯插值 → SegFix","paper_segfix":"新稿解码器 → SegFix","margin":"语义置信度筛选",
       "aligned":"语义＋对齐 RGB 筛选","shifted":"语义＋错位 RGB 筛选"}
STUDY_MODULES = (
    "boundary_refinement.data.cohorts.reserve_drivable",
    "boundary_refinement.data.cohorts.reserve_weatherproof",
    "boundary_refinement.data.cohorts.screen_dense_clips",
    "boundary_refinement.data.staging.weatherproof_files",
    "boundary_refinement.evaluation.controls.audit_fog",
    "boundary_refinement.evaluation.controls.check_directional",
    "boundary_refinement.evaluation.controls.check_drivable_lut",
    "boundary_refinement.evaluation.controls.validate_fog",
    "boundary_refinement.evaluation.paired.paired_photo",
    "boundary_refinement.evaluation.semantic.evaluate_dense_clips",
    "boundary_refinement.evaluation.semantic.fog_counts",
    "boundary_refinement.models.operators.directional_veto",
    "boundary_refinement.reporting.figures_fog",
    "boundary_refinement.reporting.manuscript_rgb",
    "boundary_refinement.reporting.report_fog",
    "boundary_refinement.training.train_directional",
)


def estimate(c):
    s=f"{c['pp']:+.3f}"
    if c["ci95"] is not None:s+=" ["+", ".join(f"{u:+.3f}" for u in c["ci95"])+"]"
    return s
def table(columns,rows):
    return "<div class='scroll'><table><thead><tr>"+"".join(f"<th>{escape(str(x))}</th>" for x in columns)+"</tr></thead><tbody>"+"".join(
        "<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table></div>"
def fig(name,description):
    return f"<figure><a href='figures/{name}'><img src='figures/{name}' alt='{escape(description)}'></a><figcaption>{escape(description)}</figcaption></figure>"

def build():
    results=read(OUT/"results.json")["strata"];dense=read(OUT/"dense-roads/results.json")
    ver=read(OUT/"verification.json");t=read(OUT/"training.json");card=read(OUT/"evaluation.json")
    grid=[
        ("BDD 可行驶区域，官方天气标签",results["bdd_tagged"]),
        ("BDD 可行驶区域，预先目视筛选",results["bdd_visible"]),
        ("WeatherProof，雾及混合天气",results["wp_fogmix"]),
        ("WeatherProof，仅目视雾",results["wp_fog"]),
        ("WeatherProof，整个公开验证分区",results["wp_validation"]),
        ("IDD-AW，已知录像诊断",results["idd_seen_drives"])]
    if dense["groups"]:grid.insert(0,("BDD 19 类，新视频关键帧",dense))
    header="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实雾场景的语义边界复验</title>
    <style>body{margin:0;background:#f4f6f5;color:#182e34;font:16px/1.72 'Segoe UI','Microsoft YaHei',sans-serif}main{max-width:1090px;margin:auto;background:#fff;border-top:7px solid #24676b;padding:40px min(5vw,65px) 65px}h1,h2,h3{line-height:1.3;color:#163e44}h1{font-size:34px}h2{font-size:23px;margin:40px 0 12px}h3{font-size:18px}p{margin:.65em 0}.lede{font-size:18px;margin-bottom:24px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%;margin:12px 0 16px;font-variant-numeric:tabular-nums}th,td{padding:7px 9px;border-bottom:1px solid #d8e0de;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#244f55;background:#ecf3f2}small,figcaption{color:#486369}a{color:#19656d}figure{margin:20px 0}img{max-width:100%;height:auto}figcaption{font-size:14px}details{margin:12px 0}summary{cursor:pointer;font-weight:600}footer{margin-top:30px;border-top:1px solid #b9cccd;padding-top:14px}</style><main>
    <h1>真实雾场景的语义边界复验</h1><p class="lede">原图颜色解码、SegFix 边界位移，以及允许拒绝位移的三种训练对照。所有目标数据均只用于推断，分割骨干与原解码器均已冻结。</p>"""
    parts=[header,"<h2>原解码器与位移模块各自带来了什么</h2>",
        "<p>表中均为原尺寸、距有效参考边界不超过六像素区域内的 mIoU，单位为百分比。每行内部采用相同标注和图像。BDD 分割真值的 19 类、可行驶区域的二类、WeatherProof 精确对齐的三类及 IDD 的 17 类分别计算。各行的绝对数值不适合直接排名。</p>"]
    arms=("released","paper","segfix","gaussian_segfix","paper_segfix","aligned")
    for target in card["targets"]:
        rows=[]
        for label,v in grid:
            reg=v["primary"]["boundary"]
            if not reg.get("informative_groups"):continue
            s=reg["targets"][target]["scores"]
            rows.append([label,f"{reg['informative_groups']} / {v['groups']}",*[f"{s[a]:.3f}" for a in arms]])
        parts.append("<h3>"+("Mask2Former" if target=="mask2former" else "SegFormer-B1")+"</h3>")
        parts.append(table(["数据队列","有边界组 / 全部组",*[NAMES[a] for a in arms]],rows))
    parts.append(fig("paired-boundary-contrasts.png","串接两种方法相对三个对照的差值。各面板采用各自的百分点刻度，有足够独立组时画出 95% 配对区间。"))
    parts.append("<h2>能否将互补性写入论文</h2><p>预定检验同时要求串接超过单用解码器、单用 SegFix 和高斯插值后接 SegFix。BDD 道路二分类与 WeatherProof 的结果均未通过全部三项。四段新增多类道路视频中，边界像素包含 12 个实际观测到的类别。Mask2Former 单用新稿解码器比原始输出高 1.46 个百分点，单用高斯插值与新稿解码器相差约 0.01 个百分点。这批视频能描述现象，尚不足以确定普遍互补或排除普通插值的解释。</p>")
    for metric,title in (("boundary","边界区"),("full","整图")):
        rows=[]
        for label,v in grid:
            reg=v["primary"][metric]
            if not reg.get("informative_groups"):continue
            c=reg["targets"]["mask2former"]["contrasts"]
            rows.append([label,*[estimate(c[n]) for n in ("paper minus released","paper_segfix minus paper","paper_segfix minus segfix","paper_segfix minus gaussian_segfix")]])
        parts.append("<details"+(" open" if metric=="boundary" else "")+"><summary>Mask2Former "+title+"差值及区间（百分点）</summary>")
        parts.append(table(["队列","原解码器 − 骨干","串接 − 原解码器","串接 − SegFix","串接 − 高斯＋SegFix"],rows)+"</details>")
    parts.append("<h2>训练得到的位移筛选器</h2><p>冻结原有权重与 SegFix 的两像素位移，只在位移两端的预测类别不一致时做一次二选一。源域中恰有一端符合标注的像素产生二元监督。语义对照读取两端归一化的类别得分差。完整模型再读取原 RGB 与 5×5 平滑 RGB 在位移方向及反方向的颜色差。错位对照把相同颜色图垂直移 17 像素、水平移 37 像素，参数量与完整模型相同。三种方法都用固定的 0.5 阈值。</p>")
    parts.append(f"<p>MUSES 的 96 帧、9 组均经过检查，三个种子均有 87 帧、8 组提供可判对错的位移。每次等概率抽取有监督的录像及帧，每步 2,048 个像素。随机种子 101、211、307 分别训练三种模型，各 300 步，累计 9 次训练。运行环境为 {t['gpu']}、PyTorch {escape(t['torch'])}、CUDA {t['cuda']}。</p>")
    rows=[]
    for label,v in grid:
        reg=v["primary"]["boundary"]
        if reg.get("informative_groups"):
            c=reg["targets"]["mask2former"]["contrasts"]
            rows.append([label,*[estimate(c["aligned minus "+a]) for a in ("paper_segfix","margin","shifted","paper")]])
    parts.append(table(["完整筛选相对各对照","直接接受位移","仅语义","相同容量、颜色错位","原解码器"],rows))
    changes=results["bdd_tagged"]["primary"]["boundary"]["targets"]["mask2former"]["changes_vs_paper"]
    parts.append("<p>BDD 道路边界中，取三种种子的平均像素数，直接位移修正约 %s 个原错误，同时使约 %s 个原正确像素出错。加入对齐颜色筛选后，这两项分别为 %s 和 %s。这个对照解释了为什么筛选能缓解位移带来的损失，以及单用原解码器仍可取得更高的分数。</p>"%tuple(f"{v:,.0f}" for a in ("paper_segfix","aligned") for v in changes[a][:2]))
    parts.append(fig("road-prediction-audit.png","按筛选表顺序预留的第一张样本。框的位置仅按标注边界密度选定。图中标出位移纠正、破坏或改成另一错误类别的像素，细线的宽度为原始预测所决定。"))
    parts.append("<h2>同一画面上的两个输出</h2><p>使用 WeatherProof 实拍配对图逐帧计算 PSNR，然后在每个相机组内取平均，最终对相机组等权。图像长边最多 512 像素。RGB 对照包含原输入、作者 MCT、作者 PGL-Net-T，以及先前由 23 个 BeDDE 城市拟合的固定混合系数 0.48154。此参数在本轮数据上冻结。</p>")
    original=read(OUT/"manuscript-rgb-results.json")["strata"]
    parts.append("<p>另将论文原有的局部候选选择、双 RGB 校正头与五节点亮度曲线按三套原权重完整推断。此项追加检验单独登记。下表列出了原稿 RGB 输出与后来拟合的单系数输出。原稿 RGB 的三个训练种子为 32003、32017、32029。先在每个分支内对其自身的三个源域种子求平均，再按相机配对比较两个输出。</p>")
    rows=[]
    for label,key in (("公开验证全部天气","wp_validation"),("雾与混合天气","wp_fogmix"),("仅目视雾","wp_fog")):
        for subset,p in (("全部相机组",original[key]["all"]),("有三类交界的同一批组",original[key]["matched_boundary"])):
            rows.append([label,subset,p["groups"],*[f"{p['scores'][m]:.2f}" for m in ("input","mct","manuscript","scalar","pgl")]])
    parts.append(table(["队列","共同覆盖","相机组","原输入","MCT","原稿完整 RGB 支路","冻结单系数","PGL-Net-T"],rows))
    match=results["wp_fogmix"]["primary"]["boundary"]["rgb_on_identical_groups"]
    both=match["joint_counts"]["mask2former/paper_segfix+scalar vs segfix+mct"]
    parts.append(f"<p>在同时具有可评三类交界和清晰参考的 {match['groups']} 组中，按组分别比较“串接分割＋校准 RGB”与“SegFix＋MCT”。两个指标均提高的有 {both['both_strictly_positive']} 组，仅 RGB 提高的有 {both['rgb_only']} 组，仅分割提高的有 {both['semantic_only']} 组，两个差值均未大于零的有 {both['neither']} 组。这里只计算严格正差值。</p>")
    actual=original["wp_fogmix"]["matched_boundary"]["joint_counts"]["mask2former/paper+manuscript vs segfix+mct"]
    parts.append(f"<p>若按论文原有的两个输出直接比较，“原解码器＋原稿 RGB”相对“SegFix＋MCT”在这六组中同时提高的有 {actual['both']} 组，仅 RGB 提高的有 {actual['rgb_only']} 组，仅分割提高的有 {actual['semantic_only']} 组，两项均未提高的有 {actual['neither']} 组。</p>")
    parts.append("<p>在全部 31 个目视雾或混合天气相机组，原稿完整 RGB 支路相对公开 MCT 的 PSNR 差为 −3.24 dB，按相机组配对的 95% 区间为 [−4.48, −2.08]。单系数版本相对 PGL-Net-T 的差为 +0.78 dB，区间为 [−0.41, +1.97]。这些结果支持把原图引导的语义边界设为主张，并在文中标明 RGB 支路的具体版本。</p>")
    parts.append(fig("weatherproof-paired-rgb.png","同一幅实拍退化图对应的 RGB 输出与清晰参考。展示样本在读取模型输出前按顺序及标注覆盖选定，图上给出该样本的 PSNR。"))
    dedup=ver["weatherproof_duplicates"]
    parts.append("<h2>独立性、数据核验与复现</h2><ul>")
    parts.append("<li>BDD 可行驶区域队列采用官方雾标签，排除了过往筛选或评估用过的文件名前缀，一段视频取一张关键帧。共 135 段，其中 130 段具有可计分边界。额外的 18 段视觉雾子集有 17 段可评边界。可行驶区域还涉及车辆能否驶入一段道路，与 Cityscapes 的 road 定义存在差异。</li>")
    parts.append(f"<li>另一批 BDD 19 类候选来自 2,048 个全新视频前缀，以事先固定的图像统计量和哈希抽样选出 288 张缩略图，放大审查 22 张，锁定 {dense['groups']} 张再取得像素标注。天气依据一名观察者的图像判断，其中有仅远景受影响的场景。独立性按视频标识界定，路线是否重叠未知。少于六组时只给点估计。</li>")
    parts.append(f"<li>WeatherProof 覆盖 60 个序列目录、54 个按相机名归并的组和 180 对图像。雾与混合天气样本来自 31 组、96 对，包含从训练分区保留下来、仅供本项目评估的相机。三类为道路、建筑、天空，其他真实类别作忽略处理，模型预测到其余类别仍可构成漏分。有效边界必须连接两个已映射类别。另报告树木映射到植被的四类敏感性分析。全部 180 项中有 {dedup['unique_adverse']} 张不同退化图、{dedup['unique_clear']} 张不同清晰参考和 {dedup['unique_semantic_masks']} 张不同类别标注，重抽样以相机组为单位。</li>")
    parts.append("<li>BDD 可行驶区域的整数 PNG 与工具包枚举使用了不同编号。已逐像素比对全部 135 份配套彩色标注，共 124,416,000 个像素。黑色背景、红色直接可行驶、蓝色可变道驶入在 PNG 中分别对应 0、1、2。全部道路评分已按真实编码重算。另在原 IDD 的 62 帧逐项复现六种历史输出；WeatherProof 与 IDD 在两次推断中的计数完全一致。</li>".replace("；","。"))
    parts.append(f"<li>先按视频或相机组合并整数混淆矩阵，再由每个种子分别计算 mIoU 并平均。点估计为固定类别宏平均；并报告仅选真实标签有支持类别的敏感性结果。95% 区间按组配对重抽 5,000 次，属于各对照的逐项区间。已对 {ver['native_PNGs_checked']} 张存盘类别图重新生成边界、重算整图与边界混淆矩阵，并从已存 RGB 图复核 PSNR。</li></ul>".replace("；","。"))
    parts.append("<p>数据与方法来源见 <a href='https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst'>BDD100K 发布说明</a>、<a href='https://arxiv.org/html/2403.14874v2'>WeatherProof 原论文</a>及 <a href='https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf'>SegFix 原论文</a>。</p>")
    parts.append("<footer><a href='replication.zip'>下载本轮训练权重、清单和可复算的逐帧计数</a>　<a href='results.json'>全部对照和区间</a>　<a href='dense-roads/results.json'>新增 19 类结果</a>　<a href='manuscript-rgb-results.json'>原稿双输出</a>　<a href='verification.json'>数据核验</a></footer></main></html>")
    (OUT/"index.html").write_text("\n".join(parts),encoding="utf-8")

def package():
    dest=OUT/"replication.zip"
    included=[*OUT.glob("*.json"),*OUT.glob("*.pth"),*OUT.glob("counts/*/*.npz"),*(OUT/"dense-roads").glob("*.json"),
              *(OUT/"dense-roads/counts").glob("*.npz")]
    included=[p for p in included if p.name!="archive-check.json"]
    readme="""The workspace used Python and CUDA PyTorch for inference. To recompute all headline scores and cluster intervals from the archived integer counts, use Python 3 plus NumPy and run

  python -m boundary_refinement.evaluation.semantic.fog_counts --root .

results.json and dense-roads/results.json contain both negative and positive contrasts. Per-frame matrices use group-paired sampling. Native model inference and source fitting require the existing project dependencies and the indicated released checkpoints. Absolute asset checksums, release links, three source seeds and fixed decisions are provided in the manifests. Run the evaluation modules from the repository root. The zip contains the nine newly fitted move-decision heads. External large checkpoints and the original natural images require their respective sources.
"""
    tmp=dest.with_suffix(".part")
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(included):z.write(p,p.relative_to(OUT).as_posix())
        for path in source_files(STUDY_MODULES):
            z.write(path, path.relative_to(SOURCE_ROOT).as_posix())
        z.writestr("REPRODUCE.txt",readme)
    tmp.replace(dest)
    with TemporaryDirectory() as temp:
        d=Path(temp)
        with ZipFile(dest) as z:
            assert z.testzip() is None
            z.extractall(d)
        stats.run(d);stats.rescore_dense(d/"dense-roads");stats.rescore_manuscript_rgb(d)
        for sub in ("results.json","dense-roads/results.json","manuscript-rgb-results.json"):
            assert read(d/sub)==read(OUT/sub)
    write(OUT/"archive-check.json",dict(sha256=sha(dest.read_bytes()),bytes=dest.stat().st_size,
          reproduced_from_extracted_counts_and_RGB_errors=["results.json","dense-roads/results.json","manuscript-rgb-results.json"]))

if __name__=="__main__":
    build();package()
