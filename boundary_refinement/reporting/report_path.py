"""Publication-style figures from actual predictions and a Chinese evidence report."""
from functools import cache
from html import escape
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from boundary_refinement.evaluation.semantic import path_counts as stats

ROOT=stats.DEFAULT
EXT=ROOT/"manuscript-extension"
colors=np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),
                 (220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
                 (0,60,100),(0,80,100),(0,0,230),(119,11,32)],np.uint8)

@cache
def view_panels():
    return stats.read(EXT / "results-view-audited.json")["panels"]


def group(key, model, reg="boundary"):
    return view_panels()[key]["suites"][model][reg]

def metric(key,model,reg="boundary"):return group(key,model,reg)["targets"][model]
def interval(c):return f'{c["pp"]:+.3f} [{c["ci95"][0]:+.3f}, {c["ci95"][1]:+.3f}]' if c["ci95"] else f'{c["pp"]:+.3f}'

def forest():
    entries=[("BDD100K · all weather","Mask2Former","bdd_unrestricted","mask2former"),
             ("BDD100K · all weather","SegFormer-B1","bdd_unrestricted","segformer"),
             ("WeatherProof · visual fog","Mask2Former / 3 classes","wp_fog","mask2former"),
             ("WeatherProof · visual fog","SegFormer-B1 / 3 classes","wp_fog","segformer"),
             ("WeatherProof · visual fog","SegFormer-B0 / 5 classes","wp_fog","ade5")]
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.65),sharey=True,gridspec_kw={"width_ratios":[1,1]})
    fig.subplots_adjust(left=.31,right=.97,top=.78,bottom=.16,wspace=.23)
    names=[]
    for i,(dataset,back,key,model) in enumerate(entries):
        y=len(entries)-i-1;m=metric(key,model)
        names.append(f'{dataset}\n{back}   (n = {group(key,model)["groups"]})')
        for ax,k,col in zip(axes,("paper - released","path - paper"),("#17695F","#674892")):
            point=m["contrasts"][k];lo,hi=point["ci95"];v=point["pp"]
            ax.plot([lo,hi],[y,y],color=col,lw=1.8);ax.plot([lo,hi],[y,y],"|",color=col,ms=7)
            ax.scatter(v,y,s=32,c=col,zorder=3)
    for ax,title in zip(axes,("Original decoder vs. released output","Extra path term vs. original decoder")):
        ax.axvline(0,color="#494F54",ls="--",lw=1);ax.grid(axis="y",color="#e2e5e7");ax.set_axisbelow(True)
        ax.set_title(title,loc="left",fontsize=11,fontweight="bold",pad=12);ax.spines[["top","right","left"]].set_visible(False)
        ax.set_xlabel("Change in boundary mIoU (percentage points)",fontsize=9);ax.tick_params(axis="both",labelsize=9,length=0)
    axes[0].set_yticks(np.arange(5),names[::-1],ha="right");axes[0].tick_params(axis="y",pad=10)
    axes[0].set_ylim(-.6,4.6)
    fig.text(.025,.94,"Boundary effects on held-out scenes",fontsize=17,fontweight="bold",color="#22323b")
    fig.text(.025,.87,"Paired 95% intervals by clip prefix or audited view group. The two panels use different x-axis scales.",fontsize=10,color="#455864")
    fig.savefig(ROOT/"boundary-effects.png",dpi=220,facecolor="white",bbox_inches="tight");plt.close(fig)

def example():
    folder=EXT/"examples/bdd";img=np.asarray(Image.open(folder/"input.jpg").convert("RGB"))
    gt,before,after=(np.array(Image.open(folder/(s+".png"))) for s in ("truth","paper","path"))
    change=before!=after;valid=gt<19;hit=(after==gt)&change&valid;lost=(before==gt)&change&valid
    neutral=change&valid&~hit&~lost
    h,w=gt.shape;dh,dw=min(270,h),min(420,w)
    locations=[(i,j) for i in np.linspace(0,h-dh,10,dtype=int) for j in np.linspace(0,w-dw,12,dtype=int)]
    # Select the region using prediction disagreements only, without reference to correctness.
    y,x=max(locations,key=lambda ij:int(change[ij[0]:ij[0]+dh,ij[1]:ij[1]+dw].sum()))
    sl=np.s_[y:y+dh,x:x+dw]
    def palette(a):
        v=np.full((*a.shape,3),26,np.uint8);ok=a<19;v[ok]=colors[a[ok]];return v
    map_pixels=np.rint(.35*img[sl]).astype(np.uint8);map_pixels[hit[sl]]=(61,224,181);map_pixels[lost[sl]]=(252,129,78);map_pixels[neutral[sl]]=(192,177,209)
    fig,ax=plt.subplots(2,3,figsize=(10.7,5.65))
    fig.subplots_adjust(left=.02,right=.99,top=.93,bottom=.12,wspace=.04,hspace=.20)
    images=(img,palette(gt),palette(after),palette(before)[sl],palette(after)[sl],map_pixels)
    labels=("Input and crop location","Reference labels","Native output with path term","Original decoder · crop","Path term · same crop","Changed pixels · same crop")
    for a,im,title in zip(ax.flat,images,labels):
        a.imshow(im,interpolation="nearest");a.set_title(title,loc="left",fontsize=10,fontweight="bold");a.axis("off")
    ax[0,0].add_patch(plt.Rectangle((x,y),dw,dh,fill=False,edgecolor="#fff47d",linewidth=1.5))
    from matplotlib.patches import Patch
    legend=[Patch(facecolor=c,label=l) for c,l in (("#3de0b5","Corrected"),("#fc814e","New error"),("#c0b1d1","Changed, still wrong"),("#1a1a1a","Unlabeled in reference"))]
    fig.legend(handles=legend,loc="lower center",bbox_to_anchor=(.5,.022),ncol=4,frameon=False,fontsize=9)
    fig.savefig(ROOT/"native-prediction.png",dpi=205,facecolor="white",bbox_inches="tight");plt.close(fig)
    s=stats.read(folder/"frame.json")
    stats.write(ROOT/"example-metadata.json",dict(**s,crop_xywh=[int(x),int(y),int(dw),int(dh)],
                    whole_valid_changed=int((change&valid).sum()),whole_repaired=int(hit.sum()),whole_harmed=int(lost.sum()),whole_other=int(neutral.sum()),
                    crop_repaired=int(hit[sl].sum()),crop_harmed=int(lost[sl].sum()),crop_other=int(neutral[sl].sum())))

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
                  '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def report():
    guide_panels = stats.read(ROOT / "results.json")["panels"]
    forest();example()
    ci=stats.read(EXT/"class-influence.json");ev=stats.read(EXT/"inference-checks.json")
    fig=stats.read(ROOT/"example-metadata.json")
    suites=[("BDD100K，第二批全天气","bdd_unrestricted","mask2former","Mask2Former，19 类"),
            ("BDD100K，第二批全天气","bdd_unrestricted","segformer","SegFormer-B1，19 类"),
            ("WeatherProof，目视雾","wp_fog","mask2former","Mask2Former，3 类"),
            ("WeatherProof，目视雾","wp_fog","segformer","SegFormer-B1，3 类"),
            ("WeatherProof，目视雾","wp_fog","ade3","ADE20K SegFormer-B0，3 类"),
            ("WeatherProof，目视雾","wp_fog","ade5","ADE20K SegFormer-B0，5 类"),
            ("WeatherProof，目视雾","wp_fog","ade6","ADE20K SegFormer-B0，含石类 6 类"),
            ("WeatherProof，雾及混合","wp_fog_mixed","ade5","ADE20K SegFormer-B0，5 类")]
    main=[];ctrl=[];full=[]
    for desc,k,t,name in suites:
        m=metric(k,t);c=m["contrasts"];s=m["scores"];mf=metric(k,t,"full")
        main.append([desc,name,group(k,t)["groups"],f'{s["released"]:.3f}',f'{s["paper"]:.3f}',f'{s["path"]:.3f}',interval(c["paper - released"]),interval(c["path - paper"])])
        ctrl.append([desc,name]+[interval(c["path - "+other]) for other in ("endpoint","shifted","permuted")])
        full.append([desc,name,group(k,t,"full")["groups"],f'{mf["scores"]["paper"]:.3f}',f'{mf["scores"]["path"]:.3f}',interval(mf["contrasts"]["path - paper"])])
    comp=[]
    for desc,k,t,name in [suites[0],suites[1],suites[2],suites[3]]:
        c=metric(k,t)["contrasts"];s=metric(k,t)["scores"]
        comp.append([desc,name,*[f'{s[a]:.3f}' for a in ("paper","segfix","gaussian_segfix","paper_segfix")],interval(c["paper_segfix - gaussian_segfix"])])
    traffic=[]
    extra_road=stats.read(EXT/"road-recheck-results.json")
    for key,label in (("idd","IDD-AW 雾，62 帧"),("new_fog","BDD，本轮初筛的 4 帧"),("old_fog","BDD，上轮的 4 帧")):
        view=extra_road[key]["boundary"]
        for model,m in view["targets"].items():
            c=m["contrasts"]
            traffic.append([label,model,view["groups"],*[interval(c[a]) for a in ("path - paper","path - shifted","paper_segfix - paper","paper_segfix - segfix","paper_segfix - gaussian_segfix")]])
    gaussian=[]
    for tag,title in (("new_random","BDD100K，第一批全天气"),("new_fog","BDD100K，新筛出的道路雾"),("old_fog","BDD100K，上轮道路雾"),("idd","IDD-AW，已用录像的新一轮评估"),("weatherproof","WeatherProof，上轮对齐三类")):
        b=guide_panels[tag]["ontologies"]["primary"]["boundary"]
        for t in ("mask2former","segformer"):
            m=b["targets"][t];gaussian.append([title,t,b["groups"],*[f'{m["scores"][k]:.3f}' for k in ("gaussian","endpoint","path")],interval(m["contrasts"]["path - gaussian"]),interval(m["contrasts"]["path - shifted"])])
    design=table(["实验","源训练","新增参数 / 权重","外部验证"],[
        ["第一阶段","MUSES 雾，96 帧 / 9 段录像，3 个种子","6 种方法，共 18 次拟合","128 个全天气片段、4 个新雾片段及 3 个既有队列"],
        ["第二阶段","冻结原稿每个种子的全部权重，复用相同源样本","4 种单参数扩展，3 个种子，共 12 次拟合","另 128 个全天气片段，67 张新编号 WeatherProof 配对图像，审核后合并或剔除重叠视角"]])
    strong=ci["bdd_mask2former"];extension=EXT.name
    markup=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>原始颜色引导与语义边界解码实验</title><style>body{{font:16px/1.65 "Segoe UI","Microsoft YaHei",sans-serif;color:#23343d;max-width:1170px;margin:45px auto;padding:0 24px}}h1,h2{{line-height:1.35}}h1{{font-size:30px;max-width:900px}}h2{{font-size:21px;border-top:1px solid #b8c7cb;padding-top:20px;margin-top:33px}}a{{color:#14625b}}th,td{{padding:8px 11px;border-bottom:1px solid #dbe3e5;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}th{{background:#eaf1f1}}table{{border-collapse:collapse;width:100%;font-size:14px}}.scroll{{overflow-x:auto}}img{{max-width:100%;height:auto}}figure{{margin:18px 0}}figcaption{{font-size:14px;color:#425961}}p{{max-width:1060px}}code{{padding:1px 4px;background:#f0f3f3}}.lead{{font-size:18px}}@media print{{body{{margin:0}}h2,figure,tr{{break-inside:avoid}}}}</style>
<h1>原始颜色引导能改善哪些语义边界？</h1>
<p class="lead">原稿解码器在新的全天气道路片段上改善了边界。在 WeatherProof 目视雾图像上，将同一组权重迁移到 ADE20K 分割器后，五个直接对应类别也表现出边界收益。新增的颜色路径项在全天气道路的 Mask2Former 上通过了等参数对照。在已查看过的 IDD-AW 真实雾录像上，SegFormer-B1 的边界 mIoU 却下降了 0.084 个百分点。</p>
<p><a href="replication.zip">实验复算包与推理权重</a>　<a href="{extension}/results-view-audited.json">经视角审核的完整结果</a>　<a href="results.json">第一阶段原始结果</a></p>
<h2>训练与假设</h2>{design}
<p>普通颜色权重看查询像素与邻居平均颜色的差距。沿着查询点到邻居中心的直线取六个 RGB 样本，将五段颜色距离之和减去两端的直线颜色距离，再用一个可学习系数惩罚这部分额外变化。若同色区域中间隔有另一种颜色，该量为正。单调的直线颜色过渡给出零。空间权重、有效邻居、边界门控、输入量化和输出回投方式在对应对照中一致。</p>
<p>直线两点之间的轮廓与路径度量已有研究。这里可检验的增量是固定已训练解码器后，这个单一系数能否改善跨模型的语义边界。<a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/fmm-cvpr03.pdf">Fowlkes、Martin 与 Malik 的亲和度研究</a> · <a href="https://doi.org/10.1145/2010324.1964964">Gastal 与 Oliveira 的路径度量</a></p>
<p>两阶段每个拟合均执行 320 个 AdamW 更新，三个种子为 101、211、307。每次更新含 4096 个工作分辨率像素，均分给有效区域和标注边界采样。三张源帧在缩小后没有有效边界，第二部分从同一帧的有效区域回退采样。运行环境为 NVIDIA GeForce RTX 5060 Ti、PyTorch 2.11.0+cu128、CUDA 12.8。权重、逐步损失和源样本的抽样散列随包保存。</p>
<h2>新增参数的独立检验</h2>
<p>下表为第二阶段的边界 mIoU，单位为百分比，成对差值的单位为百分点。先按类别汇总混淆矩阵，再计算每个训练种子的 mIoU，最后取种子均值。每次重采样均对所有方法使用相同场景，括号内为 5000 次按片段或视角组重采样的点对点 95% 区间。类别集合固定，分母为零的类别贡献为零。完整结果同时给出仅对真值出现类别求均值的敏感性、逐种子差值和逐组剔除结果。</p>
{table(["数据","骨干及评分类别","有边界的组","原模型","原稿","原稿＋路径","原稿减原模型","加路径减原稿"],main)}
<figure><a href="boundary-effects.png"><img src="boundary-effects.png" alt="新增全天气和目视雾分组的边界效应与区间"></a><figcaption>左图评估原稿解码器相对发布模型的改善，右图评估增加路径项的效果。两幅图使用不同横轴刻度。</figcaption></figure>
<h2>等参数消融与整图指标</h2>
<p>四个扩展均冻结原稿解码器，各自只训练一个系数。端点对照只看两端的颜色差。错位对照只平移新增路径线索，原有颜色和分割得分对齐。置换对照保留当地四个路径值，将它们依次分配给另一个邻居。各列均为对齐路径相对相应对照的边界差值。</p>
{table(["数据","骨干及类别","减端点对照","减错位对照","减置换对照"],ctrl)}
<p>在第二批 BDD 道路图上，Mask2Former 的 19 个类别中，{strong["positive_classes"]} 类边界 IoU 上升，{strong["negative_classes"]} 类下降，其余持平。逐个剔除任何一个类别后，平均边界增量为 {strong["class_deletion_range_pp"][0]:.3f} 至 {strong["class_deletion_range_pp"][1]:.3f} 个百分点。各类样本量和变化见<a href="{extension}/class-influence.json">类别影响分析</a>。</p>
{table(["数据","骨干及评分类别","有有效标注的组","原稿整图","加路径整图","整图差值及区间"],full)}
<h2>同一邻域下与 SegFix 的组合</h2>
<p>两个解码器使用相同工作网格、四个正权重邻居、同一张原图和同一组原始骨干得分。两者都接作者发布的同一个 SegFix 位移场。BDD 的两个骨干都出现额外边界增益。WeatherProof 仅对齐道路、建筑和天空时，组合收益对骨干和视角敏感。<a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf">SegFix 原论文</a></p>
{table(["数据","骨干及类别","原稿","SegFix","高斯＋SegFix","原稿＋SegFix","原稿组合减高斯组合"],comp)}
<p>下面直接回到真实雾道路，沿用相同权重和网格。三个队列都在第二阶段设计前查看过，结果属于复查。两批 BDD 各有四个片段，分别报告点估计。IDD-AW 对齐 17 类，按 16 段行驶记录重采样。<a href="{extension}/road-recheck-results.json">道路复查完整统计</a></p>
{table(["实雾道路","骨干","组","加路径减原稿","路径减错位","原稿组合减原稿","原稿组合减 SegFix","原稿组合减高斯组合"],traffic)}
<p>在 IDD-AW 的 16 段录像上，加入路径项使 Mask2Former 的边界 mIoU 改变 −0.067 个百分点，SegFormer-B1 改变 −0.084 个百分点，后者的 95% 区间为 [−0.120, −0.037]。将两种插值都接入同一 SegFix 后，原稿组合相对拟合高斯组合在两种骨干上的区间均包含零。4 个较新的 BDD 雾片段上的正值只能作为描述性复查。</p>
<h2>真实预测与局部改动</h2>
<figure><a href="native-prediction.png"><img src="native-prediction.png" alt="固定片段的原图、标注、预测以及局部改动"></a><figcaption>第二批 BDD 的首个预留片段 {escape(fig["id"])}，Mask2Former，种子 101。黄色框在固定网格里按两种预测的分歧量选取，全程未用标注选框。右下中绿色为修正，橙色为新增错误，淡紫色为改动前后都错。原尺寸整图有效标注中共改变 {fig["whole_valid_changed"]} 像素，修正 {fig["whole_repaired"]}、新增错误 {fig["whole_harmed"]}、前后都错 {fig["whole_other"]}。<a href="{extension}/examples/bdd/path.png">原尺寸类别 ID 图</a></figcaption></figure>
<h2>数据来源与视角核查</h2>
<p>两批各 128 个 BDD100K 片段按文件名前缀分组，用固定散列从项目历次预留清单之外抽取。另从 1536 张预留图中按既往图像的天气特征筛查 432 张，放大核查 38 张，锁定 4 张有清楚距离衰减的雾图。天气筛查未使用语义标签、模型得分或预测表现。260 张新道路图的前缀和原图 SHA256 均不同。采样无法验证地理地点是否完全不同。<a href="https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst">BDD100K 官方数据说明</a></p>
<p>WeatherProof 在排除原先预留的 104 个相机根编号后，还提供 161 个带配对清晰图和语义标注的根编号。每组固定选一个片段的中间帧。根据退化图和清晰图配对目视判断，44 张有雾型能见度下降，另 23 张有混合退化。天气判定由单人完成。项目将官方训练分区中的这些场景仅用于本地冻结评估。<a href="https://github.com/UCLA-VMG/WeatherProof">WeatherProof 作者数据说明</a></p>
<p>将新旧 271 张清晰参考图做局部特征检索、几何验证和目视复查，发现两组与旧景重叠，另有两个新编号共享同一片场景。剔除和合并后，纯雾为 43 图 / 42 个视角组，含混合退化为 65 图 / 64 组。三类道路标签只使 8 个纯雾组提供有效边界。ADE20K 五类直接映射使 34 组可用于边界评估。该分割器保留全部 150 个输出类别，先完成预测，再按建筑、道路、天空、树和草评分。六类敏感性额外将 rock 对应到 stone。其余真值忽略，受评真值上的未映射预测计错。识别到的重叠已处理，局部无交集的同地点视角仍可能漏检。<a href="{extension}/overlap-01.jpg">重叠图像对</a> · <a href="{extension}/results.json">按原始编号的统计</a> · <a href="https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512">NVIDIA 官方 ADE20K 权重</a></p>
<h2>第一阶段对照、复现与投稿表述</h2>
{table(["数据","骨干","有边界的组","拟合高斯","端点扩展","路径扩展","路径减高斯","路径减错位"],gaussian)}
<p>这组数据支持以原始颜色引导的可迁移边界解码作为论文主线。可将沿途颜色变化作为一个有道路实验支持的扩展，将与 SegFix 的互补作为全天气道路上的实证结果。真实雾主骨干的新增路径收益、整幅分割提升和双输出同步改善仍需更多证据。本轮训练及评估集中在分割解码器，去雾输出沿用原稿。</p>
<p>两轮共有 30 次 CUDA 拟合，合计 9600 个逐模型更新步。按整数混淆矩阵独立复算两轮结果。实际回放了两种类别词表的 {ev["exact_native_masks"]} 张原尺寸预测掩膜，逐像素相同。在上一轮 162 张图上，两种骨干、六种基线的混淆矩阵也完全复现。包内可运行 <code>python -m boundary_refinement.evaluation.semantic.anchor_counts --root manuscript-extension --audit-views</code> 复算第二阶段，或导入 <code>boundary_refinement.inference.apply_path.predict</code> 将三个已训练权重应用于自己的冻结骨干得分。</p>'''
    (ROOT/"index.html").write_text(markup+"</html>",encoding="utf-8")

if __name__=="__main__":report()
