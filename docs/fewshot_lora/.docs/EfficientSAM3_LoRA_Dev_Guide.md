# EfficientSAM3 LoRA Dev Guide

## 前置依赖说明

建议环境以仓库当前 PyTorch 代码为准，核心依赖如下：

- `torch >= 2.1`：需要 `torch.nn.MultiheadAttention`、`torch.nn.functional.scaled_dot_product_attention`、AMP 等能力。
- `torchvision`：仓库的 `sam3.model.decoder` 使用 `torchvision.ops.RoIAlign`。
- `peft >= 0.10`：可用于标准 `nn.Linear` LoRA 注入；但本仓库大量注意力使用 `nn.MultiheadAttention` 的 packed `in_proj_weight`，建议对这些层使用手动 LoRA 包装。
- `loralib`：仓库自己的 `sam3/sam3/model/student_sam/modeling/transformer.py` 已经支持 `loralib.Linear`。
- `yacs`、`iopath`、`huggingface_hub`、`mmdet`、`mmengine`：构建、配置、RPN 相关路径会导入。

本分析基于 `E:\Code\efficientsam3\sam3\sam3` 下源码，并已刷新 GitNexus 索引。当前仓库里有两条容易混淆的路径：

- `Sam3Image` 检测/分割路径：`sam3/sam3/model/sam3_image.py`，这是 `build_efficientsam3_image_model()` 返回的主模型，适合少样本检测二次开发。
- SAM-style prompt mask decoder 路径：`sam3/sam3/model/student_sam/modeling/*`，它有清晰的 `image_encoder -> prompt_encoder -> mask_decoder` 三段调用，适合快速写“框提示 + mask 重建”的 LoRA 记忆脚本。

本仓库当前 `fewshot_lora` 实现以第一条 `Sam3Image` 检测/分割路径为准，目标是原生 interactive find / grounding
闭环。本文后面保留的 SAM-style 建议只作为早期实验备选，不是当前生产实现路线。

当前代码结构也按这条主路线拆分：数据解析在 `fewshot_lora/data/`，SAM3 原生集成在
`fewshot_lora/sam3_integration/`，OBB 评估和 rotated NMS 在 `fewshot_lora/eval/`，
闭环调度和 summary 输出在 `fewshot_lora/runtime/`。其中模型构建入口为
`fewshot_lora/sam3_integration/factory.py::build_trainable_model()`；共享训练轮次 DTO
放在 `fewshot_lora/types.py`，避免 `sam3_integration` 反向依赖 `runtime`。

checkpoint 语义：训练模型在同一子数据集内跨轮连续优化 LoRA，`save_lora_adapter()`
只保存 LoRA 参数；默认评估会重新构建 base model、注入 LoRA、从 `adapter.pt`
加载权重后再推理，以验证磁盘 adapter 可恢复。

## 1. LoRA 最佳注入点

### 1.1 EfficientSAM3 主模型构建入口

文件：`sam3/sam3/model_builder.py`

主入口：

```python
def build_efficientsam3_image_model(
    bpe_path=None,
    device=None,
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=False,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
    backbone_type="efficientvit",
    model_name="b0",
    efficientvit_model=None,
    text_encoder_type=None,
    text_encoder_context_length=77,
    text_encoder_pos_embed_table_size=None,
    interpolate_pos_embed=False,
)
```

返回类型是 `Sam3Image`，定义在 `sam3/sam3/model/sam3_image.py`。

默认 `backbone_type="efficientvit"` 时，视觉主干来自 `_create_student_vision_backbone()`，随后包成：

- `ImageStudentEncoder`
- `ListWrapper`
- `Sam3DualViTDetNeck`
- `SAM3VLBackbone`
- `Sam3Image`

### 1.2 主模型视觉编码器注入点

#### SAM3 ViT 主干

如果使用 `backbone_type="sam3"` 或 `build_sam3_image_model()`，视觉主干是 `ViT`。

文件：`sam3/sam3/model/vitdet.py`

核心类：

- `class ViT`
- `class Block`
- `class Attention`

关键模块名：

- `blocks.*.attn.qkv`：`nn.Linear(dim, dim * 3)`，Q/K/V 合并投影。
- `blocks.*.attn.proj`：`nn.Linear(dim, dim)`，attention 输出投影。

`Attention.forward()` 内部 shape：

```python
def forward(self, x: Tensor) -> Tensor:
    # x: (B, H, W, C) 或 (B, L, C)
    qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    # q/k/v: (B, num_heads, L, C_per_head)
```

推荐 LoRA target name：

```python
["qkv", "proj"]
```

如果只想低风险微调，优先 `qkv`；如果记忆能力不足，再加入 `proj`。

#### EfficientViT 学生主干

默认 `build_efficientsam3_image_model(backbone_type="efficientvit", model_name="b0")` 会进入：

文件：`sam3/sam3/backbones/efficientvit/nn/ops.py`

核心类：

- `class LiteMLA`
- `class EfficientViTBlock`

关键模块名：

- `LiteMLA.qkv`：`ConvLayer(...)`，内部有 `qkv.conv`，本质是 `nn.Conv2d`。
- `LiteMLA.aggreg`：`nn.ModuleList`，内部是 depthwise/grouped `nn.Conv2d`。
- `LiteMLA.proj`：`ConvLayer(...)`，内部有 `proj.conv`。

`LiteMLA.forward()` 内部 shape：

```python
def forward(self, x):
    # x: (B, C, H, W)
    qkv = self.qkv(x)
    # qkv: (B, 3 * total_dim, H, W)
```

推荐 LoRA target name：

```python
["qkv.conv", "proj.conv"]
```

注意：`peft` 对 `Conv2d` 的支持取决于版本和配置；如果不稳定，建议手动替换 `nn.Conv2d` 为 `LoRAConv2d`。

#### TinyViT 学生主干

文件：`sam3/sam3/backbones/tiny_vit.py`

核心类：

- `class Attention`
- `class TinyViTBlock`

关键模块名：

- `attn.qkv`：`nn.Linear(dim, h)`
- `attn.proj`：`nn.Linear(self.dh, dim)`

推荐 LoRA target name：

```python
["qkv", "proj"]
```

#### RepViT 学生主干

文件：`sam3/sam3/backbones/repvit.py`

RepViT 主要是卷积块和 `SqueezeExcite`，不是 Transformer 注意力结构。关键模块包括：

- `Conv2d_BN.c`
- `RepVGGDW.conv`
- `RepVGGDW.conv1`
- `RepViTBlock.token_mixer`
- `RepViTBlock.channel_mixer`

如果目标是“Transformer 注意力 LoRA”，不建议优先选 `repvit`。如果必须用 RepViT，LoRA 更接近 Conv-LoRA，应注入 1x1 pointwise conv。

### 1.3 主模型 Transformer Encoder / Decoder 注入点

文件：`sam3/sam3/model_builder.py`

`_create_transformer_encoder()` 创建：

- `TransformerEncoderFusion`
- `TransformerEncoderLayer`
- `self_attention=MultiheadAttentionWrapper(...)`
- `cross_attention=MultiheadAttentionWrapper(...)`

`_create_transformer_decoder()` 创建：

- `TransformerDecoder`
- `TransformerDecoderLayer`
- `self_attn=nn.MultiheadAttention(...)`
- `ca_text=nn.MultiheadAttention(...)`
- `cross_attn=MultiheadAttentionWrapper(...)`

文件：`sam3/sam3/model/encoder.py`

类：

```python
class TransformerEncoderLayer(nn.Module)
```

关键属性：

- `self.self_attn`
- `self.cross_attn_image`
- `self.linear1`
- `self.linear2`

文件：`sam3/sam3/model/decoder.py`

类：

```python
class TransformerDecoderLayer(nn.Module)
```

关键属性：

- `self.self_attn`
- `self.ca_text`
- `self.cross_attn`
- `self.linear1`
- `self.linear2`

重要限制：`nn.MultiheadAttention` 的 Q/K/V 通常是 packed 参数 `in_proj_weight`，不是名为 `q_proj`、`k_proj`、`v_proj` 的子模块。因此 `peft.LoraConfig(target_modules=[...])` 不能直接精确命中它的 QKV 子投影。可选策略：

- 简单策略：只对 `out_proj`、`linear1`、`linear2` 注入 LoRA。
- 完整策略：自定义 `LoRAMultiheadAttention`，给 `in_proj_weight` 的 Q/V 分支加低秩增量。
- 折中策略：使用仓库 SAM-style `TwoWayTransformer`，它的 `Attention` 明确有 `q_proj`、`k_proj`、`v_proj`、`out_proj`。

### 1.4 SAM-style Mask Decoder 注入点

文件：`sam3/sam3/model/student_sam/modeling/transformer.py`

核心类：

```python
class TwoWayTransformer(nn.Module)
class TwoWayAttentionBlock(nn.Module)
class Attention(nn.Module)
```

`Attention.__init__()` 签名：

```python
def __init__(
    self,
    embedding_dim: int,
    num_heads: int,
    downsample_rate: int = 1,
    lora=False,
) -> None
```

关键模块名：

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`

源码已有 LoRA 开关：

```python
if lora:
    self.q_proj = loralib.Linear(embedding_dim, self.internal_dim, r=16)
    self.v_proj = loralib.Linear(embedding_dim, self.internal_dim, r=16)
else:
    self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
    self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
```

推荐 target name：

```python
["q_proj", "v_proj", "k_proj", "out_proj"]
```

更保守：

```python
["q_proj", "v_proj"]
```

该路径是最适合快速写 LoRA mask 重建脚本的底层接口。

## 2. Training Forward API

### 2.1 EfficientSAM3 主模型训练入口

文件：`sam3/sam3/model/sam3_image.py`

类：

```python
class Sam3Image(torch.nn.Module)
```

训练入口：

```python
def forward(self, input: BatchedDatapoint)
```

`BatchedDatapoint` 定义在 `sam3/sam3/model/data_misc.py`：

```python
@dataclass
class BatchedDatapoint:
    img_batch: torch.Tensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    raw_images: Optional[List[Any]] = None
```

关键 shape：

- `input.img_batch`: `(B_img, 3, H, W)`，仓库训练配置通常走 `1008 x 1008`。
- `input.find_text_batch`: `List[str]`，如 `["industrial part"]`。
- `FindStage.img_ids`: `(B_query,)`，每个 query 对应哪张图。
- `FindStage.text_ids`: `(B_query,)`，每个 query 对应哪条文本。
- `FindStage.input_boxes`: `(N_prompt_boxes, B_query, 4)`，注意 collator 对 `input_boxes` 使用 `torch.stack(..., dim=1)`。
- `FindStage.input_boxes_mask`: `(B_query, N_prompt_boxes)`，`False` 表示有效，`True` 表示 padding。
- `FindStage.input_boxes_label`: `(N_prompt_boxes, B_query)`。
- `BatchedFindTarget.boxes`: `(sum_gt, 4)`，归一化 `cx, cy, w, h`。
- `BatchedFindTarget.boxes_padded`: `(B_query, max_gt, 4)`。
- `BatchedFindTarget.segments`: `(sum_gt, H_mask, W_mask)`，bool mask。
- `BatchedFindTarget.is_valid_segment`: `(sum_gt,)`。

`Sam3Image.forward()` 内部调用链：

```python
backbone_out = {"img_batch_all_stages": input.img_batch}
backbone_out.update(self.backbone.forward_image(input.img_batch))
text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
backbone_out.update(text_outputs)

geometric_prompt = Prompt(
    box_embeddings=find_input.input_boxes,
    box_mask=find_input.input_boxes_mask,
    box_labels=find_input.input_boxes_label,
)

out = self.forward_grounding(
    backbone_out=backbone_out,
    find_input=find_input,
    find_target=find_target,
    geometric_prompt=geometric_prompt.clone(),
)
```

`forward_grounding()` 的核心链路：

```python
prompt, prompt_mask, backbone_out = self._encode_prompt(...)
backbone_out, encoder_out, _ = self._run_encoder(...)
out, hs = self._run_decoder(...)
self._run_segmentation_heads(...)
self._compute_matching(out, self.back_convert(find_target))
```

输出是 `SAM3Output`，最终每一步里的 `out` 字典通常包含：

- `pred_logits`: `(B_query, num_queries, 1)`
- `pred_boxes`: `(B_query, num_queries, 4)`，归一化 `cx, cy, w, h`
- `pred_boxes_xyxy`: `(B_query, num_queries, 4)`，归一化 `x1, y1, x2, y2`
- `pred_masks`: `(B_query, num_queries, H_mask_pred, W_mask_pred)`，启用 segmentation head 时存在。
- `indices`: matcher 结果，loss 会用它挑选正样本。

### 2.2 是否支持不给 Prompt

主模型支持空几何 prompt，但不等于“无条件重建全图 mask”。

`Sam3Image._get_dummy_prompt()` 会构造：

```python
Prompt(
    box_embeddings=torch.zeros(0, num_prompts, 4, device=device),
    box_mask=torch.zeros(num_prompts, 0, device=device, dtype=torch.bool),
)
```

这说明几何 prompt 可以为空；但主模型仍会使用文本 prompt，并通过 `forward_text()` 形成 grounding 任务。因此，如果你希望“只靠图片 + GT mask 死记硬背”，更简单的做法是给一个默认全图 box prompt。

推荐默认全图 box：

- 对 `Sam3Image` 主模型：使用归一化 `cxcywh = [0.5, 0.5, 1.0, 1.0]`。
- 对 SAM-style `PromptEncoder`：使用像素坐标 `xyxy = [0, 0, W - 1, H - 1]`，且需要映射到模型输入尺寸。

### 2.3 SAM-style 三段式底层接口

文件：`sam3/sam3/model/student_sam/modeling/sam.py`

类：

```python
class Sam(nn.Module)
```

端到端入口：

```python
@torch.no_grad()
def forward(
    self,
    batched_input: List[Dict[str, Any]],
    num_multimask_outputs: int = 1,
    use_stability_score: bool = False,
) -> List[Dict[str, torch.Tensor]]
```

注意：源码给 `forward()` 加了 `@torch.no_grad()`，不适合训练反传。做 LoRA 微调时应绕过 `Sam.forward()`，手动调用：

```python
input_images = torch.stack([model.preprocess(x["image"]) for x in batched_input], dim=0)
image_embeddings = model.image_encoder(input_images)
sparse_embeddings, dense_embeddings = model.prompt_encoder(points=..., boxes=..., masks=...)
low_res_masks, iou_predictions = model.mask_decoder(...)
```

三段 shape：

- 输入图片：每张 `image` 是 `(3, H, W)`，stack 后是 `(B, 3, H, W)`。
- `preprocess()` 会 normalize 并 pad 到 `image_encoder.img_size`，通常是 `1024`。
- `image_encoder(input_images)` 输出 `image_embeddings`: `(B, 256, 64, 64)`，因为 `1024 / 16 = 64`。
- `PromptEncoder.get_dense_pe()` 输出 `(1, 256, 64, 64)`。
- `PromptEncoder.forward(points, boxes, masks)` 输出：
  - `sparse_embeddings`: `(N_prompt, N_sparse_tokens, 256)`。
  - `dense_embeddings`: `(N_prompt, 256, 64, 64)`。
- `MaskDecoder.forward(...)` 输出：
  - `low_res_masks`: `(N_prompt, C, 256, 256)`，`C` 由 `num_multimask_outputs` 决定。
  - `iou_predictions`: `(N_prompt, C)`。

文件：`sam3/sam3/model/student_sam/modeling/prompt_encoder.py`

`PromptEncoder.forward()` 签名：

```python
def forward(
    self,
    points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    boxes: Optional[torch.Tensor],
    masks: Optional[torch.Tensor],
    box_labels=None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

prompt shape：

- `points`: `(coords, labels)`
- `coords`: `(B_prompt, N_points, 2)`，像素坐标。
- `labels`: `(B_prompt, N_points)`，`1` 正点，`0` 负点，`-1` padding。
- `boxes`: `(B_prompt, 4)`，像素坐标 `xyxy`。
- `masks`: `(B_prompt, 1, 256, 256)`，上一轮低分辨率 mask 输入。

文件：`sam3/sam3/model/student_sam/modeling/mask_decoder.py`

`MaskDecoder.forward()` 签名：

```python
def forward(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    num_multimask_outputs: int,
    num_prompts=None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]
```

`MaskDecoder.predict_masks()` 内部 shape：

```python
output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
src = src + dense_prompt_embeddings
pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

# src: (B_prompt, 256, 64, 64)
# tokens: (B_prompt, 1 + num_mask_tokens + N_sparse, 256)
hs, src = self.transformer(src, pos_src, tokens, kd_targets)
```

## 3. Loss Computation

### 3.1 EfficientSAM3 原生 loss wrapper

文件：`sam3/sam3/train/loss/sam3_loss.py`

类：

```python
class Sam3LossWrapper(torch.nn.Module)
```

构造签名：

```python
def __init__(
    self,
    loss_fns_find,
    normalization="global",
    matcher=None,
    o2m_matcher=None,
    o2m_weight=1.0,
    use_o2m_matcher_on_o2m_aux=True,
    loss_fn_semantic_seg=None,
    normalize_by_valid_object_num=False,
    normalize_by_stage_num=False,
    scale_by_find_batch_size=False,
)
```

入口：

```python
def forward(self, find_stages: SAM3Output, find_targets)
```

内部会对每个 stage 调：

```python
def compute_loss(self, nested_out, targets)
```

`compute_loss()` 期望 `nested_out` 中已经有 `indices`，由 `Sam3Image._compute_matching()` 写入。

### 3.2 Mask loss

文件：`sam3/sam3/train/loss/loss_fns.py`

函数：

```python
def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    reduce=True,
    triton=True,
)
```

```python
def dice_loss(inputs, targets, num_boxes, loss_on_multimask=False, reduce=True)
```

```python
def iou_loss(inputs, targets, pred_ious, num_boxes, loss_on_multimask=False, use_l1_loss=False)
```

Mask loss 类：

```python
class Masks(LossWithWeights)
```

构造签名：

```python
def __init__(
    self,
    weight_dict=None,
    compute_aux=False,
    focal_alpha=0.25,
    focal_gamma=2,
    num_sample_points=None,
    oversample_ratio=None,
    importance_sample_ratio=None,
    apply_loss_to_det_queries_in_video_grounding=True,
)
```

核心入口：

```python
def get_loss(self, outputs, targets, indices, num_boxes)
```

`Masks.get_loss()` 期待：

- `outputs["pred_masks"]`: `(B_query, num_queries, H_pred, W_pred)`。
- `targets["masks"]`: `(sum_gt, H_gt, W_gt)`，bool 或 float。
- `targets["is_valid_mask"]`: `(sum_gt,)`。
- `indices`: matcher 输出，源码使用 `src_masks[(indices[0], indices[1])]` 取正样本。

如果 `num_sample_points is None`，流程是：

```python
src_masks = outputs["pred_masks"]          # (B, Q, H_pred, W_pred)
src_masks = src_masks[(indices[0], indices[1])]  # (N_pos, H_pred, W_pred)
target_masks = targets["masks"][indices[2]]      # (N_pos, H_gt, W_gt)

src_masks = src_masks[:, None]
src_masks = interpolate(src_masks, size=target_masks.shape[-2:], mode="bilinear")
src_masks = src_masks[:, 0].flatten(1)     # (N_pos, H_gt * W_gt)
target_masks = target_masks.flatten(1)     # (N_pos, H_gt * W_gt)

loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_boxes)
loss_dice = dice_loss(src_masks, target_masks, num_boxes)
```

仓库配置示例在 `sam3/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml`，分割训练 loss 注释块中使用：

```yaml
- _target_: sam3.train.loss.loss_fns.Masks
  focal_alpha: 0.25
  focal_gamma: 2.0
  weight_dict:
    loss_mask: 200.0
    loss_dice: 10.0
  compute_aux: false
```

### 3.3 OBB 注意事项

EfficientSAM3 原生检测 boxes 是归一化水平框：

- `targets["boxes"]`: `cx, cy, w, h`
- `targets["boxes_xyxy"]`: `x1, y1, x2, y2`
- `pred_boxes`: `cx, cy, w, h`
- `pred_boxes_xyxy`: `x1, y1, x2, y2`

仓库没有在上述主 loss 中直接训练 OBB 五参数或八点多边形。你的 OBB 有两种落地方式：

- 先把 OBB 转成外接 AABB，喂给 `boxes` / `input_boxes`，mask 仍使用 OBB 多边形 rasterize 后的 `segments` 监督。
- 自己新增 OBB head/loss。这个会改源码符号，修改前必须按项目规则对目标符号跑 GitNexus impact analysis。

## 4. LoRA 微调脚本骨架

下面骨架优先走 SAM-style 三段式接口，因为它的 prompt/mask 训练链最短，而且 `TwoWayTransformer.Attention` 已经有明确的 `q_proj`、`v_proj`。若你要微调完整 `Sam3Image` 检测器，可参考后面的主模型输入构造。

### 4.1 推荐骨架：SAM-style `image_encoder -> prompt_encoder -> mask_decoder`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.student_sam.build_sam import build_sam_vit_b
from sam3.train.loss.loss_fns import dice_loss, sigmoid_focal_loss


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling


def inject_lora_linear_by_name(model: nn.Module, target_suffixes=("q_proj", "v_proj"), r=8, alpha=16):
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and any(full_name.endswith(s) for s in target_suffixes):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))


def freeze_non_lora(model: nn.Module):
    for name, p in model.named_parameters():
        p.requires_grad_(("lora_A" in name) or ("lora_B" in name))


def preprocess_one_sample(image_u8_chw, gt_mask_hw, box_xyxy, device):
    """
    image_u8_chw: torch.Tensor, (3, H, W), RGB, value range 0..255
    gt_mask_hw: torch.Tensor, (H, W), bool or 0/1
    box_xyxy: torch.Tensor, (4,), pixel coords in resized model input frame
    """
    image = image_u8_chw.to(device=device, dtype=torch.float32)
    gt_mask = gt_mask_hw.to(device=device, dtype=torch.float32)
    box = box_xyxy.to(device=device, dtype=torch.float32).view(1, 4)
    return image, gt_mask, box


def train_lora_memorization(samples, checkpoint_path=None, steps=80, lr=1e-3, device="cuda"):
    """
    samples: list of dicts:
      {
        "image": Tensor(3,H,W), RGB 0..255,
        "mask": Tensor(H,W), bool/float,
        "box_xyxy": Tensor(4), pixel xyxy prompt in the same image frame
      }
    """
    model = build_sam_vit_b(checkpoint=checkpoint_path, enable_distill=True, lora=False)
    model.to(device)
    model.train()

    # 推荐只在 mask decoder 的 TwoWayTransformer 注入 LoRA。
    inject_lora_linear_by_name(
        model.mask_decoder.transformer,
        target_suffixes=("q_proj", "v_proj"),
        r=8,
        alpha=16,
    )
    freeze_non_lora(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    for step in range(steps):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for sample in samples:
            image, gt_mask, box_xyxy = preprocess_one_sample(
                sample["image"], sample["mask"], sample["box_xyxy"], device
            )

            # 1. image encoder
            # image: (3,H,W), preprocess 后 pad 到 (3,1024,1024)
            input_image = model.preprocess(image).unsqueeze(0)  # (1,3,1024,1024)
            image_embeddings = model.image_encoder(input_image)  # (1,256,64,64)

            # 2. prompt encoder
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_xyxy,   # (1,4), pixel xyxy
                masks=None,
            )

            # 3. mask decoder
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),  # (1,256,64,64)
                sparse_prompt_embeddings=sparse_embeddings,    # (1,N_sparse,256)
                dense_prompt_embeddings=dense_embeddings,      # (1,256,64,64)
                num_multimask_outputs=1,
            )
            # low_res_masks: (1,1,256,256)

            pred_mask_logits = F.interpolate(
                low_res_masks,
                size=gt_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]  # (1,H,W)

            target_mask = gt_mask.unsqueeze(0)  # (1,H,W)
            pred_flat = pred_mask_logits.flatten(1)
            target_flat = target_mask.flatten(1)
            num_boxes = torch.tensor(1.0, device=device)

            loss_mask = sigmoid_focal_loss(
                pred_flat,
                target_flat,
                num_boxes,
                alpha=0.25,
                gamma=2,
                triton=False,
            )
            loss_dice = dice_loss(pred_flat, target_flat, num_boxes)
            loss = 200.0 * loss_mask + 10.0 * loss_dice
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step={step:03d} loss={float(total_loss.detach()):.6f}")

    lora_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    return model, lora_state
```

如果不想自己替换模块，也可以在构建时让仓库使用已有 `loralib`：

```python
from sam3.model.student_sam.build_sam import build_sam_vit_b

model = build_sam_vit_b(
    checkpoint="path/to/sam_checkpoint.pt",
    enable_distill=True,
    lora=True,
)
```

这会让 `sam3.model.student_sam.modeling.transformer.Attention` 的 `q_proj` 和 `v_proj` 变成 `loralib.Linear(r=16)`。

### 4.2 EfficientSAM3 主模型输入构造骨架

如果你坚持训练完整 `build_efficientsam3_image_model()`，应使用 `BatchedDatapoint`，不要直接喂 `(B,3,H,W)`。

```python
import torch

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
)


def make_one_query_batch(image, gt_box_cxcywh, gt_mask, text="industrial part"):
    """
    image: (1,3,1008,1008), float tensor
    gt_box_cxcywh: (1,4), normalized cxcywh
    gt_mask: (1,1008,1008), bool tensor
    """
    device = image.device

    find_input = FindStage(
        img_ids=torch.tensor([0], dtype=torch.long, device=device),
        text_ids=torch.tensor([0], dtype=torch.long, device=device),
        # full-image default prompt: one prompt box, one query
        input_boxes=torch.tensor([[[0.5, 0.5, 1.0, 1.0]]], dtype=torch.float32, device=device),
        input_boxes_mask=torch.zeros((1, 1), dtype=torch.bool, device=device),
        input_boxes_label=torch.ones((1, 1), dtype=torch.long, device=device),
        input_points=torch.empty((1, 0, 257), dtype=torch.float32, device=device),
        input_points_mask=torch.empty((1, 0), dtype=torch.bool, device=device),
        object_ids=[[0]],
    )

    find_target = BatchedFindTarget(
        num_boxes=torch.tensor([1], dtype=torch.long, device=device),
        boxes=gt_box_cxcywh.to(device=device, dtype=torch.float32),           # (1,4), cxcywh
        boxes_padded=gt_box_cxcywh.view(1, 1, 4).to(device=device),
        repeated_boxes=torch.empty((0, 4), dtype=torch.float32, device=device),
        segments=gt_mask.to(device=device, dtype=torch.bool),                 # (1,H,W)
        semantic_segments=None,
        is_valid_segment=torch.ones((1,), dtype=torch.bool, device=device),
        is_exhaustive=torch.ones((1,), dtype=torch.bool, device=device),
        object_ids=torch.tensor([0], dtype=torch.long, device=device),
        object_ids_padded=torch.tensor([[0]], dtype=torch.long, device=device),
    )

    metadata = BatchedInferenceMetadata(
        coco_image_id=torch.tensor([0], dtype=torch.long, device=device),
        original_image_id=torch.tensor([0], dtype=torch.long, device=device),
        original_category_id=torch.tensor([1], dtype=torch.int, device=device),
        original_size=torch.tensor([[image.shape[-2], image.shape[-1]]], dtype=torch.long, device=device),
        object_id=torch.tensor([0], dtype=torch.long, device=device),
        frame_index=torch.tensor([0], dtype=torch.long, device=device),
        is_conditioning_only=[False],
    )

    return BatchedDatapoint(
        img_batch=image,
        find_text_batch=[text],
        find_inputs=[find_input],
        find_targets=[find_target],
        find_metadatas=[metadata],
        raw_images=None,
    )


def train_full_efficientsam3_lora(samples, checkpoint_path=None, device="cuda"):
    model = build_efficientsam3_image_model(
        device=device,
        eval_mode=False,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b0",
    )
    model.train()

    # 对主模型，PEFT 可命中普通 Linear 的名字；MultiheadAttention 的 in_proj_weight 需要自定义处理。
    # 推荐先手动注入这些 Linear/Conv2d 名称：
    # - transformer.decoder.layers.*.linear1
    # - transformer.decoder.layers.*.linear2
    # - transformer.decoder.layers.*.self_attn.out_proj
    # - transformer.decoder.layers.*.ca_text.out_proj
    # - transformer.decoder.layers.*.cross_attn.out_proj
    # - segmentation_head.cross_attend_prompt.out_proj
    # - backbone.visual.trunk.model.backbone 的 qkv/proj 或 qkv.conv/proj.conv

    # TODO: 复用上一节的 LoRALinear 或实现 LoRAMultiheadAttention / LoRAConv2d。
    # freeze_non_lora(model)
    # optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # 原生 loss 需要 matcher 与 loss_fns_find。可参考 roboflow_v100_full_ft_100_images.yaml。
    # 对极少样本“死记硬背”，你也可以先不用 Sam3LossWrapper，而直接从 out 中取 pred_masks/pred_boxes 自写 loss。
    for sample in samples:
        batch = make_one_query_batch(
            image=sample["image"].to(device),
            gt_box_cxcywh=sample["box_cxcywh"].to(device),
            gt_mask=sample["mask"].to(device),
            text=sample.get("text", "industrial part"),
        )
        outputs = model(batch)
        # outputs 是 SAM3Output；内部最后一步 out 里有 pred_logits/pred_boxes/pred_masks。
        # 实际取法取决于 SAM3Output 的 iteration mode，可参考 Sam3LossWrapper.forward()。
```

## 5. 实战建议

### 5.1 最小可行路线

对你的“1 到 5 张工业零件图，迭代加入误检样本，几十步死记硬背”的目标，建议第一版不要直接改完整 `Sam3Image` 检测训练栈，而是：

1. 使用 SAM-style `build_sam_vit_b(..., enable_distill=True)` 或对应 student SAM 构造。
2. 冻结全模型。
3. 只在 `model.mask_decoder.transformer` 的 `q_proj`、`v_proj` 注入 LoRA。
4. 每张图使用 OBB rasterize 成 `gt_mask`，并把 OBB 外接框转成 `box_xyxy` prompt。
5. 用 `sigmoid_focal_loss + dice_loss` 训练 `low_res_masks` 上采样结果。

这样能最快验证“LoRA 是否能记住零件特征和 mask”。

### 5.2 什么时候切到完整 `Sam3Image`

当你需要模型输出检测框、类别分数、分割 mask 的完整 grounding 行为时，再切到 `build_efficientsam3_image_model()`：

- 检测损失：`Boxes` + `IABCEMdetr`
- 分割损失：`Masks`
- 输入结构：`BatchedDatapoint`
- 训练输出：`SAM3Output`

完整路径功能强，但输入和 matcher 复杂，几张图微调时调试成本明显高于 SAM-style 三段式路径。

### 5.3 推荐 target_modules 总结

SAM-style mask decoder：

```python
["q_proj", "v_proj"]
```

SAM3 ViT / TinyViT：

```python
["qkv", "proj"]
```

EfficientViT：

```python
["qkv.conv", "proj.conv"]
```

主模型 decoder 的低风险线性层：

```python
[
    "linear1",
    "linear2",
    "out_proj",
]
```

注意：`nn.MultiheadAttention` 的 `q/k/v` 不是独立子模块，不能只靠 `target_modules=["q_proj", "v_proj"]` 命中。
