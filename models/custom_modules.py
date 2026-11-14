import torch
import torch.nn as nn
import torch.nn.functional as F
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class DecoupledUnknownMiningModule(nn.Module):
    """
    【模块一】：去偏未知物体挖掘与动态背景解耦（以解耦指导蒸馏）
    1. UnbiasedUnknownSeedMiner: 基于无偏目标性得分从背景池剥离 Top-K 未知新类种子 Queries
    2. DynamicBackgroundPrompts: 受全局图像上下文调控的可学习动态背景提示词
    3. BaseUniformDebiasLoss (CEL): 强制未知种子在 48 个 Base 类上预测均匀分布，防向 person 坍塌
    """
    def __init__(
        self,
        text_dim=1024,
        hidden_dim=256,
        top_k_seeds=10,
        num_bg_prompts=4,
        cel_tau=15.0,
        bg_tau=15.0,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.top_k_seeds = top_k_seeds
        self.num_bg_prompts = num_bg_prompts
        self.cel_tau = cel_tau
        self.bg_tau = bg_tau

        # 动态背景提示词生成器：基础可学习原型 + 图像全局上下文残差调制
        self.base_bg_prompts = nn.Parameter(torch.randn(num_bg_prompts, text_dim) * 0.02)
        self.global_ctx_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_bg_prompts * text_dim),
        )
        # 零初始化最后一层，保证训练初期平滑过渡
        nn.init.zeros_(self.global_ctx_proj[-1].weight)
        nn.init.zeros_(self.global_ctx_proj[-1].bias)

    @torch.no_grad()
    def mine_seeds_and_masks(self, pred_logits, pred_boxes, roi_features, wildcard_feat, matched_mask):
        """
        从非匹配 Query 中无偏剥离未知新类种子 (unknown_seed_mask) 与纯背景 (pure_bg_mask)
        Args:
            pred_logits: [B, Q, 1] 解码器类无关目标性 logits
            pred_boxes: [B, Q, 4] (cx, cy, w, h)
            roi_features: [B, Q, D_text] 已 L2 归一化的 CLIP RoI 特征
            wildcard_feat: [1, D_text] 通配符 "object" 文本特征
            matched_mask: [B, Q] bool, True 表示已匹配上 Base GT 或离线伪框
        Returns:
            unknown_seed_mask: [B, Q] bool, 挖掘出的高置信度未知新类种子
            pure_bg_mask: [B, Q] bool, 剩余的纯背景 Queries
        """
        bs, num_q, _ = pred_logits.shape
        device = pred_logits.device

        # 1. 纯类无关目标性得分 (脱离任何具体 Base 类别偏置)
        det_obj_score = pred_logits.squeeze(-1).sigmoid()  # [B, Q]

        # 2. 通配符 "object" 通用前景语义相似度
        roi_norm = F.normalize(roi_features, dim=-1)
        wc_norm = F.normalize(wildcard_feat, dim=-1)
        wc_sim = (roi_norm @ wc_norm.t()).squeeze(-1).clamp(min=0.0)  # [B, Q]

        # 3. 几何合理性过滤（剔除极小噪点框或全图退化框）
        wh = pred_boxes[..., 2:4]
        area = wh[..., 0] * wh[..., 1]
        geo_valid = (area > 0.002) & (area < 0.85) & (wh[..., 0] > 0.01) & (wh[..., 1] > 0.01)

        unbiased_score = (det_obj_score * (wc_sim + 0.1)) * geo_valid.float()
        # 屏蔽已匹配上 Base GT 或伪框的 Query
        unbiased_score = unbiased_score.masked_fill(matched_mask, -1.0)

        unknown_seed_mask = torch.zeros((bs, num_q), dtype=torch.bool, device=device)
        for b in range(bs):
            unmatched_count = (~matched_mask[b]).sum().item()
            k = min(self.top_k_seeds, unmatched_count)
            if k > 0:
                vals, topk_idx = torch.topk(unbiased_score[b], k=k)
                valid_topk = topk_idx[vals > 0]
                unknown_seed_mask[b, valid_topk] = True

        pure_bg_mask = (~matched_mask) & (~unknown_seed_mask)
        return unknown_seed_mask, pure_bg_mask

    def forward(
        self,
        pred_logits,
        pred_boxes,
        roi_features,
        text_features,
        global_img_feat,
        matched_mask,
    ):
        """
        Args:
            text_features: [C_base + 1, D_text]，前 C_base 个为 Base 类，最后一个为 wildcard
            global_img_feat: [B, D_text] 图像全局池化特征
        """
        bs = pred_logits.shape[0]
        base_text_feat = F.normalize(text_features[:-1], dim=-1)   # [C_base, D_text]
        wildcard_feat = text_features[-1:]                         # [1, D_text]
        num_base = base_text_feat.shape[0]

        # 1. 挖掘未知新类种子 & 纯背景掩码
        unknown_seed_mask, pure_bg_mask = self.mine_seeds_and_masks(
            pred_logits, pred_boxes, roi_features, wildcard_feat, matched_mask
        )

        roi_norm = F.normalize(roi_features, dim=-1)  # [B, Q, D_text]

        # 2. 基类均匀分布去偏损失 (CEL): 防止未知新类向高频 Base 类 (如 person) 坍塌
        if unknown_seed_mask.any():
            seed_feats = roi_norm[unknown_seed_mask]  # [N_seed, D_text]
            logits_base = self.cel_tau * (seed_feats @ base_text_feat.t())  # [N_seed, C_base]
            log_probs = F.log_softmax(logits_base, dim=-1)
            # 目标为均匀分布 1 / C_base，等价于最小化负平均对数似然
            loss_cel = -log_probs.mean()
        else:
            loss_cel = pred_logits.sum() * 0.0

        # 3. 动态背景提示词对齐损失：受全局图像上下文调控，定向吸附纯背景 Queries
        ctx_delta = self.global_ctx_proj(global_img_feat).view(bs, self.num_bg_prompts, self.text_dim)
        dyn_bg_prompts = F.normalize(self.base_bg_prompts.unsqueeze(0) + ctx_delta, dim=-1)  # [B, M_bg, D_text]

        loss_dyn_bg_list = []
        for b in range(bs):
            bg_q_feats = roi_norm[b][pure_bg_mask[b]]  # [N_bg, D_text]
            if bg_q_feats.numel() > 0:
                # 随机下采样最多 64 个纯背景框，降低计算开销
                if bg_q_feats.shape[0] > 64:
                    perm = torch.randperm(bg_q_feats.shape[0], device=bg_q_feats.device)[:64]
                    bg_q_feats = bg_q_feats[perm]
                sim_base = self.bg_tau * (bg_q_feats @ base_text_feat.t())         # [N_bg, C_base]
                sim_bg = self.bg_tau * (bg_q_feats @ dyn_bg_prompts[b].t())        # [N_bg, M_bg]
                # 合并 logits，要求纯背景 Query 的最高响应落在动态背景提示词上
                all_logits = torch.cat([sim_base, sim_bg], dim=-1)                 # [N_bg, C_base + M_bg]
                bg_probs = F.softmax(all_logits, dim=-1)[:, num_base:].sum(dim=-1)
                loss_dyn_bg_list.append(-torch.log(bg_probs.clamp(min=1e-6)).mean())

        if len(loss_dyn_bg_list) > 0:
            loss_dyn_bg = torch.stack(loss_dyn_bg_list).mean()
        else:
            loss_dyn_bg = pred_logits.sum() * 0.0

        return {
            "loss_cel": loss_cel,
            "loss_dyn_bg": loss_dyn_bg,
            "unknown_seed_mask": unknown_seed_mask,
            "pure_bg_mask": pure_bg_mask,
        }


class AnchorLocalContextDistillModule(nn.Module):
    """
    【模块二】：以未知新类为锚点的局部上下文蒸馏（以蒸馏反哺解耦）
    1. AnchorBagBuilder: 以未知新类 Query 为中心锚点，联合邻近高置信度基类 Queries 构建区域袋，并过滤纯背景噪声
    2. ContextQueryAggregator: Decoder 隐状态局部上下文注意力聚合
    3. DualContrastiveKDLoss: 融合 DK-DETR 二元交叉熵 (BCE) 与 BARON 区域袋级 InfoNCE 软对比蒸馏损失
    """
    def __init__(
        self,
        hidden_dim=256,
        text_dim=1024,
        num_neighbors=3,
        nheads=8,
        kd_tau=15.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.num_neighbors = num_neighbors
        self.kd_tau = kd_tau

        # 局部上下文交互注意力与学生端特征投影器
        self.ctx_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nheads, batch_first=True)
        self.student_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim),
        )

    @torch.no_grad()
    def build_anchor_bags(self, pred_boxes, unknown_seed_mask, base_matched_mask):
        """
        以每个 unknown_seed 为锚点，检索空间邻近的 Base GT 匹配框，构建局部区域袋并计算 Union 框
        Args:
            pred_boxes: [B, Q, 4] (cx, cy, w, h) 归一化坐标
            unknown_seed_mask: [B, Q] bool
            base_matched_mask: [B, Q] bool (仅包含高置信度 Base 匹配框，天然排除纯背景)
        Returns:
            bag_batch_indices: list of batch_idx
            bag_query_indices: list of LongTensor [1 + num_neighbors]
            union_boxes_per_img: list of Tensor [N_bag_b, 4] (cx, cy, w, h)
        """
        bs = pred_boxes.shape[0]
        device = pred_boxes.device
        boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)  # [B, Q, 4]

        bag_batch_indices = []
        bag_query_indices = []
        union_boxes_per_img = []

        for b in range(bs):
            seed_idx = unknown_seed_mask[b].nonzero(as_tuple=True)[0]
            base_idx = base_matched_mask[b].nonzero(as_tuple=True)[0]

            if seed_idx.numel() == 0:
                union_boxes_per_img.append(torch.zeros((0, 4), device=device))
                continue

            img_union_boxes = []
            for s_i in seed_idx:
                anc_box_cxcywh = pred_boxes[b, s_i]     # [4]
                anc_box_xyxy = boxes_xyxy[b, s_i]       # [4]

                if base_idx.numel() > 0:
                    ctx_boxes_cxcywh = pred_boxes[b, base_idx]  # [N_base, 4]
                    # 计算中心点欧氏距离 + 空间邻近度 (1 - GIoU)
                    center_dist = torch.norm(ctx_boxes_cxcywh[:, :2] - anc_box_cxcywh[:2].unsqueeze(0), dim=-1)
                    giou = generalized_box_iou(anc_box_xyxy.unsqueeze(0), boxes_xyxy[b, base_idx]).squeeze(0)
                    spatial_cost = center_dist - 0.5 * giou

                    m = min(self.num_neighbors, base_idx.numel())
                    _, near_pos = torch.topk(spatial_cost, k=m, largest=False)
                    chosen_ctx = base_idx[near_pos]
                    # 若不足 num_neighbors，用锚点自身补齐长度，保证张量规整
                    if m < self.num_neighbors:
                        pad = s_i.repeat(self.num_neighbors - m)
                        chosen_ctx = torch.cat([chosen_ctx, pad], dim=0)
                else:
                    chosen_ctx = s_i.repeat(self.num_neighbors)

                bag_q_idx = torch.cat([s_i.unsqueeze(0), chosen_ctx], dim=0)  # [1 + M]
                bag_boxes_xyxy = boxes_xyxy[b, bag_q_idx]  # [1 + M, 4]

                # 计算区域袋外接联合框 (Union Box) -> 转回 (cx, cy, w, h)
                u_x1 = bag_boxes_xyxy[:, 0].min().clamp(0.0, 1.0)
                u_y1 = bag_boxes_xyxy[:, 1].min().clamp(0.0, 1.0)
                u_x2 = bag_boxes_xyxy[:, 2].max().clamp(0.0, 1.0)
                u_y2 = bag_boxes_xyxy[:, 3].max().clamp(0.0, 1.0)
                u_cx = (u_x1 + u_x2) * 0.5
                u_cy = (u_y1 + u_y2) * 0.5
                u_w = (u_x2 - u_x1).clamp(min=1e-4)
                u_h = (u_y2 - u_y1).clamp(min=1e-4)
                img_union_boxes.append(torch.stack([u_cx, u_cy, u_w, u_h]))

                bag_batch_indices.append(b)
                bag_query_indices.append(bag_q_idx)

            union_boxes_per_img.append(torch.stack(img_union_boxes, dim=0))

        return bag_batch_indices, bag_query_indices, union_boxes_per_img

    def forward(
        self,
        hs_last,
        pred_boxes,
        unknown_seed_mask,
        base_matched_mask,
        pure_bg_mask,
        roi_features,
        extract_union_roi_fn,
    ):
        """
        Args:
            hs_last: [B, Q, hidden_dim] Decoder 最后一层隐状态
            extract_union_roi_fn: 回调函数，输入 list of [N_bag_b, 4] 框，输出 CLIP 教师特征 [N_total_bags, text_dim]
        """
        bag_b_idx, bag_q_idx_list, union_boxes_per_img = self.build_anchor_bags(
            pred_boxes, unknown_seed_mask, base_matched_mask
        )

        if len(bag_b_idx) == 0:
            zero_loss = hs_last.sum() * 0.0
            return {"loss_kd_bce": zero_loss, "loss_kd_nce": zero_loss}

        # 1. 学生端：聚合锚点及其邻近基类 Queries 的局部上下文表征
        bag_tokens = torch.stack(
            [hs_last[b, q_idx] for b, q_idx in zip(bag_b_idx, bag_q_idx_list)], dim=0
        )  # [N_bags, 1 + M, hidden_dim]
        attn_out, _ = self.ctx_attn(bag_tokens, bag_tokens, bag_tokens)
        # 取融合了局部邻域上下文后的中心锚点特征 (index 0)
        anchor_ctx_feat = bag_tokens[:, 0, :] + attn_out[:, 0, :]
        student_bag_feat = F.normalize(self.student_proj(anchor_ctx_feat), dim=-1)  # [N_bags, text_dim]

        # 2. 教师端：提取包含局部场景上下文的 Union Box CLIP 视觉特征 (冻结梯度)
        with torch.no_grad():
            teacher_bag_feat = extract_union_roi_fn(union_boxes_per_img)  # [N_bags, text_dim]
            teacher_bag_feat = F.normalize(teacher_bag_feat, dim=-1)

        n_bags = student_bag_feat.shape[0]

        # 3. 融合 DK-DETR 二元交叉熵 (BCE) 对比损失
        sim_matrix = self.kd_tau * (student_bag_feat @ teacher_bag_feat.t())  # [N_bags, N_bags]
        pos_targets = torch.eye(n_bags, device=sim_matrix.device, dtype=sim_matrix.dtype)
        loss_kd_bce = F.binary_cross_entropy_with_logits(sim_matrix, pos_targets)

        # 4. 融合 BARON 区域袋级 InfoNCE 关系蒸馏损失 (加入纯背景特征作为额外负样本库)
        bg_neg_feats = F.normalize(roi_features[pure_bg_mask], dim=-1)
        if bg_neg_feats.shape[0] > 32:
            perm = torch.randperm(bg_neg_feats.shape[0], device=bg_neg_feats.device)[:32]
            bg_neg_feats = bg_neg_feats[perm]

        if n_bags > 1 or bg_neg_feats.shape[0] > 0:
            key_bank = torch.cat([teacher_bag_feat, bg_neg_feats], dim=0) if bg_neg_feats.numel() > 0 else teacher_bag_feat
            logits_s2t = self.kd_tau * (student_bag_feat @ key_bank.t())  # [N_bags, N_bags + N_bg]
            nce_labels = torch.arange(n_bags, device=sim_matrix.device)
            loss_nce_s2t = F.cross_entropy(logits_s2t, nce_labels)

            if n_bags > 1:
                loss_nce_t2s = F.cross_entropy(sim_matrix.t(), nce_labels)
                loss_kd_nce = 0.5 * (loss_nce_s2t + loss_nce_t2s)
            else:
                loss_kd_nce = loss_nce_s2t
        else:
            loss_kd_nce = hs_last.sum() * 0.0

        return {
            "loss_kd_bce": loss_kd_bce,
            "loss_kd_nce": loss_kd_nce,
        }