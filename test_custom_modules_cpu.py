import importlib.util
import py_compile
import torch

# 1. 静态语法与缩进检查：确保所有修改过的文件无任何 SyntaxError / IndentationError
modified_files = [
    "models/custom_modules.py",
    "config/OV_COCO/OVDQUO_RN50.py",
    "models/ov_dquo/ov_dquo.py",
    "models/criterion/ov_criterion_pseudo.py",
]
for f in modified_files:
    py_compile.compile(f, doraise=True)
print("✅ 4 个修改文件的 Python 语法与缩进检查全部通过！")

# 2. 绕过 models/__init__.py（避免触发 open_clip 与 CUDA 算子加载），直接导入 custom_modules
spec = importlib.util.spec_from_file_location("custom_modules", "models/custom_modules.py")
custom_modules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_modules)

DecoupledUnknownMiningModule = custom_modules.DecoupledUnknownMiningModule
AnchorLocalContextDistillModule = custom_modules.AnchorLocalContextDistillModule


def run_cpu_sanity_test():
    torch.manual_seed(42)
    bs, num_q, hidden_dim, text_dim, num_base = 2, 1000, 256, 1024, 48

    pred_logits = torch.randn(bs, num_q, 1, requires_grad=True)
    pred_boxes = torch.rand(bs, num_q, 4, requires_grad=True)
    hs_last = torch.randn(bs, num_q, hidden_dim, requires_grad=True)
    roi_features = torch.randn(bs, num_q, text_dim, requires_grad=True)
    global_img_feat = torch.randn(bs, text_dim)
    text_features = torch.randn(num_base + 1, text_dim)

    matched_mask = torch.zeros(bs, num_q, dtype=torch.bool)
    base_matched_mask = torch.zeros(bs, num_q, dtype=torch.bool)
    base_matched_mask[:, :8] = True
    matched_mask[:, :15] = True

    def mock_extract_union_roi(union_boxes_per_img):
        total = sum(b.shape[0] for b in union_boxes_per_img)
        return torch.randn(total, text_dim)

    m1 = DecoupledUnknownMiningModule(text_dim=text_dim, hidden_dim=hidden_dim, top_k_seeds=10)
    m2 = AnchorLocalContextDistillModule(hidden_dim=hidden_dim, text_dim=text_dim, num_neighbors=3)

    out1 = m1(pred_logits, pred_boxes, roi_features, text_features, global_img_feat, matched_mask)
    out2 = m2(
        hs_last,
        pred_boxes,
        out1["unknown_seed_mask"],
        base_matched_mask,
        out1["pure_bg_mask"],
        roi_features,
        mock_extract_union_roi,
    )

    print(f"✅ 挖掘未知新类种子数量: {out1['unknown_seed_mask'].sum().item()} (预期 20)")
    print(f"✅ Loss CEL (基类均匀分布去偏): {out1['loss_cel'].item():.4f}")
    print(f"✅ Loss DynBG (动态背景提示对齐): {out1['loss_dyn_bg'].item():.4f}")
    print(f"✅ Loss KD-BCE (DK-DETR 对比蒸馏): {out2['loss_kd_bce'].item():.4f}")
    print(f"✅ Loss KD-NCE (BARON 区域袋蒸馏): {out2['loss_kd_nce'].item():.4f}")

    total_loss = out1["loss_cel"] + out1["loss_dyn_bg"] + out2["loss_kd_bce"] + out2["loss_kd_nce"]
    total_loss.backward()

    assert hs_last.grad is not None and not torch.isnan(hs_last.grad).any(), "hs_last 梯度异常!"
    assert roi_features.grad is not None and not torch.isnan(roi_features.grad).any(), "roi_features 梯度异常!"
    for name, param in list(m1.named_parameters()) + list(m2.named_parameters()):
        assert param.grad is not None, f"参数 {name} 未收到梯度!"
        assert not torch.isnan(param.grad).any(), f"参数 {name} 梯度出现 NaN!"

    print("🎉 所有模块在本地 CPU 上的语法检查、维度对齐与反向传播梯度检查 100% 通过！")


if __name__ == "__main__":
    run_cpu_sanity_test()
