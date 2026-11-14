import argparse
import os
import torch
from PIL import Image, ImageDraw, ImageFont
from util.slconfig import SLConfig
import util.misc as utils
from datasets import build_dataset
from main import build_model_main


TARGET_IMAGE_IDS = {
    2587: "caseA_banana_donut_2587",
    2157: "caseB_plate_knife_2157",
    1993: "caseC_bed_couch_1993",
    3156: "caseD_wall_sink_3156",
}


def load_eval_model(cfg, ckpt_path, device):
    model, _, postprocessors = build_model_main(cfg)
    model.to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("ema_model", checkpoint["model"])
    state_dict = utils.clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, postprocessors["bbox"]


def draw_boxes_on_image(pil_img, boxes, labels, scores=None, cat_names=None, color=(0, 220, 0), title=""):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.rectangle([0, 0, img.width, 26], fill=(20, 35, 65))
    draw.text((8, 4), title, fill=(255, 255, 255), font=font)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        cls_id = int(labels[idx])
        cls_name = cat_names[cls_id] if cat_names and cls_id < len(cat_names) else str(cls_id)
        tag = f"{cls_name} {scores[idx]:.2f}" if scores is not None else f"GT: {cls_name}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_y = max(28, y1 - 18)
        draw.rectangle([x1, text_y, x1 + len(tag) * 9 + 6, text_y + 18], fill=color)
        draw.text((x1 + 3, text_y + 1), tag, fill=(255, 255, 255), font=font)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize 4 Failure Cases: GT vs Baseline vs Ours")
    parser.add_argument("-c", "--config_file", default="config/OV_COCO/OVDQUO_RN50.py")
    parser.add_argument("--baseline_ckpt", default="ckpt/OVDQUO_RN50_COCO.pth")
    parser.add_argument("--ours_ckpt", required=True, help="Path to trained Full Model checkpoint.pth")
    parser.add_argument("--score_thr", type=float, default=0.25)
    parser.add_argument("--save_dir", default="vis_comparison_4cases")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = str(device)
    cfg.analysis = False

    dataset_val = build_dataset(image_set="val", args=cfg)
    cat_names = dataset_val.category_list

    baseline_model, postprocessor = load_eval_model(cfg, args.baseline_ckpt, device)
    ours_model, _ = load_eval_model(cfg, args.ours_ckpt, device)

    id_to_idx = {img_id: i for i, img_id in enumerate(dataset_val.ids)}

    for img_id, case_name in TARGET_IMAGE_IDS.items():
        if img_id not in id_to_idx:
            print(f" Image ID {img_id} not found in dataset_val")
            continue
        ds_idx = id_to_idx[img_id]
        img_tensor, target = dataset_val[ds_idx]

        # 读取原始未归一化 RGB 图像
        img_info = dataset_val.coco.loadImgs(img_id)[0]
        raw_img_path = os.path.join(dataset_val.root, img_info["file_name"])
        raw_pil = Image.open(raw_img_path).convert("RGB")
        orig_size = torch.tensor([[raw_pil.height, raw_pil.width]], device=device)

        # 1. 绘制 Ground Truth
        gt_boxes_cxcywh = target["boxes"]
        h, w = raw_pil.height, raw_pil.width
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        gt_xyxy = torch.zeros_like(gt_boxes_cxcywh)
        gt_xyxy[:, 0] = (gt_boxes_cxcywh[:, 0] - 0.5 * gt_boxes_cxcywh[:, 2]) * w
        gt_xyxy[:, 1] = (gt_boxes_cxcywh[:, 1] - 0.5 * gt_boxes_cxcywh[:, 3]) * h
        gt_xyxy[:, 2] = (gt_boxes_cxcywh[:, 0] + 0.5 * gt_boxes_cxcywh[:, 2]) * w
        gt_xyxy[:, 3] = (gt_boxes_cxcywh[:, 1] + 0.5 * gt_boxes_cxcywh[:, 3]) * h
        img_gt = draw_boxes_on_image(raw_pil, gt_xyxy, target["labels"], cat_names=cat_names, color=(39, 174, 96), title="Ground Truth")

        # 2. Baseline 推理
        samples = utils.nested_tensor_from_tensor_list([img_tensor.to(device)])
        out_base = baseline_model(samples, cat_names)
        res_base = postprocessor(out_base, orig_size)[0]
        keep_b = res_base["scores"] >= args.score_thr
        img_base = draw_boxes_on_image(
            raw_pil, res_base["boxes"][keep_b], res_base["labels"][keep_b], res_base["scores"][keep_b],
            cat_names=cat_names, color=(214, 48, 49), title="OV-DQUO Baseline (AAAI 2025)"
        )

        # 3. Ours 推理
        out_ours = ours_model(samples, cat_names)
        res_ours = postprocessor(out_ours, orig_size)[0]
        keep_o = res_ours["scores"] >= args.score_thr
        img_ours = draw_boxes_on_image(
            raw_pil, res_ours["boxes"][keep_o], res_ours["labels"][keep_o], res_ours["scores"][keep_o],
            cat_names=cat_names, color=(31, 110, 165), title="Ours (Decoupled + Local Context KD)"
        )

        # 横向拼接三联对比图
        combo = Image.new("RGB", (w * 3, h))
        combo.paste(img_gt, (0, 0))
        combo.paste(img_base, (w, 0))
        combo.paste(img_ours, (w * 2, 0))
        save_path = os.path.join(args.save_dir, f"{case_name}.jpg")
        combo.save(save_path, quality=95)
        print(f" 已保存三联对比图: {save_path}")


if __name__ == "__main__":
    main()
