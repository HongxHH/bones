"""
生成纯净VBR enhance文件：只提升base_quality_vmaf，不改变segment_sizes

基于已有的中间数据，计算VBR相对于CBR的逐段VMAF提升量，加到原始base_quality_vmaf上。
manifest保持原始CBR不变。
"""
import json
import numpy as np
from pathlib import Path


def main():
    # 配置
    original_enhance = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_2080ti.json"
    intermediate_dir = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated\intermediate")
    output_dir = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_pure")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_segments = 159

    # 码率档配置
    bitrate_indices = [0, 1, 2, 3]  # 240p, 360p, 480p, 720p

    # 加载原始enhance
    with open(original_enhance) as f:
        enhance = json.load(f)

    original_base_vmaf = np.array(enhance['base_quality_vmaf'])  # (4, 159)
    new_base_vmaf = original_base_vmaf.copy()

    print("=" * 70)
    print("生成纯净VBR enhance文件（只提升VMAF，不改文件大小）")
    print("=" * 70)
    print()

    # 加载每个分辨率的中间数据，计算逐段提升量
    for bitrate_idx in bitrate_indices:
        # 找到对应的中间文件
        intermediate_files = list(intermediate_dir.glob(f"*_data.json"))
        target_file = None
        for f in intermediate_files:
            with open(f) as fp:
                data = json.load(fp)
                if data.get('bitrate_idx') == bitrate_idx:
                    target_file = f
                    break

        if target_file is None:
            print(f"  [跳过] 码率档{bitrate_idx}：未找到中间数据")
            continue

        with open(target_file) as f:
            data = json.load(f)

        resolution = data['resolution']
        cbr_vmafs = np.array(data['cbr_vmafs'])
        vbr_vmafs = np.array(data['vbr_vmafs'])

        # 计算逐段提升量
        improvements = vbr_vmafs - cbr_vmafs

        # 加到原始base_quality_vmaf上
        for seg_idx in range(num_segments):
            if cbr_vmafs[seg_idx] > 0 and vbr_vmafs[seg_idx] > 0:
                new_base_vmaf[bitrate_idx, seg_idx] = original_base_vmaf[bitrate_idx, seg_idx] + improvements[seg_idx]

        avg_orig = np.mean(original_base_vmaf[bitrate_idx])
        avg_new = np.mean(new_base_vmaf[bitrate_idx])
        avg_improve = np.mean(improvements[improvements != 0])
        print(f"  [{resolution}] 码率档{bitrate_idx}:")
        print(f"    原始平均VMAF: {avg_orig:.2f}")
        print(f"    新平均VMAF: {avg_new:.2f}")
        print(f"    平均提升量: {avg_improve:+.2f}")
        print(f"    提升段数: {np.sum(improvements != 0)}/{num_segments}")
        print()

    # 生成新的enhance文件（enhanced_quality保持不变）
    new_enhance = {
        "frame_rate": enhance["frame_rate"],
        "enhancement_fps": enhance["enhancement_fps"],
        "base_quality_vmaf": new_base_vmaf.tolist(),
        "base_quality_ssim": enhance["base_quality_ssim"],
        "base_quality_psnr": enhance["base_quality_psnr"],
        "enhanced_quality_vmaf": enhance["enhanced_quality_vmaf"],
        "enhanced_quality_ssim": enhance["enhanced_quality_ssim"],
        "enhanced_quality_psnr": enhance["enhanced_quality_psnr"],
    }

    new_enhance_path = str(output_dir / "imdn_bbb_2080ti_pure_vbr.json")
    with open(new_enhance_path, "w") as f:
        json.dump(new_enhance, f)

    print("=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'码率档':<10} {'原始VMAF':<12} {'新VMAF':<12} {'VMAF提升':<12}")
    print("-" * 50)
    for bitrate_idx in bitrate_indices:
        resolution = ["240p", "360p", "480p", "720p"][bitrate_idx]
        orig = np.mean(original_base_vmaf[bitrate_idx])
        new = np.mean(new_base_vmaf[bitrate_idx])
        print(f"{resolution:<10} {orig:<12.2f} {new:<12.2f} {new-orig:<+12.2f}")

    print(f"\n纯净VBR enhance文件已保存到: {new_enhance_path}")
    print(f"manifest保持原始CBR不变: E:\\dev\\Code\\Python\\bones\\data\\bbb\\bbb_video_info.json")
    print(f"\n下一步: 使用原始manifest + 纯净VBR enhance文件运行BONES仿真，对比QoE")


if __name__ == "__main__":
    main()
