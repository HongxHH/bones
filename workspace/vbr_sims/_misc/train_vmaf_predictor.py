"""
阶段A：内容感知 VMAF 增益预测模型

目标：从内容特征精准预测每段每个 (bitrate, level) 的 VMAF 增益，
     替代当前粗糙的内容因子缩放，使 CA-BONES 接近 oracle 效果。

数据：
- vmaf_enh_seg: (159, 5, 5) 真实段级 VMAF
- bbb_features.json: 每段内容特征（纹理、运动、场景切换）
"""
import sys
sys.path.append("..")
import json
import numpy as np
from collections import defaultdict

# ============================================================
# 1. 数据加载
# ============================================================
def load_vmaf_data(enhance_path):
    """加载增强数据，返回 vmaf_enh_seg (num_seg, num_bitrate, num_level)"""
    data = json.load(open(enhance_path))
    base_q = np.array(data['base_quality_vmaf'])      # (4, 159)
    enh_q = np.array(data['enhanced_quality_vmaf'])    # (3, 4, 159)
    num_bitrate_low = base_q.shape[0]
    num_bitrate_enh = enh_q.shape[0]
    num_method = enh_q.shape[1]
    num_seg = base_q.shape[1]
    num_bitrate = num_bitrate_low + 1

    base_q_seg = base_q.T  # (159, 4)
    base_q_seg = np.concatenate([base_q_seg, np.full((num_seg, 1), 100.0)], axis=1)

    enh_q_seg = enh_q.transpose((2, 0, 1))  # (159, 3, 4)
    enh_q_seg[enh_q_seg < 0] = -np.inf
    enh_q_seg = np.concatenate([
        enh_q_seg,
        np.full((num_seg, num_bitrate - num_bitrate_enh, num_method), -np.inf)
    ], axis=1)
    vmaf_enh_seg = np.concatenate([base_q_seg[:, :, None], enh_q_seg], axis=2)
    return vmaf_enh_seg


def load_content_features(feature_path):
    """加载内容特征，返回 dict: segment_index -> feature_vector"""
    data = json.load(open(feature_path))
    features = {}
    for item in data:
        idx = item['segment_index']
        # 特征向量：纹理、运动、场景切换、纹理std、运动std
        feat = np.array([
            item['texture_complexity'],
            item['motion_intensity'],
            item['scene_change_prob'],
            item['texture_std'],
            item['motion_std'],
        ])
        features[idx] = feat
    return features


# ============================================================
# 2. 特征工程
# ============================================================
def build_feature_vector(base_feat, bitrate_idx, level_idx):
    """
    构建完整特征向量：内容特征 + bitrate/level 条件特征 + 交叉特征
    """
    feat = list(base_feat)
    # 条件特征（one-hot 或数值）
    feat.append(bitrate_idx / 4.0)   # 归一化 bitrate
    feat.append(level_idx / 4.0)      # 归一化 level
    # 交叉特征
    feat.append(base_feat[0] * base_feat[1])  # texture × motion
    feat.append(base_feat[0] ** 2)             # texture²
    feat.append(base_feat[1] ** 2)             # motion²
    feat.append(base_feat[0] * level_idx / 4.0)  # texture × level
    feat.append(base_feat[1] * level_idx / 4.0)  # motion × level
    return np.array(feat)


# ============================================================
# 3. 模型训练
# ============================================================
def train_models(X_train, y_train, X_test, y_test):
    """训练多种模型，返回预测结果和评估指标"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {
            'model': model,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'y_pred': y_pred,
        }
        print(f"  {name:20s}: RMSE={rmse:.3f}, R²={r2:.3f}, MAE={mae:.3f}")

    return results


# ============================================================
# 4. 主流程
# ============================================================
if __name__ == "__main__":
    enhance_path = "../data/bbb/imdn_bbb_3060ti.json"
    feature_path = "../player/data/content_features/bbb_features.json"

    print("=" * 60)
    print("阶段A：内容感知 VMAF 增益预测模型")
    print("=" * 60)

    # 加载数据
    vmaf_enh_seg = load_vmaf_data(enhance_path)
    features = load_content_features(feature_path)
    num_seg, num_bitrate, num_level = vmaf_enh_seg.shape
    print(f"\n数据加载完成: {num_seg} 段, {num_bitrate} 码率, {num_level} 级别")
    print(f"内容特征: {len(features)} 段, 每段 5 维特征")

    # 计算 VMAF 增益（e > 0）
    vmaf_gain = np.zeros_like(vmaf_enh_seg)
    for e in range(num_level):
        vmaf_gain[:, :, e] = vmaf_enh_seg[:, :, e] - vmaf_enh_seg[:, :, 0]

    # 构建训练数据
    print(f"\n构建训练数据...")
    X_all = []
    y_all = []
    seg_indices = []
    valid_mask = []

    for i in range(num_seg):
        if i not in features:
            continue
        for r in range(num_bitrate):
            for e in range(1, num_level):  # e=0 增益为0，跳过
                gain = vmaf_gain[i, r, e]
                if not np.isfinite(gain):
                    continue
                feat = build_feature_vector(features[i], r, e)
                X_all.append(feat)
                y_all.append(gain)
                seg_indices.append(i)
                valid_mask.append(True)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    seg_indices = np.array(seg_indices)
    print(f"训练样本数: {len(X_all)}")
    print(f"特征维度: {X_all.shape[1]}")
    print(f"VMAF 增益范围: [{y_all.min():.2f}, {y_all.max():.2f}], 均值={y_all.mean():.2f}")

    # 按段划分训练/测试集（80/20），避免数据泄漏
    unique_segs = np.unique(seg_indices)
    np.random.seed(42)
    np.random.shuffle(unique_segs)
    n_train = int(len(unique_segs) * 0.8)
    train_segs = set(unique_segs[:n_train])
    test_segs = set(unique_segs[n_train:])

    train_mask = np.array([s in train_segs for s in seg_indices])
    test_mask = np.array([s in test_segs for s in seg_indices])

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    print(f"\n训练集: {len(X_train)} 样本 ({len(train_segs)} 段)")
    print(f"测试集: {len(X_test)} 样本 ({len(test_segs)} 段)")

    # 训练模型
    print(f"\n{'='*60}")
    print("模型训练与评估（整体）")
    print(f"{'='*60}")
    results = train_models(X_train, y_train, X_test, y_test)

    # 按增强级别评估
    print(f"\n{'='*60}")
    print("按增强级别评估（最佳模型）")
    print(f"{'='*60}")
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_model = results[best_model_name]['model']
    print(f"最佳模型: {best_model_name} (R²={results[best_model_name]['r2']:.3f})")

    for e in range(1, num_level):
        level_mask = X_test[:, 6] == e / 4.0  # level 特征在第6维
        if level_mask.sum() == 0:
            continue
        y_e = y_test[level_mask]
        y_pred_e = best_model.predict(X_test[level_mask])
        rmse = np.sqrt(np.mean((y_e - y_pred_e) ** 2))
        r2 = 1 - np.sum((y_e - y_pred_e) ** 2) / np.sum((y_e - y_e.mean()) ** 2)
        print(f"  Level {e}: RMSE={rmse:.3f}, R²={r2:.3f}, 样本数={level_mask.sum()}")

    # 保存最佳模型
    import pickle
    with open("../workspace/vmaf_gain_predictor.pkl", "wb") as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_dim': X_all.shape[1],
        }, f)
    print(f"\n最佳模型已保存到 workspace/vmaf_gain_predictor.pkl")

    # 特征重要性（树模型）
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n{'='*60}")
        print("特征重要性")
        print(f"{'='*60}")
        feat_names = ['texture', 'motion', 'scene_change', 'texture_std', 'motion_std',
                      'bitrate', 'level', 'tex×mot', 'tex²', 'mot²', 'tex×level', 'mot×level']
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            print(f"  {feat_names[idx]:20s}: {importances[idx]:.4f}")
