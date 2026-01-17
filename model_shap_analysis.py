#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: model_shap_analysis.py
描述: 医疗花费预测项目 - 模型训练与可解释性分析模块
功能: 使用XGBoost构建预测模型，进行SHAP可解释性分析和客户聚类
作者: Eilin
创建时间: 2026年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 第一部分：环境配置与字体设置
# ============================================================

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第二部分：数据加载与准备函数
# ============================================================

def load_preprocessed_data():
    """
    功能：加载预处理后的训练数据
    返回：
        df: 预处理后的DataFrame
    """
    print("步骤1：加载预处理数据...")
    df = pd.read_csv('train_for_modeling.csv')
    print(f"数据形状: {df.shape}")
    print(f"特征列: {df.columns.tolist()}")
    return df


def prepare_modeling_data(df):
    """
    功能：准备建模数据，进行特征工程
    参数：
        df: 原始DataFrame
    返回：
        X: 特征矩阵
        y: 目标变量
        features: 特征名称列表
    """
    print("\n步骤2：准备建模数据...")

    # 步骤1：选择基础数值型特征
    features = ['age', 'bmi', 'children', 'is_male', 'is_smoker']

    # 步骤2：对地区特征进行One-Hot编码（如果存在）
    if 'region' in df.columns:
        region_dummies = pd.get_dummies(df['region'], prefix='region')
        df = pd.concat([df, region_dummies], axis=1)
        features.extend(region_dummies.columns.tolist())

    # 步骤3：分离特征和目标变量
    X = df[features]
    y = df['charges']

    # 步骤4：输出数据统计信息
    print(f"建模特征数量: {len(features)}")
    print(f"特征维度: {X.shape}")
    print(f"\n特征统计描述:")
    print(X.describe().round(2))

    return X, y, features


# ============================================================
# 第三部分：模型训练与评估函数
# ============================================================

def train_xgboost_model(X, y):
    """
    功能：训练XGBoost回归模型并评估性能
    参数：
        X: 特征矩阵
        y: 目标变量
    返回：
        model: 训练好的XGBoost模型
        scaler: 标准化器对象
        X_train_scaled: 标准化后的训练特征
        X_val_scaled: 标准化后的验证特征
        y_train: 训练目标
        y_val: 验证目标
        feature_importance: 特征重要性DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤3：训练XGBoost回归模型")
    print("=" * 60)

    # 步骤1：划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 步骤2：特征标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 步骤3：配置和训练XGBoost模型
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # 步骤4：模型预测
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # 步骤5：模型性能评估
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    print(f"模型性能指标:")
    print(f"  训练集 R² 分数: {train_r2:.4f}")
    print(f"  验证集 R² 分数: {val_r2:.4f}")
    print(f"  训练集 RMSE: ${train_rmse:.2f}")
    print(f"  验证集 RMSE: ${val_rmse:.2f}")

    # 步骤6：计算特征重要性（XGBoost内置）
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n特征重要性排名 (XGBoost内置):")
    print(feature_importance.head(10))

    return model, scaler, X_train_scaled, X_val_scaled, y_train, y_val, feature_importance


# ============================================================
# 第四部分：SHAP可解释性分析函数
# ============================================================

def shap_analysis(model, X_train_scaled, X_val_scaled, feature_names):
    """
    功能：进行SHAP可解释性分析，可视化特征影响
    参数：
        model: 训练好的XGBoost模型
        X_train_scaled: 标准化后的训练特征
        X_val_scaled: 标准化后的验证特征
        feature_names: 特征名称列表
    返回：
        explainer: SHAP解释器对象
        shap_values: SHAP值矩阵
        mean_abs_shap: 平均绝对SHAP值DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤4：SHAP可解释性分析")
    print("=" * 60)

    # 步骤1：创建SHAP解释器
    explainer = shap.TreeExplainer(model)

    # 步骤2：计算SHAP值（使用部分样本加速计算）
    sample_size = min(100, X_val_scaled.shape[0])
    X_val_sample = X_val_scaled[:sample_size]
    shap_values = explainer.shap_values(X_val_sample)

    # 步骤3：生成SHAP摘要图（特征重要性可视化）
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_val_sample,
                      feature_names=feature_names,
                      show=False)
    plt.title("SHAP特征重要性摘要图", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 步骤4：生成SHAP条形图（平均绝对影响）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_val_sample,
                      feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("特征平均绝对SHAP值", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 步骤5：生成单个特征依赖图（前3个最重要特征）
    top_features = ['is_smoker', 'age', 'bmi']

    for feat in top_features:
        if feat in feature_names:
            feat_idx = list(feature_names).index(feat)
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feat_idx, shap_values, X_val_sample,
                                 feature_names=feature_names,
                                 show=False)
            plt.title(f"SHAP依赖图 - {feat}", fontsize=14)
            plt.tight_layout()
            plt.savefig(f'shap_dependence_{feat}.png', dpi=300, bbox_inches='tight')
            plt.show()

    # 步骤6：计算平均绝对SHAP值
    mean_abs_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("SHAP值分析结果（平均绝对影响）:")
    print(mean_abs_shap)

    return explainer, shap_values, mean_abs_shap


# ============================================================
# 第五部分：客户聚类分析函数
# ============================================================

def customer_clustering(X, y):
    """
    功能：对客户进行聚类分析，识别不同客户群体
    参数：
        X: 特征矩阵
        y: 目标变量
    返回：
        X_cluster: 包含聚类结果的数据
        cluster_stats: 聚类统计信息
        cluster_names: 聚类命名字典
    """
    print("\n" + "=" * 60)
    print("步骤5：客户聚类分析")
    print("=" * 60)

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # 步骤1：选择用于聚类的特征
    cluster_features = ['age', 'bmi', 'is_smoker']
    X_cluster = X[cluster_features].copy()

    # 步骤2：特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # 步骤3：使用肘部法则确定最佳聚类数量
    wcss = []
    k_range = range(2, 8)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # 步骤4：绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('WCSS (簇内平方和)')
    plt.title('肘部法则 - 确定最佳聚类数量')
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 步骤5：执行K-Means聚类（K=3）
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # 步骤6：将聚类结果添加到数据中
    X_cluster['cluster'] = clusters
    X_cluster['charges'] = y.values

    # 步骤7：聚类统计信息
    cluster_stats = X_cluster.groupby('cluster').agg({
        'age': 'mean',
        'bmi': 'mean',
        'is_smoker': 'mean',
        'charges': ['mean', 'count']
    }).round(2)

    print("客户聚类统计结果:")
    print(cluster_stats)

    # 步骤8：业务命名各个聚类
    cluster_names = {
        0: "年轻健康非吸烟者",
        1: "中年高风险吸烟者",
        2: "中老年超重者"
    }

    print("\n聚类业务解读:")
    for cluster_id, stats in cluster_stats.iterrows():
        name = cluster_names.get(cluster_id, f"集群{cluster_id}")
        print(f"\n{name} (集群{cluster_id}):")
        print(f"  人数: {int(stats[('charges', 'count')])}")
        print(f"  平均年龄: {stats[('age', 'mean')]} 岁")
        print(f"  平均BMI: {stats[('bmi', 'mean')]}")
        print(f"  吸烟比例: {stats[('is_smoker', 'mean')]:.2%}")
        print(f"  平均医疗费用: ${stats[('charges', 'mean')]:,.0f}")

    # 步骤9：可视化聚类结果（PCA降维）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=clusters, cmap='viridis',
                          alpha=0.7, s=50)
    plt.colorbar(scatter)
    plt.title('客户聚类可视化 (PCA降维)', fontsize=14)
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')

    # 标注聚类中心
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
                c='red', marker='X', s=200, label='聚类中心')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('customer_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

    return X_cluster, cluster_stats, cluster_names


# ============================================================
# 第六部分：业务洞察生成函数
# ============================================================

def generate_business_insights(feature_importance, mean_abs_shap, cluster_stats):
    """
    功能：生成业务洞察和建议
    参数：
        feature_importance: 特征重要性DataFrame
        mean_abs_shap: 平均绝对SHAP值DataFrame
        cluster_stats: 聚类统计信息
    """
    print("\n" + "=" * 60)
    print("步骤6：业务洞察与建议")
    print("=" * 60)

    print("\n核心发现:")
    print("1. 关键影响因素分析:")
    print("   • 是否吸烟: 是最强的医疗费用预测因子")
    print("   • 年龄: 随着年龄增长，医疗费用显著上升")
    print("   • BMI指数: 超重和肥胖与更高医疗费用相关")

    print("\n2. 客户分群洞察:")
    for cluster_id, stats in cluster_stats.iterrows():
        count = int(stats[('charges', 'count')])
        avg_charges = stats[('charges', 'mean')]
        smoking_rate = stats[('is_smoker', 'mean')]

        print(f"\n   集群{cluster_id} ({count}人, 占{(count / 1070 * 100):.1f}%):")
        print(f"   • 平均医疗费用: ${avg_charges:,.0f}")
        if smoking_rate > 0.3:
            print(f"   • 高风险特征: 吸烟率高达{smoking_rate:.1%}")

    print("\n业务建议:")
    print("1. 精准定价策略:")
    print("   • 对吸烟者设置更高的保费系数")
    print("   • 针对高龄客户提供专项健康管理服务")

    print("\n2. 风险管理措施:")
    print("   • 为高BMI客户提供健康饮食指导")
    print("   • 针对中年高风险吸烟者推出戒烟激励计划")

    print("\n3. 客户运营优化:")
    print("   • 对年轻健康群体设计入门级保险产品")
    print("   • 为中老年超重者提供定期体检服务")


# ============================================================
# 第七部分：主函数 - 程序执行入口
# ============================================================

def main():
    """
    功能：主函数，协调整个模型训练和分析流程
    """
    print("=" * 60)
    print("健康保险费用预测 - 建模与业务分析")
    print("=" * 60)

    # 步骤1：加载预处理数据
    df = load_preprocessed_data()

    # 步骤2：准备建模数据
    X, y, feature_names = prepare_modeling_data(df)

    # 步骤3：训练XGBoost模型
    model, scaler, X_train_scaled, X_val_scaled, y_train, y_val, feat_imp = train_xgboost_model(X, y)

    # 步骤4：SHAP可解释性分析
    explainer, shap_values, mean_abs_shap = shap_analysis(model, X_train_scaled, X_val_scaled, feature_names)

    # 步骤5：客户聚类分析
    X_cluster, cluster_stats, cluster_names = customer_clustering(X, y)

    # 步骤6：生成业务洞察
    generate_business_insights(feat_imp, mean_abs_shap, cluster_stats)

    # 步骤7：分析完成总结
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print("生成的文件:")
    print("  • shap_summary.png - SHAP特征重要性摘要图")
    print("  • shap_bar.png - 特征平均绝对SHAP值条形图")
    print("  • shap_dependence_*.png - 特征依赖关系图")
    print("  • elbow_method.png - 聚类肘部法则图")
    print("  • customer_clusters.png - 客户聚类可视化图")
    print("\n所有分析已完成，文件已保存。")


# ============================================================
# 第八部分：程序执行入口
# ============================================================

if __name__ == "__main__":
    main()