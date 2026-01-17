#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: eda_analysis.py
描述: 医疗花费预测项目 - 探索性数据分析模块
功能: 对医疗保险数据进行全面的探索性分析和可视化
作者: Eilin
创建时间: 2026年
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================
# 第一部分：环境配置与中文显示设置
# ============================================================

def setup_chinese_font():
    """
    功能：设置中文显示环境
    说明：解决matplotlib中文显示问题，设置字体和编码
    """
    # 设置字体配置，支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    print("中文显示配置完成")


# 调用字体设置函数
setup_chinese_font()
sns.set_style("whitegrid")


# ============================================================
# 第二部分：数据加载函数
# ============================================================

def load_data():
    """
    功能：加载训练和测试数据
    返回：
        train_df: 训练数据集
        test_df: 测试数据集
    """
    print("正在加载数据...")

    # 步骤1：读取训练数据CSV文件
    train_df = pd.read_csv('train_for_modeling.csv')

    # 步骤2：读取测试数据CSV文件
    test_df = pd.read_csv('test_for_modeling.csv')

    # 步骤3：输出数据基本信息
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    print(f"训练数据列名: {train_df.columns.tolist()}")

    return train_df, test_df


# ============================================================
# 第三部分：基础统计分析函数
# ============================================================

def basic_statistics(df, df_name="数据"):
    """
    功能：执行基础统计分析
    参数：
        df: 要分析的DataFrame
        df_name: 数据集名称
    返回：
        numeric_cols: 数值型列名列表
        categorical_cols: 分类型列名列表
    """
    print(f"\n{df_name}基础统计分析:")
    print("-" * 50)

    # 步骤1：定义数值型特征列
    numeric_cols = ['age', 'bmi', 'children', 'charges', 'is_male', 'is_smoker']
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    # 步骤2：输出数值型统计描述
    print("数值列描述性统计:")
    print(df[numeric_cols].describe().round(2))

    # 步骤3：定义分类型特征列
    categorical_cols = ['sex', 'smoker', 'region', 'age_group', 'bmi_category']
    categorical_cols = [col for col in categorical_cols if col in df.columns]

    # 步骤4：输出分类型分布统计
    print("\n分类列分布:")
    for col in categorical_cols:
        print(f"\n{col}分布:")
        print(df[col].value_counts())
        print(f"缺失值数量: {df[col].isnull().sum()}")

    return numeric_cols, categorical_cols


# ============================================================
# 第四部分：数据可视化分析函数
# ============================================================

def visualize_distributions(df):
    """
    功能：可视化数据分布和关系
    参数：
        df: 要可视化的DataFrame
    返回：
        correlation: 特征相关性矩阵
    """
    print("\n正在生成可视化图表...")

    # 步骤1：创建画布和子图布局
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # 步骤2：设置主标题
    fig.suptitle('医疗保险数据分析', fontsize=18, fontweight='bold', y=0.98)

    # 步骤3：医疗费用分布直方图
    axes[0, 0].hist(df['charges'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(df['charges'].mean(), color='red', linestyle='--',
                       label=f'均值: ${df["charges"].mean():.2f}')
    axes[0, 0].set_title('医疗费用分布', fontsize=14)
    axes[0, 0].set_xlabel('医疗费用 ($)', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=10)

    # 步骤4：年龄分布直方图
    axes[0, 1].hist(df['age'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(df['age'].mean(), color='red', linestyle='--',
                       label=f'均值: {df["age"].mean():.1f} 岁')
    axes[0, 1].set_title('年龄分布', fontsize=14)
    axes[0, 1].set_xlabel('年龄', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=10)

    # 步骤5：BMI分布直方图
    axes[1, 0].hist(df['bmi'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
    axes[1, 0].axvline(df['bmi'].mean(), color='red', linestyle='--',
                       label=f'均值: {df["bmi"].mean():.1f}')
    axes[1, 0].set_title('BMI分布', fontsize=14)
    axes[1, 0].set_xlabel('BMI', fontsize=12)
    axes[1, 0].set_ylabel('频数', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=10)

    # 步骤6：吸烟状况对费用的箱线图
    smoker_data = [df[df['is_smoker'] == 1]['charges'],
                   df[df['is_smoker'] == 0]['charges']]
    axes[1, 1].boxplot(smoker_data, labels=['吸烟者', '非吸烟者'])
    axes[1, 1].set_title('吸烟对医疗费用的影响', fontsize=14)
    axes[1, 1].set_ylabel('医疗费用 ($)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='both', labelsize=10)

    # 步骤7：添加吸烟者统计信息
    for i, data in enumerate(smoker_data):
        median = np.median(data)
        mean = np.mean(data)
        axes[1, 1].text(i + 1, median * 1.1, f'中位数: ${median:.0f}',
                        ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(i + 1, median * 0.9, f'均值: ${mean:.0f}',
                        ha='center', va='top', fontsize=9)

    # 步骤8：性别对费用的箱线图
    gender_data = [df[df['is_male'] == 1]['charges'],
                   df[df['is_male'] == 0]['charges']]
    axes[2, 0].boxplot(gender_data, labels=['男性', '女性'])
    axes[2, 0].set_title('性别对医疗费用的影响', fontsize=14)
    axes[2, 0].set_ylabel('医疗费用 ($)', fontsize=12)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(axis='both', labelsize=10)

    # 步骤9：特征相关性热力图
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()

    im = axes[2, 1].imshow(correlation, cmap='coolwarm', aspect='auto')
    axes[2, 1].set_title('特征相关性热力图', fontsize=14)
    axes[2, 1].set_xlabel('特征', fontsize=12)
    axes[2, 1].set_ylabel('特征', fontsize=12)
    axes[2, 1].set_xticks(range(len(correlation.columns)))
    axes[2, 1].set_xticklabels(correlation.columns, rotation=45, ha='right', fontsize=9)
    axes[2, 1].set_yticks(range(len(correlation.columns)))
    axes[2, 1].set_yticklabels(correlation.columns, fontsize=9)

    # 步骤10：添加颜色条
    cbar = plt.colorbar(im, ax=axes[2, 1])
    cbar.ax.tick_params(labelsize=9)

    # 步骤11：在热力图上添加相关系数值
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            text_color = 'white' if abs(correlation.iloc[i, j]) > 0.5 else 'black'
            axes[2, 1].text(j, i, f'{correlation.iloc[i, j]:.2f}',
                            ha='center', va='center', color=text_color, fontsize=8)

    # 步骤12：调整图表布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # 步骤13：保存图表文件
    try:
        save_dir = os.getcwd()
        save_path = os.path.join(save_dir, 'eda_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存: {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")
        try:
            plt.savefig('eda_plot.png', dpi=300)
            print("图表已保存为 eda_plot.png")
        except:
            print("图表保存失败，将继续显示图表")

    # 步骤14：显示图表
    plt.show()
    plt.close()

    return correlation


# ============================================================
# 第五部分：深度分析函数
# ============================================================

def analyze_smoking_impact(df):
    """
    功能：深入分析吸烟对医疗费用的影响
    参数：
        df: 要分析的DataFrame
    返回：
        smoker_stats: 吸烟者统计信息
    """
    print("\n吸烟影响深度分析:")
    print("-" * 50)

    # 步骤1：按吸烟状态分组统计
    smoker_stats = df.groupby('is_smoker').agg({
        'charges': ['count', 'mean', 'std', 'min', 'max'],
        'age': 'mean',
        'bmi': 'mean'
    }).round(2)

    print("吸烟者 vs 非吸烟者统计:")
    print(smoker_stats)

    # 步骤2：计算吸烟者和非吸烟者的平均费用
    smoker_mean = df[df['is_smoker'] == 1]['charges'].mean()
    non_smoker_mean = df[df['is_smoker'] == 0]['charges'].mean()
    ratio = smoker_mean / non_smoker_mean

    # 步骤3：输出关键发现
    print("\n关键发现:")
    print(f"吸烟者平均费用: ${smoker_mean:.2f}")
    print(f"非吸烟者平均费用: ${non_smoker_mean:.2f}")
    print(f"吸烟者费用是非吸烟者的 {ratio:.1f} 倍")

    return smoker_stats


# ============================================================
# 第六部分：主函数 - 程序入口
# ============================================================

def main():
    """
    功能：主函数，协调整个EDA分析流程
    """
    print("=" * 60)
    print("医疗费用数据探索性分析(EDA)")
    print("=" * 60)

    # 步骤1：加载数据
    train_df, test_df = load_data()

    # 步骤2：基础统计分析
    numeric_cols, categorical_cols = basic_statistics(train_df, "训练数据")

    # 步骤3：数据可视化分析
    correlation = visualize_distributions(train_df)

    # 步骤4：吸烟影响深度分析
    smoker_stats = analyze_smoking_impact(train_df)

    # 步骤5：输出关键业务洞察
    print("\n关键业务洞察:")
    print("-" * 50)

    # 步骤6：分析最相关的特征
    if 'charges' in correlation.columns:
        target_corr = correlation['charges'].sort_values(ascending=False)
        top_features = target_corr.index[1:4]  # 排除charges自身

        print("与医疗费用最相关的特征:")
        for feat in top_features:
            corr_value = target_corr[feat]
            direction = "正相关" if corr_value > 0 else "负相关"
            print(f"  {feat}: {direction} (r={corr_value:.3f})")

    # 步骤7：年龄分组费用分析
    if 'age_group' in train_df.columns:
        age_stats = train_df.groupby('age_group').agg({
            'charges': ['mean', 'count']
        }).round(2)
        print(f"\n年龄分组费用分析:")
        print(age_stats)

    # 步骤8：BMI分组费用分析
    if 'bmi_category' in train_df.columns:
        bmi_stats = train_df.groupby('bmi_category').agg({
            'charges': ['mean', 'count']
        }).round(2)
        print(f"\nBMI分组费用分析:")
        print(bmi_stats)

    # 步骤9：分析完成提示
    print("\n" + "=" * 60)
    print("EDA分析完成")
    print("=" * 60)


# ============================================================
# 第七部分：程序执行入口
# ============================================================

if __name__ == "__main__":
    main()