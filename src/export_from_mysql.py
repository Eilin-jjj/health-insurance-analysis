#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: export_from_mysql.py
描述: 医疗花费预测项目 - 数据导出与特征工程模块
功能: 从MySQL数据库导出医疗保险客户数据，进行特征工程处理
作者: Eilin
创建时间: 2026年
"""

import pandas as pd
from sqlalchemy import create_engine

# ============================================================
# 第一部分：数据库连接
# ============================================================

# 步骤1：建立MySQL数据库连接
# 连接格式：mysql+mysqlconnector://用户名:密码@主机/数据库名
engine = create_engine('mysql+mysqlconnector://root:你的密码@localhost/insurance_analysis')

# ============================================================
# 第二部分：导出训练数据
# ============================================================

# 步骤2：定义训练数据查询语句（包含特征工程）
train_query = """
SELECT 
    customer_id, age, sex, bmi, children, smoker, region, charges,
    -- 特征工程：性别编码（0=女性, 1=男性）
    CASE WHEN sex = 'male' THEN 1 ELSE 0 END as is_male,
    -- 特征工程：吸烟状态编码（0=不吸烟, 1=吸烟）
    CASE WHEN smoker = 'yes' THEN 1 ELSE 0 END as is_smoker,
    -- 特征工程：年龄分组（业务逻辑）
    CASE 
        WHEN age < 30 THEN '青年'
        WHEN age < 50 THEN '中年'
        ELSE '老年'
    END as age_group,
    -- 特征工程：BMI分类（医学标准）
    CASE 
        WHEN bmi < 18.5 THEN '偏瘦'
        WHEN bmi < 25 THEN '正常'
        WHEN bmi < 30 THEN '超重'
        ELSE '肥胖'
    END as bmi_category
FROM customer_info
WHERE data_source = 'train'
"""

# 步骤3：执行训练数据查询并导出CSV
train_df = pd.read_sql(train_query, engine)
train_df.to_csv('train_for_modeling.csv', index=False, encoding='utf-8')
print(f"训练数据导出完成: {len(train_df)} 条记录")

# ============================================================
# 第三部分：导出测试数据
# ============================================================

# 步骤4：定义测试数据查询语句
test_query = """
SELECT 
    customer_id, age, sex, bmi, children, smoker, region, charges,
    CASE WHEN sex = 'male' THEN 1 ELSE 0 END as is_male,
    CASE WHEN smoker = 'yes' THEN 1 ELSE 0 END as is_smoker,
    CASE 
        WHEN age < 30 THEN '青年'
        WHEN age < 50 THEN '中年'
        ELSE '老年'
    END as age_group,
    CASE 
        WHEN bmi < 18.5 THEN '偏瘦'
        WHEN bmi < 25 THEN '正常'
        WHEN bmi < 30 THEN '超重'
        ELSE '肥胖'
    END as bmi_category
FROM customer_info
WHERE data_source = 'test'
"""

# 步骤5：执行测试数据查询并导出CSV
test_df = pd.read_sql(test_query, engine)
test_df.to_csv('test_for_modeling.csv', index=False, encoding='utf-8')
print(f"测试数据导出完成: {len(test_df)} 条记录")

# ============================================================
# 第四部分：数据预览与验证
# ============================================================

# 步骤6：查看训练数据预览
print("\n" + "="*60)
print("训练数据预览 (前5行):")
print("="*60)
print(train_df.head())

# 步骤7：查看测试数据预览
print("\n" + "="*60)
print("测试数据预览 (前5行):")
print("="*60)
print(test_df.head())

# 步骤8：输出数据形状信息
print("\n" + "="*60)
print("数据集统计信息:")
print("="*60)
print(f"训练数据形状: {train_df.shape} (行数: {train_df.shape[0]}, 列数: {train_df.shape[1]})")
print(f"测试数据形状: {test_df.shape} (行数: {test_df.shape[0]}, 列数: {test_df.shape[1]})")

# 步骤9：基本统计信息
print("\n" + "="*60)
print("训练数据基本统计:")
print("="*60)
print(f"年龄范围: {train_df['age'].min()} - {train_df['age'].max()} 岁")
print(f"平均年龄: {train_df['age'].mean():.1f} 岁")
print(f"平均BMI: {train_df['bmi'].mean():.2f}")
print(f"平均医疗花费: ${train_df['charges'].mean():,.2f}")
print(f"吸烟者比例: {train_df['is_smoker'].mean()*100:.1f}%")

# 步骤10：导出完成确认
print("\n" + "="*60)
print("数据导出完成确认:")
print("="*60)
print("已生成以下文件:")
print(f"1. train_for_modeling.csv - 训练数据集")
print(f"2. test_for_modeling.csv - 测试数据集")
print("\n数据准备完毕，可用于:")
print("• 探索性数据分析 (EDA)")
print("• 机器学习模型训练")
print("• 医疗花费预测分析")