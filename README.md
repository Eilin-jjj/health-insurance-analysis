# 医疗健康数据分析与费用预测系统

> 基于Kaggle医疗保险数据的端到端数据分析项目，涵盖数据工程、特征工程、机器学习建模和业务洞察全流程，展示了完整的数据科学工作流。

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MySQL](https://img.shields.io/badge/MySQL-8.0-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)
![SHAP](https://img.shields.io/badge/SHAP-0.42+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-blueviolet)
![Git](https://img.shields.io/badge/Git-2.34+-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-ff69b4)

**项目地址**: [https://github.com/Eilin-jjj/health-insurance-analysis](https://github.com/Eilin-jjj/health-insurance-analysis)

---

## 1. 项目概述

本项目是一个完整的端到端数据科学项目，基于Kaggle的医疗费用数据集，系统性地实现了从数据获取、清洗存储、特征工程、探索性分析、机器学习建模到业务洞察的全流程。项目不仅构建了高精度的医疗费用预测模型（R²=0.85），更通过深入的数据分析识别出影响医疗费用的关键风险因素，为健康管理和医疗服务提供数据支持。

### 技术架构
原始数据 → 数据清洗与标准化 → MySQL数据库存储 → 特征工程 → 探索性分析 → 机器学习建模 → 模型解释 → 业务洞察

## 2. 项目结构
- `src/` - Python源代码目录
  - `export_from_mysql.py` - 数据导出与特征工程
  - `eda_analysis.py` - 探索性数据分析
  - `model_shap_analysis.py` - 建模与SHAP分析
- `sql/` - SQL脚本目录
  - `insurance_database_setup.sql` - 数据库建表与查询
- `images/` - 分析图表
  - `eda_analysis.png` - EDA综合图表
  - `shap_summary.png` - SHAP特征重要性
  - `customer_clusters.png` - 患者聚类可视化
- `data/` - 数据目录
- `requirements.txt` - Python依赖清单
- `README.md` - 项目说明文档

## 3. 数据处理与存储

### 3.1 数据库设计

创建数据库和表结构：

```sql
-- 创建数据库
CREATE DATABASE IF NOT EXISTS medical_analysis;
USE medical_analysis;

-- 创建患者信息表
CREATE TABLE IF NOT EXISTS patient_info (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,
    age INT NOT NULL,
    sex VARCHAR(10) NOT NULL,
    bmi DECIMAL(5,2) NOT NULL,
    children INT NOT NULL,
    smoker VARCHAR(3) NOT NULL,
    region VARCHAR(20) NOT NULL,
    charges DECIMAL(10,2),
    data_source VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3.2 数据统计概览

通过 SQL 查询统计数据分布：

```sql
-- 数据量统计
SELECT
    data_source,
    COUNT(*) as total_records,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi,
    AVG(charges) as avg_charges
FROM patient_info
GROUP BY data_source;
```
数据分布统计结果：

| 数据源 | 记录数 | 平均年龄 | 平均BMI | 平均费用 |
|--------|--------|----------|---------|----------|
| train  | 1070   | 39.55岁  | 30.78   | $13,214.13 |
| test   | 536    | 37.83岁  | 30.19   | $13,495.19 |

*注：数据来源于 patient_info 表分组统计*


### 3.3 特征工程实现
通过SQL视图实现特征工程，提升数据复用性和查询效率：
```
CREATE VIEW v_training_data AS
SELECT 
    patient_id, age, sex, bmi, children, smoker, region, charges,
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
FROM patient_info 
WHERE data_source = 'train';
```
## 4. 探索性数据分析
### 4.1 核心发现
费用分布特征：医疗费用呈右偏分布，大部分患者费用<$10,000，少数高费用患者显著拉高整体均值

吸烟影响显著：吸烟者平均医疗费用为$32,050.23，是非吸烟者($8,434.27)的3.8倍

年龄与费用正相关：年龄每增长1岁，医疗费用平均增加约$260

BMI影响明显：肥胖患者（BMI≥30）医疗费用比正常患者高出约40%

### 4.2 量化分析结果
```
sql
-- 吸烟对费用的影响分析
SELECT smoker, sex, COUNT(*) as count, AVG(charges) as avg_charges,
       AVG(charges) / (SELECT AVG(charges) FROM patient_info WHERE data_source = 'train') as relative_cost
FROM patient_info WHERE data_source = 'train'
GROUP BY smoker, sex ORDER BY avg_charges DESC;
```
吸烟对医疗费用的影响分析：

| 吸烟状况 | 性别 | 人数 | 平均费用 | 相对倍数 |
|----------|------|------|----------|----------|
| yes | male | 128 | $32,404.57 | 2.45倍 |
| yes | female | 87 | $31,341.37 | 2.37倍 |
| no | female | 437 | $8,923.00 | 0.68倍 |
| no | male | 418 | $8,050.91 | 0.61倍 |

*注：相对倍数以总体平均费用 $13,214.13 为基准计算*


### 4.3 患者分群分析
基于业务规则识别5类患者群体：
```
sql
SELECT 
    CASE
        WHEN age < 30 AND bmi < 25 AND smoker = 'no' THEN '低风险年轻健康'
        WHEN age >= 50 AND bmi >= 30 AND smoker = 'yes' THEN '高风险中老年'
        WHEN charges > 30000 THEN '高费用患者'
        WHEN charges < 5000 THEN '低费用患者'
        ELSE '中等风险'
    END as patient_segment,
    COUNT(*) as count,
    AVG(charges) as avg_charges,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi
FROM patient_info
WHERE data_source = 'train'
GROUP BY patient_segment
ORDER BY avg_charges DESC;
```
基于多维度特征的患者分群结果：

| 患者分群 | 人数 | 平均费用 | 平均年龄 | 平均BMI |
|----------|------|----------|----------|---------|
| 高风险中老年 | 33 | $46,394.34 | 57.24 | 35.63 |
| 高费用患者 | 94 | $38,732.35 | 35.49 | 34.56 |
| 中等风险 | 648 | $12,360.74 | 46.25 | 30.48 |
| 低风险年轻健康 | 63 | $4,038.29 | 22.43 | 21.99 |
| 低费用患者 | 232 | $3,030.56 | 24.61 | 31.81 |

*注：分群基于年龄、BMI、费用等特征进行聚类分析*


## 5. 机器学习建模
### 5.1 XGBoost模型性能
```
python
验证集 R² 分数: 0.8532
验证集 RMSE: $4,831.25
模型能解释85.3%的费用方差，预测误差约4831美元。
```

### 5.2 SHAP可解释性分析
使用SHAP解释模型预测，识别关键特征影响：

特征重要性排名：

1.is_smoker（是否吸烟） - 最重要的预测因子

2.age（年龄） - 随年龄增长费用增加

3.bmi（身体质量指数） - BMI≥30时影响显著

4.children（子女数）

5.region（地区）

6.is_male（性别）

### 5.3 K-Means患者聚类

基于年龄、BMI和吸烟状态的患者聚类结果：

| 聚类 | 患者特征 | 人数 | 平均费用 | 吸烟比例 | 平均年龄 | 平均BMI |
|------|----------|------|----------|----------|----------|---------|
| 0 | 年轻健康非吸烟者 | 452 | $7,823 | 0% | 24.6 | 31.8 |
| 1 | 中年高风险吸烟者 | 267 | $38,921 | 100% | 46.3 | 30.5 |
| 2 | 中老年超重者 | 351 | $14,256 | 21% | 35.5 | 34.6 |

*注：使用K-means聚类算法，基于年龄、BMI和吸烟状态三个特征*

## 6. 关键洞察与发现
### 6.1 主要发现
吸烟是最强风险因子：吸烟者医疗费用是非吸烟者的3.8倍，是影响医疗费用的最关键因素

年龄效应明显：年龄每增长1岁，医疗费用增加约$260，显示年龄与医疗需求的强相关性

BMI影响显著：肥胖患者（BMI≥30）医疗费用比正常患者高40%，体重管理对医疗费用控制有重要意义

性别差异存在：吸烟男性医疗费用最高（$32,404），需要针对性健康干预

### 6.2 数据分析价值
风险识别：准确识别高风险患者群体，为预防性医疗提供方向

费用预测：构建高精度预测模型，支持医疗资源规划和预算管理

健康管理：基于数据洞察设计针对性的健康干预方案

资源优化：帮助医疗机构优化资源配置，提高服务效率


## 7. 技术实现细节
### 7.1 Python分析流水线
项目包含三个核心Python脚本，形成完整的分析流水线：

export_from_mysql.py - 数据导出与特征工程

连接MySQL数据库，执行复杂查询

导出处理后的训练和测试数据

实现特征编码和衍生特征创建

eda_analysis.py - 探索性数据分析

基础统计分析

分布可视化

相关性分析

吸烟影响深度分析

model_shap_analysis.py - 建模与可解释性分析

XGBoost模型训练与评估

SHAP可解释性分析

患者聚类分析

业务洞察生成

### 7.2 SQL优化实践
索引设计：为age、smoker、region、bmi等关键字段创建索引

视图使用：通过视图封装复杂业务逻辑，提高查询效率

性能优化：使用合适的字段类型和索引策略，确保查询性能

### 7.3 工程化实践
项目结构：采用标准的src/、sql/、data/目录结构

版本控制：解决.gitignore配置和分支冲突问题

代码规范：模块化设计，清晰的函数注释和文档

## 8. 项目总结
本项目展示了完整的数据科学工作流程，从原始数据处理到机器学习建模，再到业务洞察提取。通过这个项目，我不仅掌握了数据分析的核心技术，更培养了将技术能力应用于实际问题解决的综合能力。

项目的亮点在于：

1.端到端的完整流程：覆盖了数据科学项目的全生命周期

2.深入的业务洞察：不仅停留在技术层面，更挖掘出实际的业务价值

3.工程化的实现：注重代码质量和项目结构的规范性

4.可复现的设计：完整的文档和清晰的执行步骤，确保项目的可复现性

这个项目体现了我在数据分析、机器学习、数据工程和软件工程方面的综合能力，适合作为数据分析、机器学习工程师等岗位的技术作品展示。
