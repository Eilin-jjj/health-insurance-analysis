-- ============================================
-- 健康保险数据分析项目 - SQL脚本
-- 执行顺序：1-23（按数字顺序执行）
-- ============================================

-- ============================================
-- 第一部分：数据库环境配置 (步骤1-2)
-- ============================================

-- 1. 开启服务器端的local_infile（用于导入CSV文件）
SET GLOBAL local_infile = 1;

-- 2. 验证local_infile是否开启（应该显示 ON）
SHOW VARIABLES LIKE 'local_infile';

-- ============================================
-- 第二部分：数据库与表结构创建 (步骤3-4)
-- ============================================

-- 3. 创建数据库
CREATE DATABASE IF NOT EXISTS insurance_analysis;
USE insurance_analysis;

-- 4. 创建客户信息表
CREATE TABLE IF NOT EXISTS customer_info (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
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

-- ============================================
-- 第三部分：数据导入与验证 (步骤5-9)
-- ============================================

-- 5. 检查表结构
DESC customer_info;

-- 6. 导入训练数据(train.csv)
LOAD DATA LOCAL INFILE 'C:/Users/Eilin/Desktop/train.csv'
INTO TABLE customer_info
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(age, sex, bmi, children, smoker, region, charges)
SET data_source = 'train';

-- 7. 查看导入警告（前5条）
SHOW WARNINGS LIMIT 5;

-- 8. 验证训练数据导入量
SELECT COUNT(*) as train_count FROM customer_info WHERE data_source = 'train';

-- 9. 查看训练数据统计信息
SELECT
    MIN(age) as min_age,
    MAX(age) as max_age,
    AVG(age) as avg_age,
    MIN(bmi) as min_bmi,
    MAX(bmi) as max_bmi,
    AVG(bmi) as avg_bmi,
    MIN(charges) as min_charges,
    MAX(charges) as max_charges,
    AVG(charges) as avg_charges
FROM customer_info WHERE data_source = 'train';

-- ============================================
-- 第四部分：测试数据导入选项 (步骤10-11)
-- ============================================

-- 10. 选项A：导入测试数据(test.csv) - 假设有charges列
LOAD DATA LOCAL INFILE 'C:/Users/yilin/Desktop/test.csv'
INTO TABLE customer_info
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(age, sex, bmi, children, smoker, region, charges)
SET data_source = 'test';

-- 11. 选项B：导入测试数据(test.csv) - 假设无charges列（只有6列）
LOAD DATA LOCAL INFILE 'C:/Users/yilin/Desktop/test.csv'
INTO TABLE customer_info
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(age, sex, bmi, children, smoker, region, @dummy)  -- @dummy忽略第7列
SET
    charges = NULL,
    data_source = 'test';

-- ============================================
-- 第五部分：性能优化 (步骤12-13)
-- ============================================

-- 12. 创建查询索引提高性能
CREATE INDEX idx_age ON customer_info(age);
CREATE INDEX idx_smoker ON customer_info(smoker);
CREATE INDEX idx_region ON customer_info(region);
CREATE INDEX idx_bmi ON customer_info(bmi);
CREATE INDEX idx_data_source ON customer_info(data_source);

-- 13. 查看所有索引信息
SHOW INDEX FROM customer_info;

-- ============================================
-- 第六部分：基础业务分析 (步骤14-18)
-- ============================================

-- 14. 基础统计分析（按数据源分组）
SELECT
    data_source,
    COUNT(*) as total_records,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi,
    AVG(charges) as avg_charges,
    COUNT(CASE WHEN smoker = 'yes' THEN 1 END) as smoker_count,
    COUNT(CASE WHEN sex = 'male' THEN 1 END) as male_count
FROM customer_info
GROUP BY data_source;

-- 15. 创建风险评分（业务逻辑）
SELECT
    customer_id,
    age,
    sex,
    bmi,
    smoker,
    region,
    charges,
    data_source,
    -- 风险评分：年龄 + BMI + 吸烟状态
    (age * 0.3 +
     CASE
        WHEN bmi < 18.5 THEN 0.1
        WHEN bmi < 25 THEN 0.2
        WHEN bmi < 30 THEN 0.4
        ELSE 0.6
     END * 100 +
     CASE WHEN smoker = 'yes' THEN 50 ELSE 10 END) as risk_score
FROM customer_info
WHERE data_source = 'train'
ORDER BY risk_score DESC
LIMIT 10;

-- 16. 客户分群分析
SELECT
    CASE
        WHEN age < 30 AND bmi < 25 AND smoker = 'no' THEN '低风险年轻健康'
        WHEN age >= 50 AND bmi >= 30 AND smoker = 'yes' THEN '高风险中老年'
        WHEN charges > 30000 THEN '高费用客户'
        WHEN charges < 5000 THEN '低费用客户'
        ELSE '中等风险'
    END as customer_segment,
    COUNT(*) as count,
    AVG(charges) as avg_charges,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi
FROM customer_info
WHERE data_source = 'train'
GROUP BY customer_segment
ORDER BY avg_charges DESC;

-- 17. 吸烟影响深度分析
SELECT
    smoker,
    sex,
    COUNT(*) as count,
    AVG(charges) as avg_charges,
    STDDEV(charges) as std_charges,
    AVG(charges) / (SELECT AVG(charges) FROM customer_info WHERE data_source = 'train') as relative_cost
FROM customer_info
WHERE data_source = 'train'
GROUP BY smoker, sex
ORDER BY avg_charges DESC;

-- 18. 地区费用对比分析
SELECT
    region,
    COUNT(*) as count,
    AVG(charges) as avg_charges,
    AVG(CASE WHEN smoker = 'yes' THEN charges END) as avg_smoker_charges,
    AVG(CASE WHEN smoker = 'no' THEN charges END) as avg_non_smoker_charges
FROM customer_info
WHERE data_source = 'train'
GROUP BY region
ORDER BY avg_charges DESC;

-- ============================================
-- 第七部分：特征工程与视图创建 (步骤19-21)
-- ============================================

-- 19. 创建训练数据视图（特征工程）
CREATE VIEW v_training_data AS
SELECT
    customer_id,
    age,
    sex,
    bmi,
    children,
    smoker,
    region,
    charges,
    -- 衍生特征：性别编码
    CASE WHEN sex = 'male' THEN 1 ELSE 0 END as is_male,
    -- 衍生特征：吸烟状态编码
    CASE WHEN smoker = 'yes' THEN 1 ELSE 0 END as is_smoker,
    -- 衍生特征：年龄分段
    CASE
        WHEN age < 30 THEN '青年'
        WHEN age < 50 THEN '中年'
        ELSE '老年'
    END as age_group,
    -- 衍生特征：BMI分类
    CASE
        WHEN bmi < 18.5 THEN '偏瘦'
        WHEN bmi < 25 THEN '正常'
        WHEN bmi < 30 THEN '超重'
        ELSE '肥胖'
    END as bmi_category
FROM customer_info
WHERE data_source = 'train';

-- 20. 使用视图：查看前5行样本数据
SELECT * FROM v_training_data LIMIT 5;

-- 21. 使用视图：统计视图数据量
SELECT COUNT(*) FROM v_training_data;

-- ============================================
-- 第八部分：数据预览与验证 (步骤22-23)
-- ============================================

-- 22. 查看训练数据前5行（原始表）
SELECT * FROM customer_info WHERE data_source = 'train' LIMIT 5;

-- 23. 查看测试数据统计（如果已导入）
SELECT 
    data_source,
    COUNT(*) as record_count,
    AVG(age) as avg_age,
    AVG(bmi) as avg_bmi
FROM customer_info 
WHERE data_source = 'test' 
GROUP BY data_source;