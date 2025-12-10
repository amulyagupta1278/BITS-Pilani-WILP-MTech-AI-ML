-- SAMPLE QUERIES (SQLite)

-- 1) Generic: average spend_per_month and churn
SELECT ROUND(AVG(spend_per_month), 2) AS avg_spm,
       ROUND(AVG(CAST(Churn AS FLOAT)), 4) AS churn_rate,
       COUNT(*) AS n
FROM generic_features;

-- 2) Generic: support burden vs churn (bucketed)
SELECT CASE
         WHEN support_calls_per_month < 0.25 THEN 'low'
         WHEN support_calls_per_month < 0.75 THEN 'med'
         ELSE 'high'
       END AS support_load,
       COUNT(*) AS n,
       ROUND(AVG(CAST(Churn AS FLOAT)), 4) AS churn_rate
FROM generic_features
GROUP BY support_load
ORDER BY n DESC;
