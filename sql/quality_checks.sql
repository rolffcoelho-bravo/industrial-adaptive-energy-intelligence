-- Databricks SQL / DuckDB adaptation target.
-- No SELECT * in production analytical queries.

SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT date) AS distinct_timestamps,
    SUM(CASE WHEN Usage_kWh IS NULL THEN 1 ELSE 0 END) AS missing_target_count,
    SUM(CASE WHEN Usage_kWh < 0 THEN 1 ELSE 0 END) AS negative_usage_count
FROM bronze_steel_energy;
