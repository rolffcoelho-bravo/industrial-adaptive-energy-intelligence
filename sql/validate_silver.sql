SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT effective_timestamp) AS unique_effective_timestamps,
    MIN(effective_timestamp) AS sample_start,
    MAX(effective_timestamp) AS sample_end,
    SUM(CASE WHEN dq_any THEN 1 ELSE 0 END) AS failed_quality_rows,
    SUM(usage_kwh) AS total_usage_kwh
FROM read_parquet('data/processed/steel_energy_silver.parquet');

SELECT
    origin_year,
    origin_month,
    load_type,
    COUNT(*) AS observations,
    AVG(usage_kwh) AS average_usage_kwh,
    MAX(usage_kwh) AS maximum_usage_kwh
FROM read_parquet('data/processed/steel_energy_silver.parquet')
GROUP BY
    origin_year,
    origin_month,
    load_type
ORDER BY
    origin_year,
    origin_month,
    load_type;
