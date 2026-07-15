-- Decision-ready aggregations will be populated after the target and split contracts are approved.
-- Every financial scenario must expose its tariff and flexibility assumptions.

SELECT
    Day_of_week,
    Load_Type,
    AVG(Usage_kWh) AS average_usage_kwh,
    MAX(Usage_kWh) AS maximum_usage_kwh
FROM silver_steel_energy
GROUP BY Day_of_week, Load_Type
ORDER BY maximum_usage_kwh DESC;
