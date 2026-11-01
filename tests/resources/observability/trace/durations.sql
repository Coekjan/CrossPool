-- Inspect both endpoint uncertainty and coverage, not only scalar slice dur.
SELECT ts AS display_ts,
       dur AS display_duration,
       json_extract(payload, '$.name') AS phase,
       json_extract(payload, '$.status') AS boundary_status,
       json_extract(payload, '$.duration.status') AS estimate_status,
       json_extract(payload, '$.duration.estimate_ns') AS estimate_ns,
       json_extract(payload, '$.duration.lower_ns') AS lower_ns,
       json_extract(payload, '$.duration.upper_ns') AS upper_ns,
       json_extract(payload, '$.calibration_coverage') AS calibration_coverage,
       payload
FROM xpool_evidence
WHERE name = 'xpool.interval'
ORDER BY ts, evidence_id;
