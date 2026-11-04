-- Identity remains queryable independently of calibration availability.
SELECT category,
       json_extract(payload, '$.identity.run_id') AS run_id,
       json_extract(payload, '$.identity.components') AS components,
       payload
FROM xpool_evidence
WHERE name = 'xpool.entity'
ORDER BY category, evidence_id;
