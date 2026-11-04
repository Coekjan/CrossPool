"""Shared legacy Observer phase names and timestamp field pairs.

Writers and offline projections reuse these declarations. Raw evidence intervals
and legacy durations retain their respective availability and zero-time rules.
"""

transport_phases: tuple[tuple[str, str, str], ...] = (
    ("request_staging", "request_staging_started", "request_staging_completed"),
    ("request_publication", "request_staging_completed", "request_published"),
    ("request_processing", "request_observed", "result_published"),
    ("execution", "execution_started", "execution_completed"),
    ("output_copy", "result_observed", "output_copied"),
    ("result_acknowledgement", "output_copied", "result_acknowledged"),
    ("result_acknowledged_total", "request_staging_started", "result_acknowledged"),
    ("closed_total", "request_staging_started", "closed"),
)

fabric_atnagent_phases: tuple[tuple[str, str, str], ...] = (
    ("admission_wait", "submission_published", "admission_observed"),
    ("output_wait", "admission_observed", "output_commit_observed"),
    ("total", "submission_prepared", "output_acknowledgement_published"),
)

fabric_coordinator_phases: tuple[tuple[str, str, str], ...] = (
    ("scheduling", "enqueued", "scheduled"),
    ("execution", "lane_execution_published", "ffnagent_completions_observed"),
    ("active_total", "scheduled", "lane_released"),
    ("total", "enqueued", "lane_released"),
)

fabric_ffnagent_phases: tuple[tuple[str, str, str], ...] = (
    ("input_wait", "lane_execution_observed", "input_ready_observed"),
    ("compute", "compute_started", "compute_completed"),
    ("total", "lane_execution_observed", "completion_published"),
)
