"""Independent version-four observation wire values; no Native imports required."""

from enum import IntEnum, unique

__all__ = [
    "AttachmentState",
    "Delivery",
    "DpRowLayout",
    "EventKind",
    "EventRole",
    "ForwardMode",
    "GraphError",
    "GraphMode",
    "GraphPhase",
    "OutputRequirement",
    "PresenceBit",
    "ResultCode",
    "SampleFailure",
    "device_execution_boundary",
    "known_presence_mask",
]


@unique
class PresenceBit(IntEnum):
    """Bit indices, not masks, for version-four semantic field groups."""

    GENERATION = 0
    ENDPOINT = 1
    OPERATION = 2
    GRAPH_TEMPLATE = 3
    GRAPH_EXECUTION = 4
    INVOCATION = 5
    LEASE = 6
    ATTACHMENT = 7
    LOCAL_TRACE_ID = 8
    REQUEST_FACTS = 9
    INVOCATION_FACTS = 10
    DP_RANK_PAYLOAD_ROWS = 11
    PAYLOAD_ROW_CAPACITY = 12
    READY_TICKET = 13
    DELIVERY = 14
    RESULT_CODE = 15
    INSTANCE_RANK = 16
    PE = 17


@unique
class EventKind(IntEnum):
    """Version-four event kind wire values."""

    REQUEST_STAGING_STARTED = 100
    REQUEST_STAGING_COMPLETED = 101
    REQUEST_PUBLISHED = 102
    REQUEST_OBSERVED = 103
    EXECUTION_STARTED = 104
    EXECUTION_COMPLETED = 105
    RESULT_PUBLISHED = 106
    RESULT_OBSERVED = 107
    OUTPUT_COPIED = 108
    RESULT_ACKNOWLEDGED = 109
    CLOSED = 110
    SUBMISSION_PREPARED = 200
    SUBMISSION_PUBLISHED = 201
    ADMISSION_OBSERVED = 202
    INPUT_READY_PUBLISHED = 203
    OUTPUT_COMMIT_OBSERVED = 204
    OUTPUT_ACKNOWLEDGEMENT_PUBLISHED = 205
    ENQUEUED = 300
    SCHEDULED = 301
    ADMISSION_PUBLISHED = 302
    LANE_EXECUTION_PUBLISHED = 303
    FFNAGENT_COMPLETIONS_OBSERVED = 304
    OUTPUT_COMMIT_PUBLISHED = 305
    OUTPUT_ACKNOWLEDGEMENTS_OBSERVED = 306
    LANE_RELEASED = 307
    LANE_EXECUTION_OBSERVED = 400
    INPUT_READY_OBSERVED = 401
    ROUTING_METADATA_PUBLISHED = 402
    ROUTING_METADATA_OBSERVED = 403
    COMPUTE_STARTED = 404
    COMPUTE_COMPLETED = 405
    PARTIAL_READY_PUBLISHED = 406
    PEER_PARTIALS_READY_OBSERVED = 407
    COMPLETION_PUBLISHED = 408
    GRAPH_CAPTURE_BEGIN = 500
    GRAPH_CAPTURE_END = 501
    GRAPH_CAPTURE_ERROR = 502
    GRAPH_TEMPLATE_INSTALLED = 503
    GRAPH_HOST_CALL_BEGIN = 504
    GRAPH_HOST_CALL_END = 505
    GRAPH_HOST_CALL_ERROR = 506
    GRAPH_DEVICE_ACTIVITY_BEGIN = 507
    GRAPH_DEVICE_ACTIVITY_END = 508
    GRAPH_TEMPLATE_RETIRED = 509
    GRAPH_INSTALL_BEGIN = 510
    GRAPH_INSTALL_END = 511
    GRAPH_INSTALL_ERROR = 512
    TRANSPORT_FABRIC_ASSOCIATED = 450
    ROUTING_PUBLISHED = 550
    CALIBRATION_SAMPLE_SUCCEEDED = 600
    CALIBRATION_SAMPLE_FAILED = 601


@unique
class EventRole(IntEnum):
    """Version-four event role wire values."""

    INSTANCE = 1
    ATNAGENT = 2
    COORDINATOR = 3
    FFNAGENT = 4
    COLLECTOR = 5


@unique
class ForwardMode(IntEnum):
    """Version-four forward mode wire values."""

    PREFILL = 1
    DECODE = 2
    IDLE = 4


@unique
class OutputRequirement(IntEnum):
    """Version-four output requirement wire values."""

    PER_RANK_COMPLETE = 1
    GROUP_SUM_COMPLETE = 2


@unique
class DpRowLayout(IntEnum):
    """Version-four dp row layout wire values."""

    NONE = 0
    UNIFORM_BY_RANK = 1
    PACKED_BY_RANK = 2


@unique
class Delivery(IntEnum):
    """Version-four delivery wire values."""

    DIRECT_PARTIAL = 1
    SINGLE_COMPLETE = 2
    REPLICATED_COMPLETE = 3


@unique
class ResultCode(IntEnum):
    """Version-four result code wire values."""

    OK = 0
    SHUTDOWN = 1
    PROTOCOL_MISMATCH = 2
    TIMEOUT = 3


@unique
class GraphMode(IntEnum):
    """Version-four graph mode wire values."""

    REPLAY = 1
    EAGER = 2
    LANE_SERVICE = 3


@unique
class GraphPhase(IntEnum):
    """Version-four graph phase wire values."""

    DECODE = 1
    PREFILL = 2
    NATIVE_FFN = 3


@unique
class GraphError(IntEnum):
    """Version-four graph error wire values."""

    NONE = 0
    CAPTURE_FAILED = 1
    INSTALL_FAILED = 2
    HOST_CALL_FAILED = 3
    CONTEXT_UNAVAILABLE = 4
    DEVICE_BOUNDARY_UNAVAILABLE = 5


@unique
class AttachmentState(IntEnum):
    """Version-four attachment state wire values."""

    RESERVED = 1
    NOT_COLLECTED = 2
    CAPACITY_EXHAUSTED = 3
    PAYLOAD_OVERSIZE = 4


@unique
class SampleFailure(IntEnum):
    """Version-four sample failure wire values."""

    ENQUEUE_FAILED = 1
    COMPLETION_FAILED = 2
    TIMED_OUT = 3
    SHUTDOWN = 4


known_presence_mask = sum(1 << bit for bit in PresenceBit)
device_execution_boundary = 1 << 63
