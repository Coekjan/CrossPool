"""Runtime transport contracts shared by daemon and runtime participants."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstanceTransportAttributes(BaseModel):
    """Transport capacity and topology declared by one instance rank.

    Attributes:
        element_size: Bytes per hidden-state element produced by this instance
            rank. The native transport arena uses this to size staging buffers.
        hidden_size: Hidden-state width produced by this instance rank.
        max_tokens: Maximum token rows the rank may publish in one FFN request.
        atn_tp_rank: Attention tensor-parallel rank for this instance rank.
        atn_tp_size: Attention tensor-parallel world size for this instance
            rank.
        atn_dp_rank: Attention data-parallel rank for this instance rank.
        atn_dp_size: Attention data-parallel world size for this instance rank.
    """

    model_config = ConfigDict(extra="forbid")

    element_size: int = Field(
        description="Bytes per hidden-state element produced by this instance rank.",
    )
    hidden_size: int = Field(description="Hidden-state width produced by this instance rank.")
    max_tokens: int = Field(ge=1, description="Maximum token rows the transport arena must support.")
    atn_tp_rank: int = Field(ge=0, description="Attention tensor-parallel rank for this instance rank.")
    atn_tp_size: int = Field(ge=1, description="Attention tensor-parallel world size for this instance rank.")
    atn_dp_rank: int = Field(ge=0, description="Attention data-parallel rank for this instance rank.")
    atn_dp_size: int = Field(ge=1, description="Attention data-parallel world size for this instance rank.")

    @model_validator(mode="after")
    def validate_geometry(self) -> InstanceTransportAttributes:
        """Validate transport geometry independently of daemon placement."""

        if self.element_size not in {2, 4}:
            raise ValueError("transport element_size must be 2 or 4 bytes")
        if self.hidden_size <= 0 or self.hidden_size % 2 != 0:
            raise ValueError("transport hidden_size must be positive and even")
        if self.atn_tp_rank >= self.atn_tp_size:
            raise ValueError("transport atn_tp_rank must be smaller than atn_tp_size")
        if self.atn_dp_rank >= self.atn_dp_size:
            raise ValueError("transport atn_dp_rank must be smaller than atn_dp_size")
        return self
