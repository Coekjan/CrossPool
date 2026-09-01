"""Runtime transport contracts shared by daemon and runtime participants."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InstanceRankTransportProfile(BaseModel):
    """Transport capacity and topology declared by one Instance rank.

    Attributes:
        hidden_size: Hidden-state width produced by this instance rank.
        payload_row_capacity: Maximum physical rows the rank may publish.
        atn_tp_rank: Attention tensor-parallel rank for this instance rank.
        atn_tp_size: Attention tensor-parallel world size for this instance
            rank.
        atn_dp_rank: Attention data-parallel rank for this instance rank.
        atn_dp_size: Attention data-parallel world size for this instance rank.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(ge=1, description="Hidden-state width produced by this instance rank.")
    payload_row_capacity: int = Field(ge=1, description="Maximum physical rows the Transport Arena supports.")
    atn_tp_rank: int = Field(ge=0, description="Attention tensor-parallel rank for this instance rank.")
    atn_tp_size: int = Field(ge=1, description="Attention tensor-parallel world size for this instance rank.")
    atn_dp_rank: int = Field(ge=0, description="Attention data-parallel rank for this instance rank.")
    atn_dp_size: int = Field(ge=1, description="Attention data-parallel world size for this instance rank.")

    @model_validator(mode="after")
    def validate_geometry(self) -> InstanceRankTransportProfile:
        """Validate transport geometry independently of daemon placement."""

        if self.atn_tp_rank >= self.atn_tp_size:
            raise ValueError("transport atn_tp_rank must be smaller than atn_tp_size")
        if self.atn_dp_rank >= self.atn_dp_size:
            raise ValueError("transport atn_dp_rank must be smaller than atn_dp_size")
        if self.atn_tp_size > 1 and self.atn_dp_size > 1:
            raise ValueError("transport does not support combined attention TP-by-DP")
        return self
