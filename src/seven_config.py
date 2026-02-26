# -*- coding: utf-8 -*-
from __future__ import annotations
from .seven_system import SevenConfig

DEFAULT_SEVEN = SevenConfig(
    no_trade_band=(0.47, 0.53),
    use_percentiles=True,
    long_percentile=0.90,
    short_percentile=0.10,
    risk_per_trade=0.007,
    stop_atr_k=1.5,
    max_positions=7,
    cooldown_bars=7,
    take_profit_R=2.0,
    take_partial_R=1.0,
    partial_size=0.4,
    move_stop_to_BE_R=1.0,
    time_stop_bars=20,
)
