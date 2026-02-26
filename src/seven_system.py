# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SevenConfig:
    # Gate 4: confidence / ranking
    no_trade_band: Tuple[float, float] = (0.47, 0.53)
    use_percentiles: bool = True
    long_percentile: float = 0.90      # top 10%
    short_percentile: float = 0.10     # bottom 10% (if short or avoid)

    # Gate 6: risk
    risk_per_trade: float = 0.007  # 0.7%
    stop_atr_k: float = 1.5
    max_positions: int = 7
    cooldown_bars: int = 7

    # Gate 7: exits
    take_profit_R: float = 2.0
    take_partial_R: float = 1.0
    partial_size: float = 0.4
    move_stop_to_BE_R: float = 1.0
    time_stop_bars: int = 20


class SevenGatesEngine:
    """
    Seven System:
    - gates 1-3 (placeholders الآن)
    - gate 4: ranking/selection بدل threshold 0.5
    - gate 6: position sizing ATR-based على 0.7% risk
    - gate 7: exit levels in R
    """

    def __init__(self, cfg: Optional[SevenConfig] = None):
        self.cfg = cfg or SevenConfig()

    # Gate 1-3 placeholders (يمرر OK لأن المشروع الحالي لا يحسب spread/calendar/index regime)
    def gate1_universe(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Placeholder for universe screening gate."""
        return True, "ok"

    def gate2_regime(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Placeholder for regime filter gate."""
        return True, "ok"

    def gate3_event_risk(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Placeholder for event/earnings risk gate."""
        return True, "ok"

    # Gate 4: decision via band + percentiles ranking
    def gate4_decision(self, proba_up: float, proba_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        lo, hi = self.cfg.no_trade_band
        proba_up = float(proba_up)

        if lo <= proba_up <= hi:
            return {"action": "NO_TRADE", "reason": f"proba_in_no_trade_band[{lo},{hi}]", "proba": proba_up}

        if self.cfg.use_percentiles and proba_series is not None:
            clean_series = pd.Series(proba_series).dropna()
            if len(clean_series) >= 20:
                q_hi = float(clean_series.quantile(self.cfg.long_percentile))
                q_lo = float(clean_series.quantile(self.cfg.short_percentile))

                if proba_up >= q_hi:
                    return {"action": "LONG", "reason": f"top_percentile>=q{self.cfg.long_percentile}",
                            "proba": proba_up, "q_hi": q_hi, "q_lo": q_lo}
                if proba_up <= q_lo:
                    return {"action": "AVOID", "reason": f"bottom_percentile<=q{self.cfg.short_percentile}",
                            "proba": proba_up, "q_hi": q_hi, "q_lo": q_lo}

                return {"action": "NO_TRADE", "reason": "not_in_selected_percentiles",
                        "proba": proba_up, "q_hi": q_hi, "q_lo": q_lo}

        # fallback: band break only
        return {"action": ("LONG" if proba_up > hi else "AVOID"), "reason": "band_break_fallback", "proba": proba_up}

    # Gate 6: ATR sizing
    def gate6_size(self, equity: float, price: float, atr_norm: float) -> Dict[str, Any]:
        risk_cash = float(equity) * self.cfg.risk_per_trade
        stop_distance = max(1e-9, self.cfg.stop_atr_k * float(max(atr_norm, 0)) * float(price))
        shares = int(max(0, np.floor(risk_cash / stop_distance)))
        stop_price = float(price) - stop_distance
        return {
            "risk_cash": risk_cash,
            "stop_distance": stop_distance,
            "shares": shares,
            "stop_price": stop_price,
        }

    # Gate 7: exits in R
    def gate7_exits(self, entry_price: float, stop_price: float) -> Dict[str, Any]:
        entry_price = float(entry_price)
        stop_price = float(stop_price)
        R = max(1e-9, entry_price - stop_price)

        return {
            "R": R,
            "take_profit": entry_price + self.cfg.take_profit_R * R,
            "partial_take_profit": entry_price + self.cfg.take_partial_R * R,
            "move_stop_to_BE_at": entry_price + self.cfg.move_stop_to_BE_R * R,
            "time_stop_bars": self.cfg.time_stop_bars,
            "partial_size": self.cfg.partial_size
        }
