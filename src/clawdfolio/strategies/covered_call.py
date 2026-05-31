"""Risk-guided Covered Call strategy.

Integrates the Bubble Risk Score from ``analysis.bubble`` with the
Options Strategy Playbook v2.1 to generate covered call signals.

Key insight from 11-year backtesting (2014-2026, 64 parameter combos):
  - Optimal threshold: Risk Score ≥ 66 (P85 of historical distribution)
  - Best-alpha delta: 0.30 (+2.8% ann. alpha over buy-and-hold, 73.8% win rate)
  - Higher delta (0.20) raises win rate to ~85% at +2.1% alpha
  - Assignment rate ~1.5% (1 in 65 trades)
  - Signal is ASYMMETRIC: only works for sell-call, NOT for sell-put

References:
  - Market-Bubble-Index-Dashboard/analysis/cc_backtest.py
  - Market-Bubble-Index-Dashboard/analysis/cc_threshold_sweep.py
  - docs/OPTIONS_STRATEGY_PLAYBOOK_v2.1.md (Scenario A / C)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..analysis.bubble import BubbleRiskResult, fetch_bubble_risk

if TYPE_CHECKING:
    from ..core.types import Portfolio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Signal types
# ═══════════════════════════════════════════════════════════════════

class CCAction(str, Enum):
    """Covered call action types."""
    SELL = "sell_call"        # Open new CC position
    ROLL = "roll"             # Roll existing CC to next month
    HOLD = "hold"             # Keep current CC, do nothing
    PAUSE = "pause"           # Risk too low — don't sell CC
    CLOSE = "close"           # Buy back CC (risk dropping, want full upside)


@dataclass
class CoveredCallSignal:
    """A covered call trading signal."""

    ticker: str
    action: CCAction
    target_delta: float
    target_dte: int
    reason: str
    bubble_risk_score: float
    regime: str
    strength: float            # 0-1, higher = stronger conviction

    # Management parameters
    profit_target_pct: float = 0.50   # buy back at 50% profit
    stop_loss_pct: float = 2.00       # buy back at 200% loss
    roll_dte: int = 14                # roll when ≤14 DTE remaining


# ═══════════════════════════════════════════════════════════════════
# Strategy
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoveredCallStrategy:
    """Risk-guided covered call strategy.

    Uses the Bubble Risk Score to decide WHEN to sell covered calls
    and at WHAT delta.  Designed for long-term holders of leveraged
    ETFs (TQQQ) or broad-market ETFs (QQQ/SPY).

    Backtested parameters (2014-2026):
      - risk_threshold=66, delta_normal=0.30 → +2.8% alpha, 73.8% WR (best alpha)
      - Conservative: delta_normal=0.20 → +2.1% alpha, ~85% WR

    Usage::

        strategy = CoveredCallStrategy(tickers=["TQQQ"])
        signals = strategy.check_signals()
        for sig in signals:
            print(f"{sig.ticker}: {sig.action.value} δ={sig.target_delta}")
    """

    # Tickers to monitor (must already be held as shares)
    tickers: list[str] = field(default_factory=lambda: ["TQQQ"])

    # Risk threshold — sell CC only when risk score ≥ this
    risk_threshold: float = 66.0

    # Delta targets (from playbook + backtest optimisation)
    delta_normal: float = 0.25         # default CC delta (backtested optimal)
    delta_elevated: float = 0.30       # higher risk → more premium / protection
    elevated_threshold: float = 75.0   # risk score that triggers elevated delta

    # Options parameters
    target_dte: int = 35               # target days to expiry
    profit_target_pct: float = 0.50    # buy back at 50% profit
    stop_loss_pct: float = 2.00        # buy back at 200% loss
    roll_dte: int = 14                 # roll when ≤14 DTE

    # Risk source
    risk_source: str = "api"           # "api" (Dashboard) or "live" (yfinance)

    def _get_risk(self) -> BubbleRiskResult:
        """Fetch current bubble risk score."""
        if self.risk_source == "api":
            return fetch_bubble_risk()
        else:
            from ..analysis.bubble import calculate_bubble_risk
            return calculate_bubble_risk()

    def check_signals(
        self,
        portfolio: Portfolio | None = None,
    ) -> list[CoveredCallSignal]:
        """Check for covered call signals.

        Args:
            portfolio: Current portfolio (optional — used for position
                       validation if provided)

        Returns:
            List of CoveredCallSignal for each monitored ticker
        """
        risk = self._get_risk()
        signals: list[CoveredCallSignal] = []

        for ticker in self.tickers:
            signal = self._evaluate_ticker(ticker, risk, portfolio)
            signals.append(signal)

        return signals

    def _evaluate_ticker(
        self,
        ticker: str,
        risk: BubbleRiskResult,
        portfolio: Portfolio | None,
    ) -> CoveredCallSignal:
        """Generate signal for a single ticker."""
        score = risk.drawdown_risk_score

        # Determine action and delta
        if score >= self.elevated_threshold:
            action = CCAction.SELL
            delta = self.delta_elevated
            strength = min((score - self.risk_threshold) / 20, 1.0)
            reason = (
                f"⚠️ Elevated risk ({score:.0f} ≥ {self.elevated_threshold:.0f}) — "
                f"sell CC at δ={delta} for stronger protection"
            )
        elif score >= self.risk_threshold:
            action = CCAction.SELL
            delta = self.delta_normal
            strength = min((score - self.risk_threshold) / 20, 1.0)
            reason = (
                f"🔶 Risk signal active ({score:.0f} ≥ {self.risk_threshold:.0f}) — "
                f"sell CC at δ={delta}"
            )
        else:
            action = CCAction.PAUSE
            delta = self.delta_normal
            strength = 0.0
            gap = self.risk_threshold - score
            reason = (
                f"✅ Risk below threshold ({score:.0f} < {self.risk_threshold:.0f}, "
                f"gap={gap:.1f}) — hold shares only, no CC"
            )

        return CoveredCallSignal(
            ticker=ticker,
            action=action,
            target_delta=delta,
            target_dte=self.target_dte,
            reason=reason,
            bubble_risk_score=score,
            regime=risk.regime,
            strength=strength,
            profit_target_pct=self.profit_target_pct,
            stop_loss_pct=self.stop_loss_pct,
            roll_dte=self.roll_dte,
        )

    def format_signals(
        self,
        signals: list[CoveredCallSignal] | None = None,
    ) -> str:
        """Format signals as a human-readable summary.

        Args:
            signals: Pre-computed signals (will fetch if None)

        Returns:
            Formatted multi-line string
        """
        if signals is None:
            signals = self.check_signals()

        lines = [
            "━━━ Covered Call Signal Dashboard ━━━",
            "",
        ]

        for sig in signals:
            lines.append(f"  {sig.ticker}")
            lines.append(f"    Risk Score: {sig.bubble_risk_score:.1f} ({sig.regime})")
            lines.append(f"    Action:     {sig.action.value}")
            lines.append(f"    {sig.reason}")

            if sig.action == CCAction.SELL:
                lines.append(f"    Target:     δ={sig.target_delta}, DTE={sig.target_dte}")
                lines.append(f"    Mgmt:       PT={sig.profit_target_pct*100:.0f}%, "
                             f"SL={sig.stop_loss_pct*100:.0f}%, "
                             f"Roll@{sig.roll_dte}DTE")
                lines.append(f"    Strength:   {sig.strength:.0%}")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════

def check_cc_signals(
    tickers: list[str] | None = None,
    risk_threshold: float = 66.0,
) -> list[CoveredCallSignal]:
    """Quick function to check covered call signals.

    Args:
        tickers: List of tickers to check (default: ["TQQQ"])
        risk_threshold: Risk score threshold for selling CC

    Returns:
        List of CoveredCallSignal
    """
    strategy = CoveredCallStrategy(
        tickers=tickers or ["TQQQ"],
        risk_threshold=risk_threshold,
    )
    return strategy.check_signals()


def get_cc_recommendation(ticker: str = "TQQQ") -> str:
    """Get a one-line covered call recommendation.

    Args:
        ticker: Ticker symbol

    Returns:
        Human-readable recommendation string
    """
    strategy = CoveredCallStrategy(tickers=[ticker])
    signals = strategy.check_signals()
    if not signals:
        return f"{ticker}: No signal"

    sig = signals[0]
    if sig.action == CCAction.SELL:
        return (f"{ticker}: SELL CC — Risk={sig.bubble_risk_score:.0f}, "
                f"δ={sig.target_delta}, DTE={sig.target_dte}")
    return (f"{ticker}: HOLD — Risk={sig.bubble_risk_score:.0f}, "
            f"below threshold ({sig.regime})")
