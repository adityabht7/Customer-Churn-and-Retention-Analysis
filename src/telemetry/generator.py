"""Physics-informed industrial sensor telemetry generator and streaming simulator."""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from src.core.constants import FLEET_ASSETS, FAILURE_MODES
from src.core.schema import TelemetryReading
from src.telemetry.signal_processing import calculate_health_index, calculate_iso_zone


class TelemetrySimulator:
    """Simulates physical sensor telemetry under nominal conditions and progressive failure modes."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.asset_map = {a["asset_id"]: a for a in FLEET_ASSETS}

    def generate_single_reading(
        self,
        asset_id: str,
        operational_stress: float = 1.0,
        injected_fault: Optional[int] = None,
        operational_hours: float = 8500.0,
        timestamp: Optional[str] = None,
    ) -> TelemetryReading:
        """Generates a physically consistent sensor reading for a specific asset."""
        if asset_id not in self.asset_map:
            raise ValueError(f"Unknown asset ID: {asset_id}")

        meta = self.asset_map[asset_id]
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        fault = injected_fault if injected_fault is not None else 0

        # Baseline equipment parameters
        rpm_nominal = float(meta["rated_rpm"])
        vibe_nominal = float(meta["nominal_vibe_rms"])
        temp_nominal = float(meta["nominal_temp_c"])
        pressure_nominal = float(meta["nominal_pressure_bar"])
        power_kw = float(meta["rated_power_kw"])

        # Base physical noise
        rpm = rpm_nominal * (1.0 + self.rng.normal(0.0, 0.005))
        vibe_de = vibe_nominal * operational_stress * (1.0 + self.rng.normal(0.0, 0.04))
        vibe_nde = vibe_de * 0.85 * (1.0 + self.rng.normal(0.0, 0.04))
        kurtosis = 3.0 + self.rng.normal(0.0, 0.15)
        crest_factor = 2.8 + self.rng.normal(0.0, 0.12)
        bearing_temp = temp_nominal * (1.0 + (operational_stress - 1.0) * 0.15) + self.rng.normal(0.0, 0.8)
        winding_temp = bearing_temp + 14.0 + self.rng.normal(0.0, 1.0)
        hyd_pressure = pressure_nominal * (1.0 + self.rng.normal(0.0, 0.02))
        diff_pressure = 2.2 + self.rng.normal(0.0, 0.15)
        acoustic_db = 48.0 + 10.0 * (operational_stress - 1.0) + self.rng.normal(0.0, 1.2)
        motor_current = (power_kw * 1000.0) / (400.0 * 1.732 * 0.88) * operational_stress + self.rng.normal(0.0, 0.8)

        # Remaining useful life default
        rul_hrs = 1500.0 / max(operational_stress, 0.5)

        # Apply specific failure mode physics
        if fault == 1:
            # Bearing Race Spalling: high-frequency impact shocks, kurtosis spike, crest factor surge
            kurtosis += self.rng.uniform(4.5, 12.0)
            crest_factor += self.rng.uniform(2.5, 6.0)
            vibe_de += self.rng.uniform(2.2, 5.5)
            vibe_nde += self.rng.uniform(1.2, 3.0)
            bearing_temp += self.rng.uniform(8.0, 22.0)
            acoustic_db += self.rng.uniform(18.0, 35.0)
            rul_hrs = float(self.rng.uniform(24.0, 140.0))

        elif fault == 2:
            # Dynamic Unbalance & Misalignment: severe 1X/2X sinusoidal vibration surge, increased motor load
            vibe_de += self.rng.uniform(4.5, 10.5)
            vibe_nde += self.rng.uniform(3.5, 8.0)
            kurtosis += self.rng.uniform(0.5, 1.5)
            motor_current *= self.rng.uniform(1.15, 1.35)
            bearing_temp += self.rng.uniform(5.0, 15.0)
            rul_hrs = float(self.rng.uniform(48.0, 220.0))

        elif fault == 3:
            # Cavitation & Impeller Erosion: erratic pressure differential, acoustic hiss, flow turbulence
            diff_pressure += self.rng.uniform(6.0, 16.0)
            hyd_pressure *= self.rng.uniform(0.65, 0.85)
            acoustic_db += self.rng.uniform(22.0, 42.0)
            vibe_de += self.rng.uniform(2.0, 4.8)
            vibe_nde += self.rng.uniform(1.5, 3.5)
            rul_hrs = float(self.rng.uniform(18.0, 96.0))

        elif fault == 4:
            # Lubrication Starvation & Thermal Runaway: boundary friction, severe temperature spike, pressure drop
            bearing_temp += self.rng.uniform(28.0, 55.0)
            winding_temp += self.rng.uniform(20.0, 45.0)
            hyd_pressure *= self.rng.uniform(0.30, 0.55)
            vibe_de += self.rng.uniform(3.0, 6.0)
            acoustic_db += self.rng.uniform(15.0, 30.0)
            rul_hrs = float(self.rng.uniform(6.0, 48.0))

        return TelemetryReading(
            timestamp=timestamp,
            asset_id=asset_id,
            asset_type=meta["asset_type"],
            rotational_speed_rpm=round(float(rpm), 1),
            vibration_de_rms=round(float(max(0.1, vibe_de)), 3),
            vibration_nde_rms=round(float(max(0.1, vibe_nde)), 3),
            vibration_kurtosis=round(float(max(1.5, kurtosis)), 2),
            vibration_crest_factor=round(float(max(1.2, crest_factor)), 2),
            bearing_temperature_c=round(float(bearing_temp), 1),
            motor_winding_temperature_c=round(float(winding_temp), 1),
            hydraulic_pressure_bar=round(float(max(0.0, hyd_pressure)), 2),
            pressure_differential_bar=round(float(max(0.0, diff_pressure)), 2),
            acoustic_emission_db=round(float(max(20.0, acoustic_db)), 1),
            motor_current_amps=round(float(max(1.0, motor_current)), 2),
            operational_hours=round(float(operational_hours), 1),
            failure_mode=fault,
            remaining_useful_life_hrs=round(float(max(0.0, rul_hrs)), 1),
        )

    def generate_fleet_dataset(
        self,
        samples_per_asset: int = 1500,
        failure_rate: float = 0.28,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generates a balanced industrial training dataset across all fleet assets with realistic wear curves."""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=60)

        rows: List[Dict[str, Any]] = []

        for asset in FLEET_ASSETS:
            asset_id = asset["asset_id"]
            current_time = start_time
            base_hours = float(self.rng.integers(2000, 18000))

            # Mix of healthy segments and degradation run-to-failure cycles
            for i in range(samples_per_asset):
                current_time += timedelta(minutes=60)
                base_hours += 1.0

                # Determine if sample is within a failure regime
                is_fault = self.rng.random() < failure_rate
                fault_mode = 0
                stress = self.rng.uniform(0.85, 1.15)

                if is_fault:
                    # Select from failure modes 1..4 with weighted probability
                    fault_mode = int(self.rng.choice([1, 2, 3, 4], p=[0.35, 0.25, 0.20, 0.20]))
                    stress = self.rng.uniform(1.05, 1.30)

                reading = self.generate_single_reading(
                    asset_id=asset_id,
                    operational_stress=stress,
                    injected_fault=fault_mode,
                    operational_hours=base_hours,
                    timestamp=current_time.isoformat(),
                )
                rows.append(reading.to_dict())

        df = pd.DataFrame(rows)
        # Calculate derived health indices and ISO zones
        df["health_index"] = df.apply(
            lambda r: calculate_health_index(
                vibration_rms=r["vibration_de_rms"],
                bearing_temp_c=r["bearing_temperature_c"],
                vibration_kurtosis=r["vibration_kurtosis"],
                pressure_diff_bar=r["pressure_differential_bar"],
                nominal_vibe=float(self.asset_map[r["asset_id"]]["nominal_vibe_rms"]),
                nominal_temp=float(self.asset_map[r["asset_id"]]["nominal_temp_c"]),
            ),
            axis=1,
        )
        df["iso_zone"] = df["vibration_de_rms"].apply(calculate_iso_zone)
        return df

    def generate_degradation_trajectory(
        self,
        asset_id: str,
        failure_mode: int,
        duration_hours: int = 168,  # 7 days lead-up
    ) -> pd.DataFrame:
        """Simulates continuous progressive degradation over time until failure threshold."""
        records: List[Dict[str, Any]] = []
        now = datetime.now() - timedelta(hours=duration_hours)
        base_hours = 9200.0

        for t in range(duration_hours):
            current_dt = now + timedelta(hours=t)
            base_hours += 1.0
            progress = t / float(duration_hours)  # 0.0 to 1.0

            # Progressive wear model: exponential acceleration towards end of life
            wear_factor = progress ** 2.2
            fault_active = failure_mode if progress > 0.35 else 0
            stress = 1.0 + (wear_factor * 0.45)

            reading = self.generate_single_reading(
                asset_id=asset_id,
                operational_stress=stress,
                injected_fault=fault_active,
                operational_hours=base_hours,
                timestamp=current_dt.isoformat(),
            )

            # Adjust RUL to reflect countdown
            remaining_hrs = max(2.0, (1.0 - progress) * duration_hours)
            d = reading.to_dict()
            d["remaining_useful_life_hrs"] = round(remaining_hrs, 1)
            d["health_index"] = round(max(5.0, 100.0 - (wear_factor * 85.0) + self.rng.normal(0, 1.5)), 1)
            d["iso_zone"] = calculate_iso_zone(d["vibration_de_rms"])
            records.append(d)

        return pd.DataFrame(records)
