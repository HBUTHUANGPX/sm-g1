"""CPU-only validation and presentation of completed evaluation jobs."""

import json
from pathlib import Path

from .profiles import PROFILES, PROTOCOL_VERSION


def validate_runtime(checks, profile, *, recovery_enabled=False):
    """Require observed before/after rollout evidence; never infer it from flags."""
    for sample in (checks, checks.get("after_rollout", {})):
        if (
            sample.get("profile") != profile
            or sample.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError("Missing or incompatible runtime profile evidence")
        for key in (
            "domain_randomization",
            "reset_noise",
            "recovery_reset_roles",
            "full_trajectory_is_tracking_success",
        ):
            if sample.get(key) is not False:
                raise ValueError(f"Invalid runtime evidence: {key}")
        corruption = sample.get("observation_corruption")
        if (
            not isinstance(corruption, dict)
            or not corruption
            or any(v is not False for v in corruption.values())
        ):
            raise ValueError("Missing or enabled observation corruption evidence")
        terms = sample.get("active_termination_terms")
        delays = sample.get("action_delay_ticks")
        if not isinstance(terms, list) or not isinstance(delays, dict):
            raise ValueError("Missing termination or delay evidence")  # noqa: TRY004
        if profile == "full_clean":
            if (
                terms
                or not delays
                or any(
                    not lags or any(lag != 0 for lag in lags)
                    for lags in delays.values()
                )
            ):
                raise ValueError(
                    "Full-clean runtime has termination or nonzero/missing delay"
                )
        else:
            required = {"height", "tilt", "motion_end", "time_out"}
            if recovery_enabled:
                required |= {"ankle", "wrist"}
            if not required.issubset(terms):
                raise ValueError("Strict runtime is missing native termination terms")


def build_report(results, payloads, identity):
    from ..cli.evaluate import validate_results

    expected_profiles = PROFILES if identity.get("profile") == "both" else None
    if expected_profiles is not None and set(results) != set(expected_profiles):
        raise ValueError("Both-profile report requires strict and full_clean results")
    profiles = {}
    for profile, summary in results.items():
        if (
            profile not in PROFILES
            or summary.get("profile") != profile
            or summary.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError("Invalid report protocol")
        if summary.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash changed during evaluation")
        payload = payloads[profile]
        metrics = validate_results(
            payload,
            summary["expected_clips"],
            profile=profile,
            expected_steps=summary["expected_steps"],
        )
        validate_runtime(
            payload.get("runtime_checks", {}),
            profile,
            recovery_enabled=summary.get("recovery_enabled", False),
        )
        profiles[profile] = {
            **summary,
            **metrics,
            "runtime_evidence_validated": True,
            "clips_detail": payload["clips"],
        }
    if len({tuple(sorted(p["expected_clips"])) for p in profiles.values()}) != 1:
        raise ValueError("Profile clip sets differ")
    combined = {
        **identity,
        "protocol_version": PROTOCOL_VERSION,
        "full_trajectory_is_tracking_success": False,
        "profiles": profiles,
    }
    lines = [
        "# G1 evaluation",
        "",
        f"Checkpoint SHA256: `{identity['checkpoint_sha256']}`",
        f"Source revision: `{identity['source']['source_revision']}`",
        f"Execution: `{identity['execution']}`",
        f"Scheduling: `{identity.get('scheduling', 'sequential')}`",
        f"Evaluation wall seconds: {identity.get('evaluation_wall_seconds', 'unrecorded')}",
        "",
        "Full-clean coverage is not tracking success. Strict completion uses native tracking termination.",
        "",
    ]
    for profile, summary in profiles.items():
        rate = (
            f"{summary['completed']}/40 completed"
            if profile == "strict"
            else f"{summary['covered_clips']}/40 covered; tracking success not measured"
        )
        lines += [
            f"## {profile}",
            "",
            f"Profile elapsed seconds: {summary.get('elapsed_seconds', 'unrecorded')}; log: `evaluation/{profile}/evaluation.log`",
            "",
            f"{rate}; mean normalized return: {summary['mean_normalized_return']:.6f}.",
            "",
            "| Clip | Steps | Covered to end | Strict completed | Normalized return | Root XY RMSE | Joint RMSE |",
            "|---|---:|:---:|:---:|---:|---:|---:|",
        ]
        for row in sorted(summary["clips_detail"], key=lambda r: r["name"]):

            def metric(key, row=row):
                value = row.get(key)
                return "—" if value is None else f"{value:.6f}"

            completed = (
                str(row.get("completed", False))
                if profile == "strict"
                else "not measured"
            )
            lines.append(
                f"| {row['name']} | {row['steps']} | {row.get('covered_to_end')} | {completed} | {metric('normalized_return')} | {metric('root_xy_rmse')} | {metric('joint_rmse')} |"
            )
        lines.append("")
    return combined, "\n".join(lines)


def write_report(output, results, identity):
    output = Path(output)
    payloads = {}
    for profile in results:
        directory = output / "evaluation" / profile
        payload = json.loads((directory / "perclip.json").read_text())
        runtime = json.loads((directory / "runtime_profile.json").read_text())
        if payload.get("metadata", {}).get("evaluation") != runtime:
            raise ValueError(
                "Per-clip runtime evidence differs from runtime_profile.json"
            )
        for sample in (runtime, runtime.get("after_rollout", {})):
            if sample.get("execution") != identity["execution"]:
                raise ValueError("Observed execution differs from requested execution")
        payloads[profile] = {**payload, "runtime_checks": runtime}
    combined, markdown = build_report(results, payloads, identity)
    (output / "summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")
    return combined
