"""Tests for the SOTAppR autonomous SWE application builder."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from src.cli import cli
from src.sotappr import BuilderRequest, SOTAppRBuilder, SOTAppRStop
from src.sotappr.models import ComplexityItem, EvolutionScore, FeatureInput


def _request(confirmed: bool = True) -> BuilderRequest:
    return BuilderRequest(
        organism_name="Autonomous SWE Reactor",
        stated_problem="Build and ship scoped software changes safely.",
        root_need="Reliable autonomous software delivery with proofs.",
        user_confirmed_phase1=confirmed,
    )


class TestSOTAppRBuilder:
    def test_build_generates_all_phases(self):
        builder = SOTAppRBuilder()
        report = builder.build(_request(confirmed=True))

        assert report.phase1.phase == 1
        assert report.phase2.phase == 2
        assert report.phase3.phase == 3
        assert report.phase4.phase == 4
        assert report.phase5.phase == 5
        assert report.phase6.phase == 6
        assert report.phase7.phase == 7
        assert report.phase8.phase == 8
        assert len(report.phase2.sketches) >= 3
        assert report.phase8.health_card.sota_confidence >= 1

    def test_build_stops_when_phase1_not_confirmed(self):
        builder = SOTAppRBuilder()
        with pytest.raises(SOTAppRStop, match="Phase 1 requires explicit confirmation"):
            builder.build(_request(confirmed=False))

    def test_phase2_guard_stops_on_complexity_overrun(self):
        builder = SOTAppRBuilder()
        phase1 = builder.phase1_contract_specification(_request(confirmed=True))
        phase2 = builder.phase2_divergent_architecture(_request(confirmed=True), phase1)
        phase2.complexity_budget.append(ComplexityItem(component="overbudget", points=80))

        with pytest.raises(SOTAppRStop, match="Complexity budget exceeded"):
            builder._enforce_phase2_guards(phase2)

    def test_phase6_guard_stops_on_low_evolution_score(self):
        builder = SOTAppRBuilder()
        phase1 = builder.phase1_contract_specification(_request(confirmed=True))
        phase2 = builder.phase2_divergent_architecture(_request(confirmed=True), phase1)
        phase6 = builder.phase6_future_proofing(phase2)
        phase6.evolution_scores[0] = EvolutionScore(component="phase-engine", score=2)

        with pytest.raises(SOTAppRStop, match="component below evolution score 3"):
            builder._enforce_phase6_guards(phase6)


class TestPhase3DynamicBacklog:
    """Test that phase3_growth_layers generates feature-based backlog actions."""

    def test_no_features_returns_default_backlog(self):
        builder = SOTAppRBuilder()
        request = _request(confirmed=True)
        phase1 = builder.phase1_contract_specification(request)
        phase2 = builder.phase2_divergent_architecture(request, phase1)
        phase3 = builder.phase3_growth_layers(request, phase2)

        assert len(phase3.backlog_actions) == 5
        assert phase3.backlog_actions == SOTAppRBuilder._DEFAULT_BACKLOG

    def test_features_generate_backlog_actions(self):
        builder = SOTAppRBuilder()
        request = BuilderRequest(
            organism_name="Test App",
            stated_problem="Build a test app.",
            root_need="Testing",
            user_confirmed_phase1=True,
            features=[
                FeatureInput(name="auth", description="JWT-based authentication"),
                FeatureInput(name="api", description="RESTful API endpoints"),
                FeatureInput(name="db", description="PostgreSQL integration"),
            ],
        )
        phase1 = builder.phase1_contract_specification(request)
        phase2 = builder.phase2_divergent_architecture(request, phase1)
        phase3 = builder.phase3_growth_layers(request, phase2)

        assert len(phase3.backlog_actions) == 5
        assert phase3.backlog_actions[0] == "Implement auth: JWT-based authentication"
        assert phase3.backlog_actions[1] == "Implement api: RESTful API endpoints"
        assert phase3.backlog_actions[2] == "Implement db: PostgreSQL integration"
        # Remaining 2 are padded from defaults
        assert phase3.backlog_actions[3] == SOTAppRBuilder._DEFAULT_BACKLOG[0]
        assert phase3.backlog_actions[4] == SOTAppRBuilder._DEFAULT_BACKLOG[1]

    def test_five_features_no_padding(self):
        builder = SOTAppRBuilder()
        features = [
            FeatureInput(name=f"feat-{i}", description=f"Feature number {i}")
            for i in range(5)
        ]
        request = BuilderRequest(
            organism_name="Test App",
            stated_problem="Build a test app.",
            root_need="Testing",
            user_confirmed_phase1=True,
            features=features,
        )
        phase1 = builder.phase1_contract_specification(request)
        phase2 = builder.phase2_divergent_architecture(request, phase1)
        phase3 = builder.phase3_growth_layers(request, phase2)

        assert len(phase3.backlog_actions) == 5
        for i in range(5):
            assert phase3.backlog_actions[i].startswith(f"Implement feat-{i}:")
        # No default backlog items should be present
        for default_action in SOTAppRBuilder._DEFAULT_BACKLOG:
            assert default_action not in phase3.backlog_actions

    def test_more_than_five_features_caps_at_five(self):
        builder = SOTAppRBuilder()
        features = [
            FeatureInput(name=f"feat-{i}", description=f"Feature number {i}")
            for i in range(8)
        ]
        request = BuilderRequest(
            organism_name="Test App",
            stated_problem="Build a test app.",
            root_need="Testing",
            user_confirmed_phase1=True,
            features=features,
        )
        phase1 = builder.phase1_contract_specification(request)
        phase2 = builder.phase2_divergent_architecture(request, phase1)
        phase3 = builder.phase3_growth_layers(request, phase2)

        assert len(phase3.backlog_actions) == 5
        # Only first 5 features should be used
        for i in range(5):
            assert phase3.backlog_actions[i].startswith(f"Implement feat-{i}:")

    def test_long_description_is_truncated(self):
        builder = SOTAppRBuilder()
        long_desc = "A" * 200
        request = BuilderRequest(
            organism_name="Test App",
            stated_problem="Build a test app.",
            root_need="Testing",
            user_confirmed_phase1=True,
            features=[FeatureInput(name="long-feat", description=long_desc)],
        )
        phase1 = builder.phase1_contract_specification(request)
        phase2 = builder.phase2_divergent_architecture(request, phase1)
        phase3 = builder.phase3_growth_layers(request, phase2)

        action = phase3.backlog_actions[0]
        assert action.startswith("Implement long-feat: ")
        # Description portion should be truncated to 120 chars (117 + "...")
        desc_part = action[len("Implement long-feat: "):]
        assert len(desc_part) == 120
        assert desc_part.endswith("...")


class TestSOTAppRCli:
    def test_cli_sotappr_build_writes_report(self, tmp_path):
        spec_path = tmp_path / "spec.json"
        out_path = tmp_path / "report.json"
        spec_path.write_text(
            json.dumps(
                {
                    "organism_name": "Autonomous SWE Reactor",
                    "stated_problem": "Build software safely and fast.",
                    "root_need": "Autonomous delivery with hard guarantees.",
                    "user_confirmed_phase1": True,
                }
            ),
            encoding="utf-8",
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "sotappr-build",
                "--spec",
                str(spec_path),
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["phase8"]["phase"] == 8
