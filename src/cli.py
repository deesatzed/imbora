"""CLI entrypoint for The Associate."""

from __future__ import annotations

import json
import logging
import shutil
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import click
import yaml

from src.sotappr import BuilderRequest, SOTAppRBuilder, SOTAppRStop

_MODEL_ROLES = ("builder", "sentinel", "librarian", "council", "research")

# Track the most recently active execution context for graceful shutdown
_active_project_id: str | None = None
_active_run_id: str | None = None
_tasks_processed: int = 0


def _sigint_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C with a checkpoint summary instead of bare traceback."""
    click.echo("\n")
    click.echo(click.style("Interrupted.", fg="yellow", bold=True))
    if _active_run_id:
        click.echo(f"  Run ID:          {_active_run_id}")
    if _active_project_id:
        click.echo(f"  Project ID:      {_active_project_id}")
    click.echo(f"  Tasks processed: {_tasks_processed}")
    click.echo(
        "\nRun state is persisted in the database. Resume with:\n"
        f"  associate sotappr-resume-run --run-id {_active_run_id or '<run-id>'}"
    )
    sys.exit(130)


def _setup_logging(verbose: bool = False) -> None:
    """Apply logging configuration from config/default.yaml."""
    from src.core.config import load_config

    try:
        config = load_config()
        level_name = config.logging.level
        fmt = config.logging.format
    except Exception:
        level_name = "INFO"
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    level = logging.DEBUG if verbose else getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


@click.group()
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose (DEBUG) logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """The Associate command line interface."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose=verbose)
    signal.signal(signal.SIGINT, _sigint_handler)


@cli.command("sotappr-build")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to JSON or YAML BuilderRequest spec.",
)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("sotappr_report.json"),
    show_default=True,
    help="Output path for serialized SOTAppR report.",
)
def sotappr_build(spec_path: Path, out_path: Path) -> None:
    """Run the full SOTAppR Phase 0-8 builder flow."""
    payload = _load_spec(spec_path)
    request = BuilderRequest(**payload)
    builder = SOTAppRBuilder()

    try:
        report = builder.build(request)
    except SOTAppRStop as exc:
        raise click.ClickException(str(exc)) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Wrote SOTAppR report to {out_path}")


@cli.command("sotappr-models")
@click.option(
    "--criteria",
    required=False,
    type=click.Choice(["cost", "latency", "newness"]),
    default="newness",
    show_default=True,
    help="Ranking criteria for returned models.",
)
@click.option(
    "--limit",
    required=False,
    type=int,
    default=20,
    show_default=True,
    help="Maximum number of models to display.",
)
@click.option("--provider", required=False, default=None, help="Optional provider filter.")
@click.option(
    "--search",
    required=False,
    default=None,
    help="Optional substring filter over model id/name/description.",
)
@click.option(
    "--max-prompt-cost",
    required=False,
    type=float,
    default=None,
    help="Optional max input-token price filter.",
)
@click.option(
    "--max-completion-cost",
    required=False,
    type=float,
    default=None,
    help="Optional max output-token price filter.",
)
@click.option(
    "--include-free/--only-paid",
    default=True,
    show_default=True,
    help="Include or exclude zero-cost models.",
)
@click.option(
    "--latency-probe-limit",
    required=False,
    type=int,
    default=80,
    show_default=True,
    help="How many recent models to probe for live latency metrics.",
)
@click.option(
    "--json-out",
    is_flag=True,
    default=False,
    help="Emit JSON instead of table output.",
)
@click.option(
    "--choose",
    is_flag=True,
    default=False,
    help="Interactively choose one listed model and apply it to a role.",
)
@click.option(
    "--role",
    required=False,
    type=click.Choice(_MODEL_ROLES),
    default=None,
    help="Role to update when using --choose.",
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional config directory containing models.yaml.",
)
@click.option(
    "--api-key",
    required=False,
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="Optional OpenRouter API key for catalog queries.",
)
def sotappr_models(
    criteria: str,
    limit: int,
    provider: str | None,
    search: str | None,
    max_prompt_cost: float | None,
    max_completion_cost: float | None,
    include_free: bool,
    latency_probe_limit: int,
    json_out: bool,
    choose: bool,
    role: str | None,
    config_dir: Path | None,
    api_key: str | None,
) -> None:
    """Query OpenRouter and rank latest models by cost, latency, or newness."""
    catalog_cls = _load_openrouter_model_catalog()
    catalog = catalog_cls(api_key=api_key)
    try:
        models = catalog.list_models(
            criteria=criteria,
            limit=limit,
            provider=provider,
            search=search,
            max_prompt_cost=max_prompt_cost,
            max_completion_cost=max_completion_cost,
            include_free=include_free,
            latency_probe_limit=latency_probe_limit,
        )
    except Exception as exc:
        raise click.ClickException(
            f"Failed to query OpenRouter models: {exc}\n"
            "Verify your API key: associate --verbose sotappr-models --api-key <key>"
        ) from exc

    if not models:
        raise click.ClickException("No models matched the requested filters.")

    if json_out:
        _echo_json(
            {
                "criteria": criteria,
                "count": len(models),
                "models": models,
            }
        )
    else:
        _echo_model_table(models=models, criteria=criteria)

    if not choose:
        return

    picked_index = click.prompt(
        "Select model number",
        type=click.IntRange(1, len(models)),
    )
    selected = models[picked_index - 1]
    selected_model_id = str(selected["id"])
    target_role = role
    if target_role is None:
        target_role = click.prompt(
            "Role to update",
            type=click.Choice(_MODEL_ROLES, case_sensitive=False),
        ).lower()

    models_path = _persist_model_role(
        role=target_role,
        model_id=selected_model_id,
        config_dir=config_dir,
    )
    click.echo(
        f"Updated role '{target_role}' to '{selected_model_id}' in {models_path}"
    )


@cli.command("sotappr-execute")
@click.option(
    "--spec",
    "spec_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to JSON or YAML BuilderRequest spec.",
)
@click.option(
    "--repo-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Repository path to execute tasks against.",
)
@click.option(
    "--report-out",
    "report_out",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("sotappr_report.json"),
    show_default=True,
    help="Output path for serialized SOTAppR report.",
)
@click.option(
    "--export-tasks",
    "export_tasks_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to export seeded task preview JSON.",
)
@click.option(
    "--governance-pack",
    required=False,
    default=None,
    help="Override governance pack (defaults to config.sotappr.governance_pack).",
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional config directory for ComponentFactory.",
)
@click.option("--env", required=False, default=None, help="Optional config overlay environment.")
@click.option(
    "--api-key",
    required=False,
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="OpenRouter API key (falls back to OPENROUTER_API_KEY).",
)
@click.option(
    "--test-command",
    required=False,
    default="pytest tests/ -v",
    show_default=True,
    help="Command used by Builder to run tests.",
)
@click.option(
    "--max-iterations",
    required=False,
    type=int,
    default=100,
    show_default=True,
    help="Maximum TaskLoop iterations.",
)
@click.option(
    "--max-runtime-hours",
    required=False,
    type=float,
    default=None,
    help="Optional runtime stop cap in hours (run pauses if exceeded).",
)
@click.option(
    "--max-runtime-cost-usd",
    required=False,
    type=float,
    default=None,
    help="Optional runtime stop cap in USD (estimated from tokens).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Build report and seed tasks, but do not run TaskLoop execution.",
)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Skip execution confirmation prompt.",
)
def sotappr_execute(
    spec_path: Path,
    repo_path: Path,
    report_out: Path,
    export_tasks_path: Path | None,
    governance_pack: str | None,
    config_dir: Path | None,
    env: str | None,
    api_key: str | None,
    test_command: str,
    max_iterations: int,
    max_runtime_hours: float | None,
    max_runtime_cost_usd: float | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Build via SOTAppR, seed backlog tasks, and execute via TaskLoop."""
    global _active_project_id, _active_run_id, _tasks_processed

    payload = _load_spec(spec_path)
    request = BuilderRequest(**payload)
    builder = SOTAppRBuilder()

    click.echo(click.style("SOTAppR Phase 0-8: building architecture plan...", bold=True))
    try:
        report = builder.build(request)
    except SOTAppRStop as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(click.style("SOTAppR plan complete.", bold=True))

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    task_count = len(report.phase3.backlog_actions)
    if not dry_run and not yes:
        click.echo(
            f"Seeding {task_count} task(s) and executing with max {max_iterations} iterations."
        )
        if not click.confirm("Proceed?", default=True):
            raise click.Abort()

    def _progress(message: str) -> None:
        """CLI progress callback — styled output to stderr."""
        global _tasks_processed
        if message.startswith("[TASK]"):
            _tasks_processed += 1
            click.echo(click.style(message, fg="cyan"))
        elif message.startswith("[DONE]"):
            click.echo(click.style(message, fg="green", bold=True))
        elif message.startswith("[WARN]"):
            click.echo(click.style(message, fg="yellow"))
        elif "FAILED" in message or "REJECTED" in message:
            click.echo(click.style(message, fg="red"))
        elif "DONE" in message:
            click.echo(click.style(message, fg="green"))
        else:
            click.echo(message)

    component_factory = _load_component_factory()
    executor_class = _load_sotappr_executor()

    bundle = None
    try:
        bundle = component_factory.create(config_dir=config_dir, env=env, api_key=api_key)
        executor = executor_class.from_bundle(
            bundle=bundle,
            repo_path=str(repo_path),
            test_command=test_command,
            progress_callback=_progress,
        )
        task_preview = executor.preview_seeded_tasks(report.phase3.backlog_actions)
        if export_tasks_path is not None:
            export_tasks_path.parent.mkdir(parents=True, exist_ok=True)
            export_tasks_path.write_text(
                json.dumps(task_preview, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

        effective_pack = governance_pack or bundle.config.sotappr.governance_pack
        mode = "dry-run" if dry_run else "execute"

        # Set global state for Ctrl+C handler
        _active_project_id = None
        _active_run_id = None

        execution_summary = executor.bootstrap_and_execute(
            request=request,
            report=report,
            repo_path=str(repo_path),
            max_iterations=max_iterations,
            mode=mode,
            governance_pack=effective_pack,
            report_path=str(report_out),
            execute=not dry_run,
            budget_contract=_runtime_budget_contract(
                max_hours=max_runtime_hours,
                max_cost_usd=max_runtime_cost_usd,
                estimated_cost_per_1k_tokens_usd=getattr(
                    bundle.config.sotappr,
                    "estimated_cost_per_1k_tokens_usd",
                    0.02,
                ),
            ),
        )

        _active_project_id = execution_summary.project_id
        _active_run_id = execution_summary.run_id

        summary = {
            "project_id": execution_summary.project_id,
            "tasks_seeded": execution_summary.tasks_seeded,
            "tasks_processed": execution_summary.tasks_processed,
            "run_id": execution_summary.run_id,
            "mode": execution_summary.mode,
            "status": execution_summary.status,
            "stop_reason": execution_summary.stop_reason,
            "estimated_cost_usd": execution_summary.estimated_cost_usd,
            "elapsed_hours": execution_summary.elapsed_hours,
        }
        _archive_report(
            report_out=report_out,
            archive_dir=Path(bundle.config.sotappr.report_archive_dir),
            run_id=execution_summary.run_id,
        )
    except SOTAppRStop as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(
            f"SOTAppR execution failed: {exc}\n"
            "Check logs with --verbose for details. "
            "If the run was partially completed, resume with:\n"
            f"  associate sotappr-resume-run --run-id {_active_run_id or '<run-id>'}"
        ) from exc
    finally:
        if bundle is not None:
            component_factory.close(bundle)

    click.echo(f"Wrote SOTAppR report to {report_out}")
    if export_tasks_path is not None:
        click.echo(f"Exported seeded-task preview to {export_tasks_path}")
    click.echo(
        click.style("\nExecution summary:", bold=True) + "\n"
        f"  Project ID:      {summary['project_id']}\n"
        f"  Run ID:          {summary['run_id']}\n"
        f"  Mode:            {summary['mode']}\n"
        f"  Status:          {summary['status']}\n"
        f"  Tasks seeded:    {summary['tasks_seeded']}\n"
        f"  Tasks processed: {summary['tasks_processed']}\n"
        f"  Est. cost:       ${summary['estimated_cost_usd']:.4f}\n"
        f"  Elapsed hours:   {summary['elapsed_hours']:.4f}"
    )
    if summary.get("stop_reason"):
        click.echo(click.style(f"  Stop reason:     {summary['stop_reason']}", fg="yellow"))
    if dry_run:
        click.echo(click.style("Dry-run mode: TaskLoop execution was skipped.", fg="yellow"))


@cli.command("mission-start")
@click.option(
    "--task",
    "task_text",
    required=True,
    help="Primary task statement (new app or enhancement request).",
)
@click.option(
    "--target-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path where the project exists or should be created.",
)
@click.option(
    "--feature",
    "features",
    multiple=True,
    help="Feature to include in alignment scope (repeatable).",
)
@click.option(
    "--ux-expectation",
    "ux_expectations",
    multiple=True,
    help="UX expectation to enforce (repeatable).",
)
@click.option(
    "--acceptance-criterion",
    "acceptance_criteria",
    multiple=True,
    help="Acceptance criterion for approval gates (repeatable).",
)
@click.option(
    "--risk",
    "risks",
    multiple=True,
    help="Known risk or constraint (repeatable).",
)
@click.option(
    "--methodology",
    required=False,
    default="retrieval-grounded multi-candidate build with governance gates",
    show_default=True,
)
@click.option("--max-hours", required=False, type=float, default=None)
@click.option("--max-cost-usd", required=False, type=float, default=None)
@click.option(
    "--knowledge-local",
    "knowledge_local",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Local file/folder to add into the knowledge set (repeatable).",
)
@click.option(
    "--knowledge-url",
    "knowledge_urls",
    multiple=True,
    help="External URL to include in knowledge manifest (repeatable).",
)
@click.option(
    "--knowledge-query",
    "knowledge_queries",
    multiple=True,
    help="Search query to gather and save into knowledge folder (repeatable).",
)
@click.option(
    "--auto-gather-knowledge",
    is_flag=True,
    default=False,
    help="Run Tavily searches for --knowledge-query and save results.",
)
@click.option(
    "--anticipated-end-goal",
    required=False,
    default=None,
    help="Final expected outcome summary.",
)
@click.option(
    "--execute",
    is_flag=True,
    default=False,
    help="After gate approvals, run SOTAppR execution immediately.",
)
@click.option(
    "--max-iterations",
    required=False,
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    "--governance-pack",
    required=False,
    default=None,
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--env", required=False, default=None)
@click.option("--api-key", required=False, default=None, envvar="OPENROUTER_API_KEY")
@click.option("--test-command", required=False, default="pytest tests/ -v", show_default=True)
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Do not prompt for confirmations; auto-approve gates.",
)
def mission_start(
    task_text: str,
    target_path: Path,
    features: tuple[str, ...],
    ux_expectations: tuple[str, ...],
    acceptance_criteria: tuple[str, ...],
    risks: tuple[str, ...],
    methodology: str,
    max_hours: float | None,
    max_cost_usd: float | None,
    knowledge_local: tuple[Path, ...],
    knowledge_urls: tuple[str, ...],
    knowledge_queries: tuple[str, ...],
    auto_gather_knowledge: bool,
    anticipated_end_goal: str | None,
    execute: bool,
    max_iterations: int,
    governance_pack: str | None,
    config_dir: Path | None,
    env: str | None,
    api_key: str | None,
    test_command: str,
    non_interactive: bool,
) -> None:
    """Guided UX-first workflow with explicit approval gates before execution."""
    from src.core.config import load_config, load_model_registry
    from src.tools.search import TavilyClient
    from src.workflow.session import (
        AlignmentSpec,
        ExecutionCharter,
        IntakeCard,
        SessionPlan,
        build_runtime_manifest,
        estimate_budget,
        materialize_knowledge_items,
        write_session_artifacts,
    )

    preexisting = target_path.exists()
    target_path.mkdir(parents=True, exist_ok=True)
    config = load_config(config_dir=config_dir, env=env)
    model_registry = load_model_registry(config_dir=config_dir)
    session_id = _new_session_id()
    feature_list = list(features) if features else [task_text]
    ux_list = list(ux_expectations) if ux_expectations else ["clear and predictable user flow"]
    acceptance_list = (
        list(acceptance_criteria)
        if acceptance_criteria
        else ["all critical tests pass", "sentinel approvals complete"]
    )
    risk_list = list(risks) if risks else []

    intake = IntakeCard(
        session_id=session_id,
        task=task_text,
        target_path=str(target_path),
        project_mode="existing" if preexisting else "new",
    )
    alignment = AlignmentSpec(
        ux_expectations=ux_list,
        features=feature_list,
        methodology=methodology,
        acceptance_criteria=acceptance_list,
        risks=risk_list,
        approved=False,
    )

    budget = estimate_budget(
        task=task_text,
        feature_count=len(feature_list),
        knowledge_count=len(knowledge_local) + len(knowledge_urls) + len(knowledge_queries),
    )
    if max_hours is not None:
        budget.max_hours = max_hours
    if max_cost_usd is not None:
        budget.max_cost_usd = max_cost_usd

    click.echo(click.style("Gate 1: Alignment", bold=True))
    click.echo(f"- Task: {task_text}")
    click.echo(f"- Target path: {target_path}")
    click.echo(f"- Features: {len(feature_list)}")
    click.echo(f"- Methodology: {methodology}")
    if not non_interactive:
        alignment = _run_alignment_gate(alignment=alignment)
    else:
        alignment.approved = True

    click.echo(click.style("Gate 2: Budget", bold=True))
    click.echo(
        f"- Estimated: {budget.estimated_hours:.1f}h, ${budget.estimated_cost_usd:.2f}, "
        f"{budget.estimated_tokens:,} tokens"
    )
    if not non_interactive:
        budget = _run_budget_gate(budget=budget)
    else:
        budget.approved = True

    session_root = target_path / ".associate" / "sessions" / session_id
    knowledge_dir = session_root / "knowledge"
    local_knowledge = list(knowledge_local)
    url_knowledge = list(knowledge_urls)
    query_knowledge = list(knowledge_queries)

    query_results = _gather_query_knowledge(
        queries=query_knowledge,
        auto_gather=auto_gather_knowledge,
        search_client=TavilyClient(config=config.search) if auto_gather_knowledge else None,
    )
    knowledge_manifest = materialize_knowledge_items(
        knowledge_dir=knowledge_dir,
        local_paths=local_knowledge,
        urls=url_knowledge,
        query_results=query_results or None,
    )

    charter_goal = anticipated_end_goal or (
        f"Deliver '{task_text}' in {target_path.name or str(target_path)} with aligned UX, "
        "budget compliance, and governance-passing implementation."
    )
    charter = ExecutionCharter(
        anticipated_end_goal=charter_goal,
        execution_summary=(
            "Associate will configure runtime schema/agents/prompts, connect knowledge sources, "
            "and execute through bounded multi-candidate delivery loops."
        ),
        next_actions=[
            "initialize runtime manifest",
            "seed implementation plan",
            "execute task loop with governance",
        ],
        approved=non_interactive,
    )

    if not non_interactive:
        while True:
            click.echo(click.style("Gate 3: Final Charter", bold=True))
            click.echo(f"- End goal: {charter.anticipated_end_goal}")
            click.echo(f"- Knowledge items: {len(knowledge_manifest.items)}")
            click.echo(
                f"- Budget cap: {budget.max_hours:.1f}h / ${budget.max_cost_usd:.2f}"
            )
            if click.confirm("Approve execution charter?", default=True):
                charter.approved = True
                break

            loop_back = click.prompt(
                "Send back to gate",
                type=click.Choice(["alignment", "budget", "knowledge", "charter", "abort"]),
                default="charter",
                show_default=True,
            )
            if loop_back == "abort":
                raise click.Abort()
            if loop_back == "alignment":
                alignment = _run_alignment_gate(alignment=alignment)
                continue
            if loop_back == "budget":
                budget = _run_budget_gate(budget=budget)
                continue
            if loop_back == "knowledge":
                local_knowledge, url_knowledge, query_knowledge = _run_knowledge_gate(
                    local_paths=local_knowledge,
                    urls=url_knowledge,
                    queries=query_knowledge,
                )
                if knowledge_dir.exists():
                    shutil.rmtree(knowledge_dir)
                refreshed_queries = _gather_query_knowledge(
                    queries=query_knowledge,
                    auto_gather=auto_gather_knowledge,
                    search_client=TavilyClient(config=config.search) if auto_gather_knowledge else None,
                )
                knowledge_manifest = materialize_knowledge_items(
                    knowledge_dir=knowledge_dir,
                    local_paths=local_knowledge,
                    urls=url_knowledge,
                    query_results=refreshed_queries or None,
                )
                continue

            charter.anticipated_end_goal = click.prompt(
                "Anticipated end goal",
                default=charter.anticipated_end_goal,
                show_default=True,
            )
            charter.execution_summary = click.prompt(
                "Execution summary",
                default=charter.execution_summary,
                show_default=True,
            )
            charter.next_actions = _prompt_csv_items(
                label="Next actions (comma-separated)",
                current=charter.next_actions,
            )
    else:
        charter.approved = True

    runtime_manifest = build_runtime_manifest(
        model_roles=model_registry.roles,
        prompts_dir=_resolve_prompts_dir(config_dir=config_dir),
        knowledge_manifest=knowledge_manifest,
    )
    plan = SessionPlan(
        session_id=session_id,
        intake=intake,
        alignment=alignment,
        budget=budget,
        knowledge=knowledge_manifest,
        charter=charter,
        runtime=runtime_manifest,
        status="approved",
    )
    outputs = write_session_artifacts(plan=plan, session_root=session_root)
    click.echo(f"Session artifacts written: {outputs['session']}")

    if not execute:
        click.echo("Plan-only mode complete. Run with --execute to start autonomous delivery.")
        return

    spec_path = session_root / "builder_request.json"
    spec_payload = {
        "organism_name": _derive_organism_name(task_text=task_text),
        "stated_problem": task_text,
        "root_need": charter_goal,
        "features": [{"name": f, "description": f} for f in feature_list],
        "user_confirmed_phase1": True,
        "budget_ceiling": f"${budget.max_cost_usd:.2f}",
    }
    spec_path.write_text(json.dumps(spec_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    report_out = session_root / "sotappr_report.json"
    ctx = click.get_current_context()
    ctx.invoke(
        sotappr_execute,
        spec_path=spec_path,
        repo_path=target_path,
        report_out=report_out,
        export_tasks_path=session_root / "seeded_tasks.json",
        governance_pack=governance_pack,
        config_dir=config_dir,
        env=env,
        api_key=api_key,
        test_command=test_command,
        max_iterations=max_iterations,
        max_runtime_hours=budget.max_hours,
        max_runtime_cost_usd=budget.max_cost_usd,
        dry_run=False,
        yes=True,
    )


@cli.command("sotappr-status")
@click.option("--run-id", required=False, default=None, help="Run ID to inspect.")
@click.option("--project-id", required=False, default=None, help="Optional project filter.")
@click.option("--limit", required=False, default=20, show_default=True, type=int)
def sotappr_status(run_id: str | None, project_id: str | None, limit: int) -> None:
    """Show SOTAppR run status and review queues."""
    ctx = _open_repo_context()
    try:
        if run_id is None:
            project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
            runs = ctx.repository.list_sotappr_runs(limit=limit, project_id=project_uuid)
            rows = [
                {
                    "run_id": str(row["id"]),
                    "project_id": str(row["project_id"]),
                    "mode": row["mode"],
                    "status": row["status"],
                    "governance_pack": row.get("governance_pack"),
                    "tasks_seeded": int(row.get("tasks_seeded") or 0),
                    "tasks_processed": int(row.get("tasks_processed") or 0),
                    "stop_reason": row.get("stop_reason"),
                    "estimated_cost_usd": float(row.get("estimated_cost_usd") or 0.0),
                    "elapsed_hours": float(row.get("elapsed_hours") or 0.0),
                    "paused_by_budget": _is_budget_pause(row.get("stop_reason"), row.get("last_error")),
                    "created_at": _iso(row.get("created_at")),
                    "updated_at": _iso(row.get("updated_at")),
                }
                for row in runs
            ]
            _echo_json({"runs": rows, "count": len(rows)})
            return

        run_uuid = _parse_uuid(run_id, "run_id")
        run = ctx.repository.get_sotappr_run(run_uuid)
        if run is None:
            raise click.ClickException(
                f"Run not found: {run_id}\n"
                "List available runs with: associate sotappr-status --limit 20"
            )

        project_uuid = UUID(str(run["project_id"]))
        status_counts = ctx.repository.get_task_status_summary(project_id=project_uuid)
        review_queue = ctx.repository.list_review_queue(limit=limit, project_id=project_uuid)
        artifacts = ctx.repository.list_sotappr_artifacts(run_id=run_uuid, limit=200)
        latest_failover_state = _latest_model_failover_state(artifacts)
        retry_cause_counts = _retry_cause_counts(artifacts)
        trace_event_count = sum(1 for row in artifacts if str(row.get("artifact_type")) == "trace_event")

        _echo_json(
            {
                "run": {
                    "id": str(run["id"]),
                    "project_id": str(run["project_id"]),
                    "mode": run["mode"],
                    "status": run["status"],
                    "governance_pack": run.get("governance_pack"),
                    "tasks_seeded": int(run.get("tasks_seeded") or 0),
                    "tasks_processed": int(run.get("tasks_processed") or 0),
                    "last_error": run.get("last_error"),
                    "stop_reason": run.get("stop_reason"),
                    "estimated_cost_usd": float(run.get("estimated_cost_usd") or 0.0),
                    "elapsed_hours": float(run.get("elapsed_hours") or 0.0),
                    "paused_by_budget": _is_budget_pause(run.get("stop_reason"), run.get("last_error")),
                    "created_at": _iso(run.get("created_at")),
                    "updated_at": _iso(run.get("updated_at")),
                    "completed_at": _iso(run.get("completed_at")),
                },
                "task_status_counts": status_counts,
                "review_queue": [
                    {
                        "task_id": str(task.id),
                        "title": task.title,
                        "priority": task.priority,
                        "attempt_count": task.attempt_count,
                    }
                    for task in review_queue
                ],
                "artifact_count": len(artifacts),
                "trace_event_count": trace_event_count,
                "retry_cause_counts": retry_cause_counts,
                "model_failover_state": latest_failover_state,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-resume-run")
@click.option("--run-id", required=True, help="Run ID to resume.")
@click.option(
    "--repo-path",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional repository path override (defaults to stored run repo_path).",
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
)
@click.option("--env", required=False, default=None)
@click.option("--api-key", required=False, default=None, envvar="OPENROUTER_API_KEY")
@click.option("--test-command", required=False, default="pytest tests/ -v", show_default=True)
@click.option("--max-iterations", required=False, default=100, show_default=True, type=int)
@click.option("--max-runtime-hours", required=False, default=None, type=float)
@click.option("--max-runtime-cost-usd", required=False, default=None, type=float)
def sotappr_resume_run(
    run_id: str,
    repo_path: Path | None,
    config_dir: Path | None,
    env: str | None,
    api_key: str | None,
    test_command: str,
    max_iterations: int,
    max_runtime_hours: float | None,
    max_runtime_cost_usd: float | None,
) -> None:
    """Resume a paused/failed SOTAppR run."""
    run_uuid = _parse_uuid(run_id, "run_id")
    component_factory = _load_component_factory()
    executor_class = _load_sotappr_executor()

    bundle = None
    try:
        bundle = component_factory.create(config_dir=config_dir, env=env, api_key=api_key)
        run = bundle.repository.get_sotappr_run(run_uuid)
        if run is None:
            raise click.ClickException(
                f"Run not found: {run_id}\n"
                "List available runs with: associate sotappr-status --limit 20"
            )

        existing_processed = int(run.get("tasks_processed") or 0)
        existing_seeded = int(run.get("tasks_seeded") or 0)
        click.echo(
            f"Resuming run {run_id[:8]}... "
            f"({existing_processed}/{existing_seeded} tasks previously processed)"
        )

        effective_repo_path = (
            str(repo_path) if repo_path is not None else str(run.get("repo_path") or "")
        )
        if not effective_repo_path:
            raise click.ClickException(
                "Run record has no repo_path; pass --repo-path explicitly."
            )

        def _resume_progress(message: str) -> None:
            if message.startswith("[TASK]"):
                click.echo(click.style(message, fg="cyan"))
            elif message.startswith("[DONE]"):
                click.echo(click.style(message, fg="green", bold=True))
            elif "FAILED" in message or "REJECTED" in message:
                click.echo(click.style(message, fg="red"))
            else:
                click.echo(message)

        executor = executor_class.from_bundle(
            bundle=bundle,
            repo_path=effective_repo_path,
            test_command=test_command,
            progress_callback=_resume_progress,
        )
        budget_contract = _runtime_budget_contract(
            max_hours=max_runtime_hours,
            max_cost_usd=max_runtime_cost_usd,
            estimated_cost_per_1k_tokens_usd=getattr(
                bundle.config.sotappr,
                "estimated_cost_per_1k_tokens_usd",
                0.02,
            ),
        )
        try:
            summary = executor.resume_run(
                run_uuid,
                max_iterations=max_iterations,
                budget_contract=budget_contract,
            )
        except TypeError:
            summary = executor.resume_run(run_uuid, max_iterations=max_iterations)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"SOTAppR resume failed: {exc}\n"
            "Check logs with --verbose for details."
        ) from exc
    finally:
        if bundle is not None:
            component_factory.close(bundle)

    click.echo(
        click.style("\nResume summary:", bold=True) + "\n"
        f"  Project ID:      {summary.project_id}\n"
        f"  Run ID:          {summary.run_id}\n"
        f"  Status:          {summary.status}\n"
        f"  Tasks seeded:    {summary.tasks_seeded}\n"
        f"  Tasks processed: {summary.tasks_processed}\n"
        f"  Est. cost:       ${summary.estimated_cost_usd:.4f}\n"
        f"  Elapsed hours:   {summary.elapsed_hours:.4f}"
    )
    if summary.stop_reason:
        click.echo(click.style(f"  Stop reason:     {summary.stop_reason}", fg="yellow"))


@cli.command("sotappr-approve-task")
@click.option("--task-id", required=True, help="Task ID waiting in REVIEWING.")
def sotappr_approve_task(task_id: str) -> None:
    """Approve a REVIEWING task and mark it DONE."""
    task_uuid = _parse_uuid(task_id, "task_id")
    ctx = _open_repo_context()
    try:
        task = ctx.repository.approve_task(task_uuid)
        if task is None:
            existing = ctx.repository.get_task(task_uuid)
            if existing is None:
                raise click.ClickException(f"Task not found: {task_id}")
            raise click.ClickException(
                f"Task {task_id} is in status {existing.status.value}; "
                "only REVIEWING can be approved."
            )
        click.echo(f"Approved task {task.id} ({task.title}) -> {task.status.value}")
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-reset-task-safe")
@click.option("--task-id", required=True, help="Task ID to reset for retry.")
@click.option(
    "--status",
    required=False,
    default="PENDING",
    type=click.Choice(["PENDING", "RESEARCHING", "CODING", "REVIEWING"]),
    show_default=True,
    help="Status to assign after reset.",
)
def sotappr_reset_task_safe(task_id: str, status: str) -> None:
    """Reset a task for retry and reconcile attempt counters safely."""
    from src.core.models import TaskStatus

    task_uuid = _parse_uuid(task_id, "task_id")
    status_enum = TaskStatus(status)
    ctx = _open_repo_context()
    try:
        task = ctx.repository.reset_task_for_retry(task_uuid, status=status_enum)
        if task is None:
            raise click.ClickException(f"Task not found: {task_id}")
        next_attempt = ctx.repository.get_next_hypothesis_attempt(task_uuid)
        _echo_json(
            {
                "task_id": str(task.id),
                "title": task.title,
                "status": task.status.value,
                "attempt_count": task.attempt_count,
                "council_count": task.council_count,
                "next_hypothesis_attempt": next_attempt,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-dashboard")
@click.option("--project-id", required=False, default=None)
@click.option("--limit", required=False, default=20, type=int, show_default=True)
def sotappr_dashboard(project_id: str | None, limit: int) -> None:
    """Operational dashboard with run/task/error telemetry."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        runs = ctx.repository.list_sotappr_runs(limit=limit, project_id=project_uuid)
        status_counts = ctx.repository.get_task_status_summary(project_id=project_uuid)
        review_queue = ctx.repository.list_review_queue(limit=limit, project_id=project_uuid)
        error_stats = ctx.repository.get_hypothesis_error_stats(project_id=project_uuid)

        run_status_counts: dict[str, int] = {}
        budget_paused_runs = 0
        latest_budget_pause: list[dict[str, Any]] = []
        aggregated_retry_causes: dict[str, int] = {}
        failover_models_in_cooldown: dict[str, int] = {}
        failover_models_failure_counts: dict[str, int] = {}
        runs_with_failover_state = 0
        protocol_effectiveness: list[dict[str, Any]] = []
        transfer_quality: list[dict[str, Any]] = []
        drift_correlation: list[dict[str, Any]] = []
        transfer_arbitration_summary: dict[str, Any] = _empty_transfer_arbitration_summary()
        for row in runs:
            key = str(row.get("status") or "unknown")
            run_status_counts[key] = run_status_counts.get(key, 0) + 1
            if _is_budget_pause(row.get("stop_reason"), row.get("last_error")):
                budget_paused_runs += 1
                latest_budget_pause.append(
                    {
                        "run_id": str(row.get("id")),
                        "stop_reason": row.get("stop_reason") or row.get("last_error"),
                        "estimated_cost_usd": float(row.get("estimated_cost_usd") or 0.0),
                        "elapsed_hours": float(row.get("elapsed_hours") or 0.0),
                    }
                )
            run_id_raw = row.get("id")
            if run_id_raw is None or not hasattr(ctx.repository, "list_sotappr_artifacts"):
                continue
            try:
                run_artifacts = ctx.repository.list_sotappr_artifacts(
                    run_id=UUID(str(run_id_raw)),
                    limit=200,
                )
            except Exception:
                continue
            for cause, count in _retry_cause_counts(run_artifacts).items():
                aggregated_retry_causes[cause] = aggregated_retry_causes.get(cause, 0) + count
            failover_state = _latest_model_failover_state(run_artifacts)
            if failover_state:
                runs_with_failover_state += 1
                _accumulate_model_failover_state(
                    failover_state=failover_state,
                    cooldown_counts=failover_models_in_cooldown,
                    failure_counts=failover_models_failure_counts,
                )
        if hasattr(ctx.repository, "get_protocol_effectiveness"):
            try:
                effectiveness_rows = ctx.repository.get_protocol_effectiveness(
                    project_id=project_uuid,
                    limit=10,
                )
            except Exception:
                effectiveness_rows = []
            for row in effectiveness_rows:
                transfer_count = int(row.get("transfer_count") or 0)
                accepted_count = int(row.get("accepted_count") or 0)
                acceptance_rate = (accepted_count / transfer_count) if transfer_count > 0 else 0.0
                avg_outcome = float(row.get("avg_outcome")) if row.get("avg_outcome") is not None else 0.0
                avg_drift = float(row.get("avg_drift_risk")) if row.get("avg_drift_risk") is not None else 1.0
                score = (0.5 * acceptance_rate) + (0.35 * avg_outcome) + (0.15 * (1.0 - avg_drift))
                protocol_effectiveness.append(
                    {
                        "protocol_id": row.get("protocol_id"),
                        "transfer_count": transfer_count,
                        "accepted_count": accepted_count,
                        "acceptance_rate": round(acceptance_rate, 6),
                        "avg_outcome": round(avg_outcome, 6),
                        "avg_drift_risk": round(avg_drift, 6),
                        "effectiveness_score": round(score, 6),
                    }
                )
            protocol_effectiveness = sorted(
                protocol_effectiveness,
                key=lambda row: float(row.get("effectiveness_score") or 0.0),
                reverse=True,
            )[:10]
        if hasattr(ctx.repository, "get_transfer_quality_rollup"):
            try:
                quality_rows = ctx.repository.get_transfer_quality_rollup(project_id=project_uuid)
            except Exception:
                quality_rows = []
            transfer_quality = [
                {
                    "transfer_mode": row.get("transfer_mode"),
                    "total_count": int(row.get("total_count") or 0),
                    "accepted_count": int(row.get("accepted_count") or 0),
                    "cross_use_case_count": int(row.get("cross_use_case_count") or 0),
                }
                for row in quality_rows
            ]
        if hasattr(ctx.repository, "get_transfer_eval_correlation"):
            try:
                corr_rows = ctx.repository.get_transfer_eval_correlation(project_id=project_uuid)
            except Exception:
                corr_rows = []
            drift_correlation = [
                {
                    "has_cross_use_case_transfer": bool(row.get("has_cross_use_case_transfer", False)),
                    "task_count": int(row.get("task_count") or 0),
                    "avg_outcome": (
                        round(float(row.get("avg_outcome")), 6)
                        if row.get("avg_outcome") is not None
                        else None
                    ),
                    "avg_drift_risk": (
                        round(float(row.get("avg_drift_risk")), 6)
                        if row.get("avg_drift_risk") is not None
                        else None
                    ),
                }
                for row in corr_rows
            ]
        transfer_arbitration_summary = _summarize_transfer_arbitration(
            _collect_transfer_arbitration_decisions(
                repository=ctx.repository,
                project_id=project_uuid,
                limit=max(100, limit * 3),
            )
        )

        _echo_json(
            {
                "project_id": str(project_uuid) if project_uuid else None,
                "run_status_counts": run_status_counts,
                "budget_paused_runs": budget_paused_runs,
                "latest_budget_pauses": latest_budget_pause[:10],
                "retry_cause_counts": aggregated_retry_causes,
                "model_failover_summary": {
                    "runs_with_failover_state": runs_with_failover_state,
                    "models_in_cooldown": failover_models_in_cooldown,
                    "models_failure_counts": failover_models_failure_counts,
                },
                "task_status_counts": status_counts,
                "review_queue_size": len(review_queue),
                "review_queue": [
                    {
                        "task_id": str(task.id),
                        "title": task.title,
                        "priority": task.priority,
                    }
                    for task in review_queue[:10]
                ],
                "top_hypothesis_errors": [
                    {
                        "error_signature": row.get("error_signature"),
                        "count": int(row.get("cnt") or 0),
                    }
                    for row in error_stats[:10]
                ],
                "protocol_effectiveness_leaderboard": protocol_effectiveness,
                "transfer_quality": transfer_quality,
                "transfer_drift_correlation": drift_correlation,
                "transfer_arbitration_summary": transfer_arbitration_summary,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-benchmark-freeze")
@click.option("--project-id", required=False, default=None)
@click.option("--limit", required=False, default=20, type=int, show_default=True)
@click.option(
    "--statuses",
    required=False,
    default="completed,paused",
    show_default=True,
    help="Comma-separated run statuses to include.",
)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("artifacts/sotappr/frozen_benchmark.json"),
    show_default=True,
)
def sotappr_benchmark_freeze(
    project_id: str | None,
    limit: int,
    statuses: str,
    out_path: Path,
) -> None:
    """Freeze a benchmark baseline snapshot from recent SOTAppR runs."""
    from src.sotappr.benchmark import (
        BenchmarkTolerances,
        collect_run_metrics_from_repository,
        make_frozen_snapshot,
    )

    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        status_filter = _parse_status_filters(statuses)
        runs = ctx.repository.list_sotappr_runs(limit=limit, project_id=project_uuid)
        selected_runs = _filter_runs_by_status(runs, status_filter)
        if not selected_runs:
            raise click.ClickException(
                "No runs matched the requested benchmark filter."
            )
        run_metrics = collect_run_metrics_from_repository(
            repository=ctx.repository,
            runs=selected_runs,
        )
        snapshot = make_frozen_snapshot(
            run_metrics=run_metrics,
            project_id=str(project_uuid) if project_uuid else None,
            tolerances=BenchmarkTolerances(),
        )
        snapshot["status_filter"] = sorted(status_filter)
        snapshot["source_limit"] = limit
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
        _echo_json(
            {
                "frozen_snapshot_path": str(out_path),
                "run_count": len(run_metrics),
                "aggregate": snapshot.get("aggregate", {}),
                "status_filter": sorted(status_filter),
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-benchmark-gate")
@click.option(
    "--baseline",
    "baseline_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to frozen benchmark snapshot JSON.",
)
@click.option("--project-id", required=False, default=None)
@click.option("--limit", required=False, default=None, type=int)
@click.option(
    "--statuses",
    required=False,
    default=None,
    help="Optional comma-separated statuses (overrides baseline filter).",
)
def sotappr_benchmark_gate(
    baseline_path: Path,
    project_id: str | None,
    limit: int | None,
    statuses: str | None,
) -> None:
    """Evaluate current run metrics against a frozen benchmark baseline."""
    from src.sotappr.benchmark import (
        BenchmarkTolerances,
        collect_run_metrics_from_repository,
        compare_to_baseline,
        summarize_benchmark_runs,
    )

    raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_run_ids = raw.get("run_ids") or []
    default_limit = len(baseline_run_ids) if baseline_run_ids else 20
    effective_limit = int(limit or default_limit)
    if effective_limit <= 0:
        raise click.ClickException("Benchmark limit must be > 0.")

    status_filter = (
        _parse_status_filters(statuses)
        if statuses is not None
        else _parse_status_filters(",".join(raw.get("status_filter") or []))
    )
    tolerances = BenchmarkTolerances.from_payload(raw.get("tolerances"))

    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        runs = ctx.repository.list_sotappr_runs(limit=effective_limit, project_id=project_uuid)
        selected_runs = _filter_runs_by_status(runs, status_filter)
        if not selected_runs:
            raise click.ClickException("No runs available for benchmark gate evaluation.")
        run_metrics = collect_run_metrics_from_repository(
            repository=ctx.repository,
            runs=selected_runs,
        )
        current_aggregate = summarize_benchmark_runs(run_metrics)
        result = compare_to_baseline(
            baseline=raw,
            current_aggregate=current_aggregate,
            tolerances=tolerances,
        )
        payload = {
            "baseline_path": str(baseline_path),
            "status_filter": sorted(status_filter),
            "evaluated_run_count": len(run_metrics),
            "gate_passed": bool(result.get("passed")),
            "checks": result.get("checks", []),
            "baseline_aggregate": result.get("baseline_aggregate", {}),
            "current_aggregate": result.get("current_aggregate", {}),
        }
        _echo_json(payload)
        if not payload["gate_passed"]:
            raise click.ClickException("Benchmark regression gate failed.")
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-portfolio")
@click.option("--limit", required=False, default=50, type=int, show_default=True)
def sotappr_portfolio(limit: int) -> None:
    """List project portfolio and throughput snapshots."""
    ctx = _open_repo_context()
    try:
        rows = ctx.repository.get_portfolio_summary(limit=limit)
        payload = [
            {
                "project_id": str(row["id"]),
                "name": row.get("name"),
                "repo_path": row.get("repo_path"),
                "total_tasks": int(row.get("total_tasks") or 0),
                "pending_tasks": int(row.get("pending_tasks") or 0),
                "done_tasks": int(row.get("done_tasks") or 0),
                "stuck_tasks": int(row.get("stuck_tasks") or 0),
                "last_run_at": _iso(row.get("last_run_at")),
            }
            for row in rows
        ]
        _echo_json({"projects": payload, "count": len(payload)})
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-replay-search")
@click.option("--query", required=True, help="Free-text query over experience replay payloads.")
@click.option("--project-id", required=False, default=None)
@click.option("--limit", required=False, default=20, type=int, show_default=True)
def sotappr_replay_search(query: str, project_id: str | None, limit: int) -> None:
    """Search SOTAppR experience replay artifacts."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        rows = ctx.repository.search_sotappr_replay(query=query, limit=limit)
        if project_uuid is not None:
            rows = [r for r in rows if str(r.get("project_id")) == str(project_uuid)]

        payload = []
        for row in rows:
            item = row.get("payload") or {}
            payload.append(
                {
                    "artifact_id": str(row.get("id")),
                    "run_id": str(row.get("run_id")),
                    "project_id": str(row.get("project_id")),
                    "created_at": _iso(row.get("created_at")),
                    "context": item.get("context"),
                    "transferable_insight": item.get("transferable_insight"),
                }
            )

        _echo_json({"matches": payload, "count": len(payload)})
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-autotune")
@click.option("--project-id", required=False, default=None)
def sotappr_autotune(project_id: str | None) -> None:
    """Generate tuning recommendations from recent run/task telemetry."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        status_counts = ctx.repository.get_task_status_summary(project_id=project_uuid)
        errors = ctx.repository.get_hypothesis_error_stats(project_id=project_uuid)
        review_queue = ctx.repository.list_review_queue(limit=50, project_id=project_uuid)

        done = int(status_counts.get("DONE", 0))
        pending = int(status_counts.get("PENDING", 0))
        stuck = int(status_counts.get("STUCK", 0))
        reviewing = int(status_counts.get("REVIEWING", 0))
        total = sum(int(v) for v in status_counts.values())

        recommendations: list[str] = []
        if stuck > 0:
            recommendations.append(
                "Break backlog actions into smaller units and lower per-task "
                "complexity to reduce STUCK tasks."
            )
        if pending > done and pending >= 5:
            recommendations.append(
                "Increase `max_iterations` during `sotappr-execute` or split "
                "portfolio into parallel runs."
            )
        if reviewing > 0 and ctx.config.sotappr.require_human_review_before_done:
            recommendations.append(
                "Set up a regular `sotappr-approve-task` cadence to prevent review-queue buildup."
            )
        if len(review_queue) >= 10:
            recommendations.append(
                "Review queue is large; consider temporary auto-approval mode "
                "for low-risk maintenance tasks."
            )

        if errors:
            top_error = str(errors[0].get("error_signature") or "").lower()
            if "timeout" in top_error:
                recommendations.append(
                    "Timeout errors dominate; increase test timeout and reduce per-attempt scope."
                )
            if "import" in top_error or "module" in top_error:
                recommendations.append(
                    "Dependency/import failures dominate; tighten dependency "
                    "tribunal and lockfile verification."
                )
            if "assert" in top_error or "test" in top_error:
                recommendations.append(
                    "Assertion-heavy failures suggest contract drift; regenerate "
                    "Phase 1 contracts and acceptance criteria."
                )

        if not recommendations:
            recommendations.append("No urgent tuning changes detected from current telemetry.")

        _echo_json(
            {
                "project_id": str(project_uuid) if project_uuid else None,
                "task_status_counts": status_counts,
                "total_tasks": total,
                "recommendations": recommendations,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-drift-check")
@click.option("--project-id", required=False, default=None)
@click.option("--threshold", required=False, default=None, type=float)
def sotappr_drift_check(project_id: str | None, threshold: float | None) -> None:
    """Compute drift score from operational signals and compare against threshold."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        status_counts = ctx.repository.get_task_status_summary(project_id=project_uuid)
        errors = ctx.repository.get_hypothesis_error_stats(project_id=project_uuid)

        total_tasks = sum(int(v) for v in status_counts.values())
        stuck = int(status_counts.get("STUCK", 0))
        reviewing = int(status_counts.get("REVIEWING", 0))
        stuck_ratio = (stuck / total_tasks) if total_tasks else 0.0
        review_ratio = (reviewing / total_tasks) if total_tasks else 0.0

        total_error_count = sum(int(row.get("cnt") or 0) for row in errors)
        top_error_count = int(errors[0].get("cnt") or 0) if errors else 0
        error_concentration = (
            (top_error_count / total_error_count) if total_error_count else 0.0
        )

        drift_score = round(
            (0.50 * stuck_ratio) + (0.30 * review_ratio) + (0.20 * error_concentration),
            4,
        )
        threshold_value = (
            threshold if threshold is not None else ctx.config.sentinel.drift_threshold
        )

        _echo_json(
            {
                "project_id": str(project_uuid) if project_uuid else None,
                "timestamp": datetime.now(UTC).isoformat(),
                "drift_score": drift_score,
                "threshold": threshold_value,
                "drifted": drift_score >= threshold_value,
                "components": {
                    "stuck_ratio": round(stuck_ratio, 4),
                    "review_ratio": round(review_ratio, 4),
                    "error_concentration": round(error_concentration, 4),
                },
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("sotappr-chaos-drill")
@click.option("--run-id", required=True, help="Run ID to extract Phase 5 playbooks from.")
def sotappr_chaos_drill(run_id: str) -> None:
    """Generate a drill checklist from Phase 5 chaos playbooks."""
    run_uuid = _parse_uuid(run_id, "run_id")
    ctx = _open_repo_context()
    try:
        artifacts = ctx.repository.list_sotappr_artifacts(run_id=run_uuid, phase=5, limit=50)
        phase5 = next((row for row in artifacts if row.get("artifact_type") == "phase5"), None)
        if phase5 is None:
            raise click.ClickException(
                f"No Phase 5 artifact found for run {run_id}."
            )

        payload = phase5.get("payload") or {}
        playbooks = payload.get("chaos_playbooks") or []
        drills = [
            {
                "scenario": row.get("scenario"),
                "detection": row.get("detection"),
                "inject": f"Simulate: {row.get('scenario')}",
                "expected_auto_response": row.get("automated_response"),
                "manual_escalation": row.get("manual_escalation"),
                "recovery": row.get("recovery"),
                "post_mortem_hook": row.get("post_mortem_hook"),
            }
            for row in playbooks
        ]

        _echo_json(
            {
                "run_id": run_id,
                "drill_count": len(drills),
                "drills": drills,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("packet-evaluate-formula")
@click.option(
    "--packet-in",
    "packet_in",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input APC/1.0 packet JSON file.",
)
@click.option(
    "--out-dir",
    "out_dir",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("artifacts/packets"),
    show_default=True,
    help="Directory for emitted evaluation artifacts and follow-up packets.",
)
@click.option(
    "--mc-runs",
    required=False,
    type=int,
    default=200,
    show_default=True,
    help="Monte Carlo run count.",
)
@click.option(
    "--seed",
    required=False,
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--param-noise-std",
    required=False,
    type=float,
    default=0.12,
    show_default=True,
    help="Multiplicative parameter noise standard deviation.",
)
@click.option(
    "--init-jitter-ratio",
    required=False,
    type=float,
    default=0.10,
    show_default=True,
    help="Initial-condition multiplicative jitter ratio.",
)
@click.option(
    "--json-out",
    is_flag=True,
    default=False,
    help="Echo emitted packet bundle JSON to stdout.",
)
def packet_evaluate_formula(
    packet_in: Path,
    out_dir: Path,
    mc_runs: int,
    seed: int,
    param_noise_std: float,
    init_jitter_ratio: float,
    json_out: bool,
) -> None:
    """Validate and evaluate formula bundle packets, then emit EVAL/ARTIFACT packets."""
    from src.core.models import AgentPacket, PacketActor, PacketRecipient, PacketType, RunPhase
    from src.protocol.formula_evaluator import evaluate_formula_packet, evaluation_artifact_hash
    from src.protocol.validation import PacketValidationError, validate_agent_packet

    raw = _load_spec(packet_in)
    try:
        packet = validate_agent_packet(raw)
    except PacketValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    result = evaluate_formula_packet(
        packet=packet,
        mc_runs=mc_runs,
        seed=seed,
        param_noise_std=param_noise_std,
        init_jitter_ratio=init_jitter_ratio,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    eval_artifact_path = out_dir / f"formula_eval_{packet.packet_id}_{timestamp}.json"
    eval_artifact_path.write_text(
        result.artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )

    artifact_hash = evaluation_artifact_hash(result)
    sender = PacketActor(
        agent_id="formula-evaluator",
        role="validator",
        swarm_id="swarm-evaluator",
        capabilities=["formula-eval", "monte-carlo", "stability-analysis"],
    )
    recipient = PacketRecipient(
        agent_id=packet.sender.agent_id,
        role=packet.sender.role,
        swarm_id=packet.sender.swarm_id,
    )

    artifact_packet = AgentPacket(
        packet_type=PacketType.ARTIFACT,
        channel=packet.channel,
        run_phase=RunPhase.VALIDATION,
        sender=sender,
        recipients=[recipient],
        trace=packet.trace.model_copy(update={"parent_packet_id": packet.packet_id, "step": packet.trace.step + 1}),
        routing=packet.routing.model_copy(update={"requires_ack": False}),
        confidence=min(1.0, max(0.0, packet.confidence)),
        symbolic_keys=list(dict.fromkeys([*packet.symbolic_keys, "formula-eval", "artifact"])),
        payload={
            "artifact_id": f"formula-eval-{packet.packet_id}",
            "artifact_kind": "METRIC_REPORT",
            "artifact_uri": str(eval_artifact_path),
            "artifact_hash": artifact_hash,
            "language": "python",
            "adapter": "formula_bundle/1.0",
        },
    )

    eval_proof_bundle = packet.proof_bundle
    if eval_proof_bundle is None:
        eval_proof_bundle = {
            "evidence_refs": [str(eval_artifact_path)],
            "falsifiers": [],
            "gate_scores": result.metrics,
            "signatures": [],
        }
    else:
        eval_proof_bundle = eval_proof_bundle.model_dump(mode="json")
        eval_proof_bundle.setdefault("evidence_refs", [])
        eval_proof_bundle["evidence_refs"].append(str(eval_artifact_path))
        eval_proof_bundle["gate_scores"] = result.metrics

    eval_packet = AgentPacket(
        packet_type=PacketType.EVAL,
        channel=packet.channel,
        run_phase=RunPhase.VALIDATION,
        sender=sender,
        recipients=[recipient],
        trace=packet.trace.model_copy(update={"parent_packet_id": packet.packet_id, "step": packet.trace.step + 2}),
        routing=packet.routing.model_copy(update={"requires_ack": True}),
        confidence=min(1.0, max(0.0, packet.confidence)),
        symbolic_keys=list(dict.fromkeys([*packet.symbolic_keys, "formula-eval", "gates"])),
        proof_bundle=eval_proof_bundle,
        payload={
            "metric_set": result.metrics,
            "thresholds": result.thresholds,
            "pass": result.passed,
            "notes": result.notes,
        },
    )

    bundle_path = out_dir / f"formula_eval_packets_{packet.packet_id}_{timestamp}.json"
    bundle_payload = {
        "input_packet_id": str(packet.packet_id),
        "artifact_path": str(eval_artifact_path),
        "artifact_packet": artifact_packet.model_dump(mode="json"),
        "eval_packet": eval_packet.model_dump(mode="json"),
    }
    bundle_path.write_text(json.dumps(bundle_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    click.echo(f"Wrote evaluation artifact: {eval_artifact_path}")
    click.echo(f"Wrote emitted packet bundle: {bundle_path}")
    if json_out:
        _echo_json(bundle_payload)


@cli.command("packet-transfer-arbitrate")
@click.option(
    "--packets-in",
    "packets_in",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON file containing TRANSFER packet list or {'packets': [...]} object.",
)
@click.option(
    "--top-k",
    required=False,
    default=2,
    show_default=True,
    type=int,
    help="Maximum number of transfer candidates to select.",
)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write arbitration JSON payload.",
)
def packet_transfer_arbitrate(
    packets_in: Path,
    top_k: int,
    out_path: Path | None,
) -> None:
    """Rank and select TRANSFER packets using APC runtime transfer policies."""
    from src.protocol.packet_runtime import PacketPolicyError, PacketRuntime
    from src.protocol.validation import PacketValidationError, validate_agent_packet

    text = packets_in.read_text(encoding="utf-8")
    if packets_in.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    raw_packets: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_packets = payload
    elif isinstance(payload, dict):
        raw = payload.get("packets")
        if not isinstance(raw, list):
            raise click.ClickException("Expected list payload or object containing key 'packets'.")
        raw_packets = raw
    else:
        raise click.ClickException("Invalid packet input payload type.")

    runtime = PacketRuntime()
    ranked: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_packets, start=1):
        if not isinstance(raw, dict):
            rejected.append({"index": idx, "reason": "packet must be an object"})
            continue
        try:
            packet = validate_agent_packet(raw)
        except PacketValidationError as exc:
            rejected.append(
                {
                    "index": idx,
                    "packet_id": raw.get("packet_id"),
                    "reason": f"validation_error: {str(exc)[:500]}",
                }
            )
            continue
        if packet.packet_type.value != "TRANSFER":
            rejected.append(
                {
                    "index": idx,
                    "packet_id": str(packet.packet_id),
                    "reason": f"packet_type_not_transfer:{packet.packet_type.value}",
                }
            )
            continue
        try:
            scored = runtime.rank_transfer_candidates([packet])
        except PacketPolicyError as exc:
            rejected.append(
                {
                    "index": idx,
                    "packet_id": str(packet.packet_id),
                    "reason": f"policy_reject: {str(exc)}",
                }
            )
            continue
        if scored:
            ranked.append(scored[0])

    ranked = sorted(ranked, key=lambda row: float(row.get("selection_score") or 0.0), reverse=True)
    selected = ranked[: max(0, int(top_k))]
    selected_ids = {str(row.get("packet_id")) for row in selected}
    not_selected = [row for row in ranked if str(row.get("packet_id")) not in selected_ids]

    arbitration_payload = {
        "input_count": len(raw_packets),
        "valid_transfer_count": len(ranked),
        "selected_count": len(selected),
        "top_k": int(top_k),
        "selected": selected,
        "not_selected": not_selected,
        "rejected": rejected,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(arbitration_payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        click.echo(f"Wrote transfer arbitration: {out_path}")
    _echo_json(arbitration_payload)


@cli.command("packet-trace")
@click.option("--run-id", required=False, default=None, help="Optional run UUID filter.")
@click.option("--task-id", required=False, default=None, help="Optional task UUID filter.")
@click.option("--packet-id", required=False, default=None, help="Optional packet UUID filter.")
@click.option("--packet-type", required=False, default=None, help="Optional packet type filter.")
@click.option("--run-phase", required=False, default=None, help="Optional run phase filter.")
@click.option("--trace-id", required=False, default=None, help="Optional trace_id filter.")
@click.option("--limit", required=False, default=200, show_default=True, type=int)
@click.option(
    "--include-events/--no-events",
    "include_events",
    default=True,
    show_default=True,
    help="Include packet state-machine event rows.",
)
def packet_trace(
    run_id: str | None,
    task_id: str | None,
    packet_id: str | None,
    packet_type: str | None,
    run_phase: str | None,
    trace_id: str | None,
    limit: int,
    include_events: bool,
) -> None:
    """Inspect APC packet traces from first-class packet tables (or artifact fallback)."""
    ctx = _open_repo_context()
    try:
        run_uuid = _parse_uuid(run_id, "run_id") if run_id else None
        task_uuid = _parse_uuid(task_id, "task_id") if task_id else None
        packet_uuid = _parse_uuid(packet_id, "packet_id") if packet_id else None
        packets: list[dict[str, Any]]
        events: list[dict[str, Any]] = []
        source = "agent_packets"

        if hasattr(ctx.repository, "list_agent_packets"):
            packets = ctx.repository.list_agent_packets(
                run_id=run_uuid,
                task_id=task_uuid,
                packet_id=packet_uuid,
                packet_type=packet_type,
                run_phase=run_phase,
                trace_id=trace_id,
                limit=limit,
            )
            if include_events and hasattr(ctx.repository, "list_packet_events"):
                events = ctx.repository.list_packet_events(
                    run_id=run_uuid,
                    task_id=task_uuid,
                    packet_id=packet_uuid,
                    limit=max(200, limit * 8),
                )
        else:
            source = "sotappr_artifacts"
            if run_uuid is None:
                raise click.ClickException(
                    "Repository does not expose list_agent_packets; provide --run-id for artifact fallback."
                )
            artifacts = ctx.repository.list_sotappr_artifacts(run_id=run_uuid, limit=max(200, limit * 8))
            packets = []
            for row in artifacts:
                if str(row.get("artifact_type")) != "agent_packet":
                    continue
                payload = _artifact_payload(row)
                packet = payload.get("packet")
                if not isinstance(packet, dict):
                    continue
                if packet_uuid and str(packet.get("packet_id")) != str(packet_uuid):
                    continue
                if packet_type and str(packet.get("packet_type")) != packet_type:
                    continue
                if run_phase and str(packet.get("run_phase")) != run_phase:
                    continue
                if trace_id and str(payload.get("trace_id")) != trace_id:
                    continue
                if task_uuid and str(payload.get("task_id")) != str(task_uuid):
                    continue
                packets.append(
                    {
                        "id": row.get("id"),
                        "run_id": row.get("run_id"),
                        "task_id": payload.get("task_id"),
                        "attempt_number": payload.get("attempt"),
                        "packet_id": packet.get("packet_id"),
                        "packet_type": packet.get("packet_type"),
                        "channel": packet.get("channel"),
                        "run_phase": packet.get("run_phase"),
                        "sender_role": ((packet.get("sender") or {}).get("role") if isinstance(packet.get("sender"), dict) else None),
                        "trace_id": payload.get("trace_id"),
                        "lifecycle_state": payload.get("lifecycle_state"),
                        "packet_json": packet,
                        "payload_json": packet.get("payload"),
                        "created_at": row.get("created_at"),
                    }
                )
                if len(packets) >= limit:
                    break

        packet_rows = []
        for row in packets:
            packet_json = row.get("packet_json") if isinstance(row.get("packet_json"), dict) else {}
            payload_json = row.get("payload_json") if isinstance(row.get("payload_json"), dict) else {}
            if not packet_json and isinstance(row.get("packet"), dict):
                packet_json = row.get("packet")
            packet_rows.append(
                {
                    "packet_id": str(row.get("packet_id")),
                    "task_id": str(row.get("task_id")) if row.get("task_id") is not None else None,
                    "run_id": str(row.get("run_id")) if row.get("run_id") is not None else None,
                    "attempt_number": int(row.get("attempt_number") or 0),
                    "packet_type": row.get("packet_type") or packet_json.get("packet_type"),
                    "channel": row.get("channel") or packet_json.get("channel"),
                    "run_phase": row.get("run_phase") or packet_json.get("run_phase"),
                    "sender_role": row.get("sender_role")
                    or ((packet_json.get("sender") or {}).get("role") if isinstance(packet_json.get("sender"), dict) else None),
                    "trace_id": row.get("trace_id"),
                    "lifecycle_state": row.get("lifecycle_state"),
                    "created_at": _iso(row.get("created_at")),
                    "payload_keys": sorted(payload_json.keys()) if payload_json else [],
                }
            )

        event_rows = []
        if include_events:
            for ev in events:
                event_rows.append(
                    {
                        "packet_id": str(ev.get("packet_id")),
                        "event": ev.get("event"),
                        "from_state": ev.get("from_state"),
                        "to_state": ev.get("to_state"),
                        "metadata": ev.get("metadata") if isinstance(ev.get("metadata"), dict) else {},
                        "occurred_at": _iso(ev.get("occurred_at")),
                    }
                )

        _echo_json(
            {
                "source": source,
                "count": len(packet_rows),
                "event_count": len(event_rows),
                "packets": packet_rows,
                "events": event_rows,
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("packet-lineage-summary")
@click.option("--project-id", required=False, default=None, help="Optional project UUID filter.")
@click.option("--run-id", required=False, default=None, help="Optional run UUID filter.")
@click.option("--protocol-id", required=False, default=None, help="Optional protocol ID filter.")
@click.option("--limit", required=False, default=100, show_default=True, type=int)
def packet_lineage_summary(
    project_id: str | None,
    run_id: str | None,
    protocol_id: str | None,
    limit: int,
) -> None:
    """Summarize cross-run protocol propagation using packet lineage records."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id") if project_id else None
        run_uuid = _parse_uuid(run_id, "run_id") if run_id else None
        source = "packet_lineage"
        summary_rows: list[dict[str, Any]]
        propagation_rows: list[dict[str, Any]]
        quality_rows: list[dict[str, Any]] = []
        correlation_rows: list[dict[str, Any]] = []
        effectiveness_rows: list[dict[str, Any]] = []
        arbitration_decisions: list[dict[str, Any]] = []

        if hasattr(ctx.repository, "get_packet_lineage_summary") and hasattr(ctx.repository, "list_protocol_propagation"):
            summary_rows = ctx.repository.get_packet_lineage_summary(
                project_id=project_uuid,
                run_id=run_uuid,
                limit=limit,
            )
            propagation_rows = ctx.repository.list_protocol_propagation(
                protocol_id=protocol_id,
                project_id=project_uuid,
                run_id=run_uuid,
                limit=max(200, limit * 2),
            )
            if hasattr(ctx.repository, "get_transfer_quality_rollup"):
                quality_rows = ctx.repository.get_transfer_quality_rollup(
                    project_id=project_uuid,
                    run_id=run_uuid,
                )
            if hasattr(ctx.repository, "get_transfer_eval_correlation"):
                correlation_rows = ctx.repository.get_transfer_eval_correlation(
                    project_id=project_uuid,
                    run_id=run_uuid,
                )
            if hasattr(ctx.repository, "get_protocol_effectiveness"):
                effectiveness_rows = ctx.repository.get_protocol_effectiveness(
                    project_id=project_uuid,
                    run_id=run_uuid,
                    limit=max(20, limit),
                )
        else:
            source = "sotappr_artifacts"
            if run_uuid is None:
                raise click.ClickException(
                    "Repository does not expose lineage tables; provide --run-id for artifact fallback."
                )
            artifacts = ctx.repository.list_sotappr_artifacts(run_id=run_uuid, limit=max(200, limit * 4))
            propagation_rows = []
            for row in artifacts:
                if str(row.get("artifact_type")) != "agent_packet":
                    continue
                payload = _artifact_payload(row)
                packet = payload.get("packet")
                if not isinstance(packet, dict):
                    continue
                lineage = packet.get("lineage")
                if not isinstance(lineage, dict):
                    continue
                pid = str(lineage.get("protocol_id") or "")
                if protocol_id and pid != protocol_id:
                    continue
                propagation_rows.append(
                    {
                        "protocol_id": pid or "(none)",
                        "parent_protocol_ids": lineage.get("parent_protocol_ids") or [],
                        "ancestor_swarms": lineage.get("ancestor_swarms") or [],
                        "cross_use_case": bool(lineage.get("cross_use_case", False)),
                        "transfer_mode": lineage.get("transfer_mode"),
                        "packet_id": packet.get("packet_id"),
                        "packet_type": packet.get("packet_type"),
                        "run_phase": packet.get("run_phase"),
                        "run_id": row.get("run_id"),
                        "project_id": None,
                        "task_id": payload.get("task_id"),
                        "trace_id": payload.get("trace_id"),
                        "created_at": row.get("created_at"),
                    }
                )

            protocol_index: dict[str, dict[str, Any]] = {}
            for row in propagation_rows:
                pid = str(row.get("protocol_id") or "(none)")
                bucket = protocol_index.setdefault(
                    pid,
                    {
                        "protocol_id": pid,
                        "transfer_count": 0,
                        "run_ids": set(),
                        "task_ids": set(),
                        "cross_use_case_count": 0,
                        "last_seen_at": None,
                    },
                )
                bucket["transfer_count"] += 1
                if row.get("run_id") is not None:
                    bucket["run_ids"].add(str(row.get("run_id")))
                if row.get("task_id") is not None:
                    bucket["task_ids"].add(str(row.get("task_id")))
                if row.get("cross_use_case"):
                    bucket["cross_use_case_count"] += 1
            summary_rows = []
            for bucket in protocol_index.values():
                summary_rows.append(
                    {
                        "protocol_id": bucket["protocol_id"],
                        "transfer_count": bucket["transfer_count"],
                        "run_count": len(bucket["run_ids"]),
                        "task_count": len(bucket["task_ids"]),
                        "cross_use_case_count": bucket["cross_use_case_count"],
                        "last_seen_at": bucket["last_seen_at"],
                    }
                )
            summary_rows = sorted(
                summary_rows,
                key=lambda row: int(row.get("transfer_count") or 0),
                reverse=True,
            )[:limit]
            quality_index: dict[str, dict[str, Any]] = {}
            for row in propagation_rows:
                mode = str(row.get("transfer_mode") or "unknown")
                bucket = quality_index.setdefault(
                    mode,
                    {
                        "transfer_mode": mode,
                        "total_count": 0,
                        "accepted_count": 0,
                        "cross_use_case_count": 0,
                    },
                )
                bucket["total_count"] += 1
                if row.get("cross_use_case"):
                    bucket["cross_use_case_count"] += 1
            quality_rows = list(quality_index.values())

        if protocol_id:
            summary_rows = [row for row in summary_rows if str(row.get("protocol_id")) == protocol_id]

        arbitration_decisions = _collect_transfer_arbitration_decisions(
            repository=ctx.repository,
            run_id=run_uuid,
            project_id=project_uuid,
            limit=max(50, limit),
        )
        arbitration_summary = _summarize_transfer_arbitration(arbitration_decisions)

        transfer_count = sum(int(row.get("transfer_count") or 0) for row in summary_rows)
        cross_use_case_count = sum(int(row.get("cross_use_case_count") or 0) for row in summary_rows)
        runs_seen = {
            str(row.get("run_id"))
            for row in propagation_rows
            if row.get("run_id") is not None
        }
        mode_counts: dict[str, int] = {}
        for row in propagation_rows:
            mode = str(row.get("transfer_mode") or "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        transfer_quality_payload = []
        mode_acceptance_rates: dict[str, float] = {}
        for row in quality_rows:
            total = int(row.get("total_count") or 0)
            accepted = int(row.get("accepted_count") or 0)
            rate = (accepted / total) if total > 0 else 0.0
            mode = str(row.get("transfer_mode") or "unknown")
            mode_acceptance_rates[mode] = round(rate, 6)
            transfer_quality_payload.append(
                {
                    "transfer_mode": mode,
                    "total_count": total,
                    "accepted_count": accepted,
                    "acceptance_rate": round(rate, 6),
                    "cross_use_case_count": int(row.get("cross_use_case_count") or 0),
                    "avg_sender_score": (
                        round(float(row.get("avg_sender_score")), 6)
                        if row.get("avg_sender_score") is not None
                        else None
                    ),
                    "avg_receiver_score": (
                        round(float(row.get("avg_receiver_score")), 6)
                        if row.get("avg_receiver_score") is not None
                        else None
                    ),
                }
            )
        drift_correlation_payload = [
            {
                "has_cross_use_case_transfer": bool(row.get("has_cross_use_case_transfer", False)),
                "task_count": int(row.get("task_count") or 0),
                "avg_outcome": (
                    round(float(row.get("avg_outcome")), 6)
                    if row.get("avg_outcome") is not None
                    else None
                ),
                "avg_drift_risk": (
                    round(float(row.get("avg_drift_risk")), 6)
                    if row.get("avg_drift_risk") is not None
                    else None
                ),
            }
            for row in correlation_rows
        ]
        baseline_outcome = 0.0
        baseline_drift = 0.0
        baseline_weight = 0
        for row in effectiveness_rows:
            task_count = int(row.get("task_count") or 0)
            outcome = row.get("avg_outcome")
            drift = row.get("avg_drift_risk")
            if task_count <= 0 or outcome is None or drift is None:
                continue
            baseline_weight += task_count
            baseline_outcome += float(outcome) * task_count
            baseline_drift += float(drift) * task_count
        if baseline_weight > 0:
            baseline_outcome /= baseline_weight
            baseline_drift /= baseline_weight

        effectiveness_payload = []
        for row in effectiveness_rows:
            transfer_count = int(row.get("transfer_count") or 0)
            accepted_count = int(row.get("accepted_count") or 0)
            acceptance_rate = (accepted_count / transfer_count) if transfer_count > 0 else 0.0
            avg_outcome = float(row.get("avg_outcome")) if row.get("avg_outcome") is not None else None
            avg_drift = float(row.get("avg_drift_risk")) if row.get("avg_drift_risk") is not None else None
            outcome_lift = (avg_outcome - baseline_outcome) if avg_outcome is not None and baseline_weight > 0 else None
            drift_delta = (avg_drift - baseline_drift) if avg_drift is not None and baseline_weight > 0 else None
            normalized_outcome = max(0.0, min(1.0, avg_outcome or 0.0))
            normalized_drift = max(0.0, min(1.0, avg_drift or 0.0))
            effectiveness_score = (0.5 * acceptance_rate) + (0.35 * normalized_outcome) + (0.15 * (1.0 - normalized_drift))
            effectiveness_payload.append(
                {
                    "protocol_id": row.get("protocol_id"),
                    "transfer_count": transfer_count,
                    "accepted_count": accepted_count,
                    "acceptance_rate": round(acceptance_rate, 6),
                    "run_count": int(row.get("run_count") or 0),
                    "task_count": int(row.get("task_count") or 0),
                    "avg_outcome": round(avg_outcome, 6) if avg_outcome is not None else None,
                    "avg_drift_risk": round(avg_drift, 6) if avg_drift is not None else None,
                    "outcome_lift": round(outcome_lift, 6) if outcome_lift is not None else None,
                    "drift_risk_delta": round(drift_delta, 6) if drift_delta is not None else None,
                    "effectiveness_score": round(effectiveness_score, 6),
                }
            )
        effectiveness_payload = sorted(
            effectiveness_payload,
            key=lambda row: float(row.get("effectiveness_score") or 0.0),
            reverse=True,
        )[:limit]

        protocols_payload = [
            {
                "protocol_id": row.get("protocol_id"),
                "transfer_count": int(row.get("transfer_count") or 0),
                "run_count": int(row.get("run_count") or 0),
                "task_count": int(row.get("task_count") or 0),
                "cross_use_case_count": int(row.get("cross_use_case_count") or 0),
                "last_seen_at": _iso(row.get("last_seen_at")),
            }
            for row in summary_rows
        ]
        propagation_payload = [
            {
                "protocol_id": row.get("protocol_id"),
                "packet_id": str(row.get("packet_id")) if row.get("packet_id") is not None else None,
                "packet_type": row.get("packet_type"),
                "run_phase": row.get("run_phase"),
                "run_id": str(row.get("run_id")) if row.get("run_id") is not None else None,
                "project_id": str(row.get("project_id")) if row.get("project_id") is not None else None,
                "task_id": str(row.get("task_id")) if row.get("task_id") is not None else None,
                "trace_id": row.get("trace_id"),
                "transfer_mode": row.get("transfer_mode"),
                "cross_use_case": bool(row.get("cross_use_case", False)),
                "parent_protocol_ids": row.get("parent_protocol_ids") or [],
                "ancestor_swarms": row.get("ancestor_swarms") or [],
                "created_at": _iso(row.get("created_at")),
            }
            for row in propagation_rows[:limit]
        ]

        _echo_json(
            {
                "source": source,
                "filters": {
                    "project_id": str(project_uuid) if project_uuid else None,
                    "run_id": str(run_uuid) if run_uuid else None,
                    "protocol_id": protocol_id,
                    "limit": int(limit),
                },
                "summary": {
                    "protocol_count": len(protocols_payload),
                    "transfer_count": transfer_count,
                    "cross_use_case_count": cross_use_case_count,
                    "run_count": len(runs_seen),
                    "mode_counts": mode_counts,
                    "mode_acceptance_rates": mode_acceptance_rates,
                    "cross_run_protocols": sum(1 for row in protocols_payload if row["run_count"] > 1),
                    "baseline_outcome": round(baseline_outcome, 6) if baseline_weight > 0 else None,
                    "baseline_drift_risk": round(baseline_drift, 6) if baseline_weight > 0 else None,
                    "arbitration_decision_count": arbitration_summary["decision_count"],
                    "arbitration_selected_total": arbitration_summary["selected_total"],
                    "arbitration_blocked_total": arbitration_summary["blocked_total"],
                    "arbitration_rejected_total": arbitration_summary["rejected_total"],
                },
                "protocols": protocols_payload,
                "protocol_effectiveness": effectiveness_payload,
                "transfer_quality": transfer_quality_payload,
                "drift_correlation": drift_correlation_payload,
                "propagation": propagation_payload,
                "transfer_arbitration": {
                    "summary": arbitration_summary,
                    "decisions": arbitration_decisions[:limit],
                },
            }
        )
    finally:
        _close_repo_context(ctx)


@cli.command("packet-replay")
@click.option("--run-id", required=True, help="Run UUID to replay packet timeline.")
@click.option("--task-id", required=False, default=None, help="Optional task UUID filter.")
@click.option("--limit", required=False, default=500, show_default=True, type=int)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to export replay JSON payload.",
)
def packet_replay(run_id: str, task_id: str | None, limit: int, out_path: Path | None) -> None:
    """Reconstruct packet state-machine timeline for a run/task."""
    ctx = _open_repo_context()
    try:
        run_uuid = _parse_uuid(run_id, "run_id")
        task_uuid = _parse_uuid(task_id, "task_id") if task_id else None
        source = "packet_events"
        timeline_rows: list[dict[str, Any]] = []

        if hasattr(ctx.repository, "get_packet_timeline"):
            timeline_rows = ctx.repository.get_packet_timeline(
                run_id=run_uuid,
                task_id=task_uuid,
                limit=limit,
            )
        else:
            source = "sotappr_artifacts"
            artifacts = ctx.repository.list_sotappr_artifacts(run_id=run_uuid, limit=max(200, limit * 4))
            for row in artifacts:
                if str(row.get("artifact_type")) != "agent_packet":
                    continue
                payload = _artifact_payload(row)
                packet = payload.get("packet")
                if not isinstance(packet, dict):
                    continue
                if task_uuid and str(payload.get("task_id")) != str(task_uuid):
                    continue
                history = payload.get("lifecycle_history")
                if not isinstance(history, list):
                    history = []
                for event in history:
                    if not isinstance(event, dict):
                        continue
                    timeline_rows.append(
                        {
                            "packet_id": packet.get("packet_id"),
                            "packet_type": packet.get("packet_type"),
                            "run_phase": packet.get("run_phase"),
                            "lifecycle_state": payload.get("lifecycle_state"),
                            "trace_id": payload.get("trace_id"),
                            "packet_created_at": row.get("created_at"),
                            "event": event.get("event"),
                            "from_state": event.get("from"),
                            "to_state": event.get("to"),
                            "metadata": event.get("metadata") if isinstance(event.get("metadata"), dict) else {},
                            "occurred_at": event.get("at"),
                        }
                    )

        packets: dict[str, dict[str, Any]] = {}
        event_stream: list[dict[str, Any]] = []
        for row in timeline_rows:
            packet_id = str(row.get("packet_id")) if row.get("packet_id") is not None else None
            if not packet_id:
                continue
            packet_entry = packets.setdefault(
                packet_id,
                {
                    "packet_id": packet_id,
                    "packet_type": row.get("packet_type"),
                    "run_phase": row.get("run_phase"),
                    "trace_id": row.get("trace_id"),
                    "created_at": _iso(row.get("packet_created_at")),
                    "final_state": row.get("lifecycle_state"),
                    "transitions": [],
                },
            )
            event_name = row.get("event")
            if event_name is None:
                continue
            transition = {
                "event": event_name,
                "from_state": row.get("from_state"),
                "to_state": row.get("to_state"),
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                "occurred_at": _iso(row.get("occurred_at")),
            }
            packet_entry["transitions"].append(transition)
            event_stream.append(
                {
                    "packet_id": packet_id,
                    "packet_type": packet_entry.get("packet_type"),
                    "run_phase": packet_entry.get("run_phase"),
                    **transition,
                }
            )

        packets_payload = list(packets.values())
        for packet in packets_payload:
            packet["transitions"] = sorted(
                packet["transitions"],
                key=lambda item: item.get("occurred_at") or "",
            )
            if packet["transitions"]:
                packet["final_state"] = packet["transitions"][-1].get("to_state") or packet["final_state"]
        event_stream = sorted(event_stream, key=lambda item: item.get("occurred_at") or "")

        replay_payload = {
            "source": source,
            "run_id": str(run_uuid),
            "task_id": str(task_uuid) if task_uuid else None,
            "packet_count": len(packets_payload),
            "event_count": len(event_stream),
            "packets": packets_payload,
            "event_stream": event_stream[:limit],
        }
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(replay_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            click.echo(f"Wrote packet replay: {out_path}")

        _echo_json(replay_payload)
    finally:
        _close_repo_context(ctx)


@dataclass
class _RepoContext:
    config: Any
    repository: Any
    db_engine: Any


def _open_repo_context(config_dir: Path | None = None, env: str | None = None) -> _RepoContext:
    from src.core.config import load_config
    from src.db.engine import DatabaseEngine
    from src.db.repository import Repository

    config = load_config(config_dir=config_dir, env=env)
    db_engine = DatabaseEngine(config.database)
    db_engine.initialize_schema()
    repository = Repository(db_engine)
    return _RepoContext(config=config, repository=repository, db_engine=db_engine)


def _close_repo_context(ctx: _RepoContext) -> None:
    ctx.db_engine.close()


def _load_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise click.ClickException("Spec must be a JSON/YAML object.")
    return data


def _load_component_factory():
    from src.core.factory import ComponentFactory

    return ComponentFactory


def _load_sotappr_executor():
    from src.sotappr.executor import SOTAppRExecutor

    return SOTAppRExecutor


def _load_openrouter_model_catalog():
    from src.llm.model_catalog import OpenRouterModelCatalog

    return OpenRouterModelCatalog


def _parse_uuid(raw: str | None, field_name: str) -> UUID:
    if raw is None:
        raise click.ClickException(f"Missing required UUID value for {field_name}")
    try:
        return UUID(str(raw))
    except ValueError as exc:
        raise click.ClickException(f"Invalid UUID for {field_name}: {raw}") from exc


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _echo_json(payload: dict[str, Any]) -> None:
    click.echo(json.dumps(payload, indent=2, ensure_ascii=True))


def _new_session_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"session_{stamp}_{uuid4().hex[:8]}"


def _runtime_budget_contract(
    *,
    max_hours: float | None,
    max_cost_usd: float | None,
    estimated_cost_per_1k_tokens_usd: float,
) -> dict[str, Any] | None:
    if max_hours is None and max_cost_usd is None:
        return None
    return {
        "max_hours": max_hours,
        "max_cost_usd": max_cost_usd,
        "estimated_cost_per_1k_tokens_usd": estimated_cost_per_1k_tokens_usd,
    }


def _run_alignment_gate(alignment: Any) -> Any:
    while True:
        click.echo(f"  UX expectations: {len(alignment.ux_expectations)}")
        click.echo(f"  Features:        {len(alignment.features)}")
        click.echo(f"  Methodology:     {alignment.methodology}")
        click.echo(f"  Acceptance:      {len(alignment.acceptance_criteria)}")
        click.echo(f"  Risks:           {len(alignment.risks)}")
        if click.confirm("Approve alignment spec?", default=True):
            alignment.approved = True
            return alignment

        edit_target = click.prompt(
            "Edit section",
            type=click.Choice(["ux", "features", "methodology", "acceptance", "risks", "abort"]),
            default="features",
            show_default=True,
        )
        if edit_target == "abort":
            raise click.Abort()
        if edit_target == "ux":
            alignment.ux_expectations = _prompt_csv_items(
                label="UX expectations (comma-separated)",
                current=alignment.ux_expectations,
            )
            continue
        if edit_target == "features":
            alignment.features = _prompt_csv_items(
                label="Features (comma-separated)",
                current=alignment.features,
            )
            continue
        if edit_target == "methodology":
            alignment.methodology = click.prompt(
                "Methodology",
                default=alignment.methodology,
                show_default=True,
            )
            continue
        if edit_target == "acceptance":
            alignment.acceptance_criteria = _prompt_csv_items(
                label="Acceptance criteria (comma-separated)",
                current=alignment.acceptance_criteria,
            )
            continue
        alignment.risks = _prompt_csv_items(
            label="Risks (comma-separated)",
            current=alignment.risks,
        )


def _run_budget_gate(budget: Any) -> Any:
    while True:
        budget.max_hours = click.prompt(
            "Set max hours",
            type=float,
            default=budget.max_hours,
            show_default=True,
        )
        budget.max_cost_usd = click.prompt(
            "Set max cost USD",
            type=float,
            default=budget.max_cost_usd,
            show_default=True,
        )
        if click.confirm("Approve budget contract?", default=True):
            budget.approved = True
            return budget
        if not click.confirm("Edit budget again?", default=True):
            raise click.Abort()


def _run_knowledge_gate(
    *,
    local_paths: list[Path],
    urls: list[str],
    queries: list[str],
) -> tuple[list[Path], list[str], list[str]]:
    local_raw = _prompt_csv_items(
        label="Knowledge local paths (comma-separated)",
        current=[str(p) for p in local_paths],
    )
    next_local: list[Path] = []
    for raw in local_raw:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise click.ClickException(f"Knowledge path does not exist: {path}")
        next_local.append(path)

    next_urls = _prompt_csv_items(
        label="Knowledge URLs (comma-separated)",
        current=urls,
    )
    next_queries = _prompt_csv_items(
        label="Knowledge queries (comma-separated)",
        current=queries,
    )
    return next_local, next_urls, next_queries


def _prompt_csv_items(label: str, current: list[str]) -> list[str]:
    current_raw = ", ".join(item for item in current if item)
    raw = click.prompt(
        label,
        default=current_raw,
        show_default=bool(current_raw),
    )
    return _parse_csv_items(raw)


def _parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_budget_pause(stop_reason: Any, last_error: Any) -> bool:
    reason = str(stop_reason or "").lower()
    err = str(last_error or "").lower()
    tokens = ("budget_cost_exceeded", "budget_time_exceeded")
    return any(token in reason for token in tokens) or any(token in err for token in tokens)


def _parse_status_filters(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    statuses = {item.strip().lower() for item in str(raw).split(",") if item.strip()}
    return statuses


def _filter_runs_by_status(runs: list[dict[str, Any]], statuses: set[str]) -> list[dict[str, Any]]:
    if not statuses:
        return list(runs)
    return [row for row in runs if str(row.get("status") or "").lower() in statuses]


def _artifact_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    if isinstance(payload, dict):
        return payload
    return {}


def _latest_model_failover_state(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    for row in artifacts:
        if str(row.get("artifact_type")) != "model_failover_state":
            continue
        payload = _artifact_payload(row)
        state = payload.get("state")
        if isinstance(state, dict):
            return state
    return {}


def _retry_cause_counts(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in artifacts:
        if str(row.get("artifact_type")) != "retry_telemetry":
            continue
        payload = _artifact_payload(row)
        cause = str(payload.get("retry_cause") or "unknown")
        counts[cause] = counts.get(cause, 0) + 1
    return counts


def _empty_transfer_arbitration_summary() -> dict[str, Any]:
    return {
        "decision_count": 0,
        "candidate_total": 0,
        "selected_total": 0,
        "blocked_total": 0,
        "rejected_total": 0,
        "avg_selected_per_decision": 0.0,
        "selection_rate": 0.0,
        "block_reason_counts": {},
        "reject_reason_counts": {},
    }


def _collect_transfer_arbitration_decisions(
    *,
    repository: Any,
    run_id: UUID | None = None,
    project_id: UUID | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if not hasattr(repository, "list_sotappr_artifacts"):
        return []

    run_ids: list[UUID] = []
    if run_id is not None:
        run_ids = [run_id]
    elif project_id is not None and hasattr(repository, "list_sotappr_runs"):
        try:
            runs = repository.list_sotappr_runs(limit=max(20, limit), project_id=project_id)
        except Exception:
            runs = []
        for row in runs:
            raw = row.get("id")
            if raw is None:
                continue
            try:
                run_ids.append(UUID(str(raw)))
            except (TypeError, ValueError):
                continue
    else:
        return []

    decisions: list[dict[str, Any]] = []
    for rid in run_ids:
        try:
            artifacts = repository.list_sotappr_artifacts(
                run_id=rid,
                limit=max(200, limit * 3),
            )
        except Exception:
            continue
        for row in artifacts:
            if str(row.get("artifact_type")) != "transfer_arbitration_decision":
                continue
            payload = _artifact_payload(row)
            blocked_raw = payload.get("blocked") if isinstance(payload.get("blocked"), list) else []
            rejected_raw = payload.get("rejected") if isinstance(payload.get("rejected"), list) else []
            decisions.append(
                {
                    "run_id": str(rid),
                    "task_id": payload.get("task_id"),
                    "trace_id": payload.get("trace_id"),
                    "attempt": int(payload.get("attempt") or 0),
                    "candidate_count": int(payload.get("candidate_count") or 0),
                    "validated_count": int(payload.get("validated_count") or 0),
                    "selected_count": int(payload.get("selected_count") or 0),
                    "blocked_count": int(payload.get("blocked_count") or len(blocked_raw)),
                    "rejected_count": int(payload.get("rejected_count") or len(rejected_raw)),
                    "blocked": blocked_raw,
                    "rejected": rejected_raw,
                    "selected_packet_ids": payload.get("selected_packet_ids") or [],
                    "created_at": _iso(row.get("created_at")),
                }
            )
            if len(decisions) >= limit:
                return decisions[:limit]
    return decisions[:limit]


def _reason_bucket(reason: Any) -> str:
    text = str(reason or "unknown").strip()
    if not text:
        return "unknown"
    token = text.split(":", 1)[0]
    return token[:80]


def _summarize_transfer_arbitration(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return _empty_transfer_arbitration_summary()

    candidate_total = 0
    selected_total = 0
    blocked_total = 0
    rejected_total = 0
    block_reason_counts: dict[str, int] = {}
    reject_reason_counts: dict[str, int] = {}
    for decision in decisions:
        candidate_total += int(decision.get("candidate_count") or 0)
        selected_total += int(decision.get("selected_count") or 0)
        blocked_total += int(decision.get("blocked_count") or 0)
        rejected_total += int(decision.get("rejected_count") or 0)

        blocked = decision.get("blocked")
        if isinstance(blocked, list):
            for row in blocked:
                if not isinstance(row, dict):
                    continue
                key = _reason_bucket(row.get("reason"))
                block_reason_counts[key] = block_reason_counts.get(key, 0) + 1
        rejected = decision.get("rejected")
        if isinstance(rejected, list):
            for row in rejected:
                if not isinstance(row, dict):
                    continue
                key = _reason_bucket(row.get("reason"))
                reject_reason_counts[key] = reject_reason_counts.get(key, 0) + 1

    selection_rate = (selected_total / candidate_total) if candidate_total > 0 else 0.0
    return {
        "decision_count": len(decisions),
        "candidate_total": candidate_total,
        "selected_total": selected_total,
        "blocked_total": blocked_total,
        "rejected_total": rejected_total,
        "avg_selected_per_decision": round(selected_total / len(decisions), 6),
        "selection_rate": round(selection_rate, 6),
        "block_reason_counts": block_reason_counts,
        "reject_reason_counts": reject_reason_counts,
    }


def _accumulate_model_failover_state(
    *,
    failover_state: dict[str, Any],
    cooldown_counts: dict[str, int],
    failure_counts: dict[str, int],
) -> None:
    for role_state in failover_state.values():
        if not isinstance(role_state, dict):
            continue
        for model, model_state in role_state.items():
            if not isinstance(model_state, dict):
                continue
            failures = int(model_state.get("failure_count") or 0)
            cooldown = float(model_state.get("cooldown_remaining_seconds") or 0.0)
            if failures > 0:
                failure_counts[model] = failure_counts.get(model, 0) + failures
            if cooldown > 0:
                cooldown_counts[model] = cooldown_counts.get(model, 0) + 1


def _derive_organism_name(task_text: str) -> str:
    compact = " ".join(task_text.split())
    if len(compact) <= 72:
        return compact
    return compact[:69] + "..."


def _resolve_prompts_dir(config_dir: Path | None = None) -> Path:
    if config_dir is None:
        return Path(__file__).resolve().parent.parent / "config" / "prompts"
    return config_dir / "prompts"


def _gather_query_knowledge(
    *,
    queries: list[str],
    auto_gather: bool,
    search_client: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if not queries:
        return {}
    if not auto_gather or search_client is None:
        return {query: [] for query in queries}

    gathered: dict[str, list[dict[str, Any]]] = {}
    try:
        for query in queries:
            hits = search_client.search(query)
            gathered[query] = [hit.to_dict() for hit in hits]
    finally:
        if hasattr(search_client, "close"):
            search_client.close()
    return gathered


def _archive_report(report_out: Path, archive_dir: Path, run_id: str | None) -> None:
    if run_id is None:
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"sotappr_{run_id}_{timestamp}.json"
    shutil.copyfile(report_out, archive_path)


def _echo_model_table(models: list[dict[str, Any]], criteria: str) -> None:
    click.echo(f"OpenRouter models ranked by {criteria}:")
    click.echo(
        " #  model_id                           provider   prompt        completion    "
        "created                 latency_ms"
    )
    for index, row in enumerate(models, start=1):
        model_id = str(row.get("id") or "")[:33].ljust(33)
        provider = str(row.get("provider") or "")[:8].ljust(8)
        prompt_cost = _format_cost(row.get("prompt_cost")).ljust(12)
        completion_cost = _format_cost(row.get("completion_cost")).ljust(12)
        created = str(row.get("created_iso") or "n/a")[:23].ljust(23)
        latency = _format_latency(row.get("latency_ms")).rjust(9)
        click.echo(
            f"{str(index).rjust(2)}  {model_id} {provider} {prompt_cost} "
            f"{completion_cost} {created} {latency}"
        )


def _format_cost(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        numeric = float(value)
        return f"{numeric:.8f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_latency(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def _persist_model_role(role: str, model_id: str, config_dir: Path | None = None) -> Path:
    models_path = _resolve_models_path(config_dir=config_dir)
    if not models_path.exists():
        models_path.parent.mkdir(parents=True, exist_ok=True)
        models_path.write_text("roles:\n", encoding="utf-8")

    # Create backup before modifying
    backup_path = models_path.with_suffix(".yaml.bak")
    shutil.copyfile(models_path, backup_path)

    lines = models_path.read_text(encoding="utf-8").splitlines()
    updated = _update_role_mapping_in_lines(lines=lines, role=role, model_id=model_id)
    models_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return models_path


def _resolve_models_path(config_dir: Path | None = None) -> Path:
    base = config_dir or (Path(__file__).resolve().parent.parent / "config")
    return base / "models.yaml"


def _update_role_mapping_in_lines(lines: list[str], role: str, model_id: str) -> list[str]:
    if not lines:
        return ["roles:", f"  {role}: {model_id}"]

    roles_idx = None
    roles_indent = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("roles:"):
            roles_idx = idx
            roles_indent = len(line) - len(line.lstrip(" "))
            break

    if roles_idx is None:
        appended = list(lines)
        if appended and appended[-1].strip():
            appended.append("")
        appended.extend(["roles:", f"  {role}: {model_id}"])
        return appended

    end_idx = len(lines)
    for idx in range(roles_idx + 1, len(lines)):
        line = lines[idx]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= roles_indent:
            end_idx = idx
            break

    role_prefix = " " * (roles_indent + 2) + f"{role}:"
    updated_lines = list(lines)
    for idx in range(roles_idx + 1, end_idx):
        stripped = updated_lines[idx].strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = stripped.replace("\t", " ")
        if normalized.startswith(f"{role}:"):
            updated_lines[idx] = f"{' ' * (roles_indent + 2)}{role}: {model_id}"
            return updated_lines

    updated_lines.insert(end_idx, f"{role_prefix} {model_id}")
    return updated_lines


@cli.command("datasci-status")
@click.option("--project-id", required=False, default=None, help="Filter by project ID.")
@click.option("--phase", required=False, default=None, help="Filter by DS phase (e.g. 'data_audit').")
@click.option("--limit", required=False, default=50, show_default=True, type=int)
def datasci_status(project_id: str | None, phase: str | None, limit: int) -> None:
    """Show data science experiment status and results."""
    ctx = _open_repo_context()
    try:
        if project_id is None:
            # Show all DS experiments across projects
            experiments = ctx.repository.engine.fetch_all(
                "SELECT * FROM ds_experiments ORDER BY created_at DESC LIMIT %s",
                [limit],
            )
        else:
            project_uuid = _parse_uuid(project_id, "project_id")
            experiments = ctx.repository.get_ds_experiments(
                project_id=project_uuid, phase=phase, limit=limit,
            )

        rows = []
        for exp in experiments:
            row: dict[str, Any] = {
                "experiment_id": str(exp["id"]) if "id" in exp else str(exp.get("experiment_id", "")),
                "project_id": str(exp.get("project_id", "")),
                "phase": exp.get("experiment_phase", ""),
                "status": exp.get("status", ""),
                "config": exp.get("experiment_config"),
                "metrics": exp.get("metrics"),
                "created_at": _iso(exp.get("created_at")),
                "updated_at": _iso(exp.get("updated_at")),
            }
            rows.append(row)

        _echo_json({"experiments": rows, "count": len(rows)})
    finally:
        _close_repo_context(ctx)


@cli.command("datasci-report")
@click.option("--project-id", required=True, help="Project ID to fetch DS report for.")
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output file path (JSON). Prints to stdout if not specified.",
)
def datasci_report(project_id: str, out_path: Path | None) -> None:
    """Generate a consolidated data science pipeline report for a project."""
    ctx = _open_repo_context()
    try:
        project_uuid = _parse_uuid(project_id, "project_id")
        experiments = ctx.repository.get_ds_experiments(
            project_id=project_uuid, limit=100,
        )

        if not experiments:
            raise click.ClickException(
                f"No DS experiments found for project {project_id}.\n"
                "Run a DS pipeline first or verify the project ID."
            )

        # Group experiments by phase
        phases: dict[str, list[dict[str, Any]]] = {}
        for exp in experiments:
            phase_name = exp.get("experiment_phase", "unknown")
            phases.setdefault(phase_name, []).append(exp)

        # Build report
        phase_summaries: list[dict[str, Any]] = []
        for phase_name, phase_exps in sorted(phases.items()):
            latest = phase_exps[0]  # Already ordered by created_at DESC
            artifacts = []
            exp_id = latest.get("id") or latest.get("experiment_id")
            if exp_id:
                try:
                    artifact_rows = ctx.repository.get_ds_artifacts(UUID(str(exp_id)))
                    artifacts = [
                        {
                            "type": a.get("artifact_type", ""),
                            "path": a.get("artifact_path", ""),
                            "hash": a.get("artifact_hash", ""),
                        }
                        for a in artifact_rows
                    ]
                except Exception:
                    pass

            phase_summaries.append({
                "phase": phase_name,
                "status": latest.get("status", ""),
                "experiment_count": len(phase_exps),
                "latest_experiment_id": str(exp_id) if exp_id else "",
                "metrics": latest.get("metrics"),
                "artifacts_manifest": latest.get("artifacts_manifest"),
                "artifacts": artifacts,
                "created_at": _iso(latest.get("created_at")),
                "updated_at": _iso(latest.get("updated_at")),
            })

        # Overall status
        all_statuses = [s.get("status", "") for s in phase_summaries]
        overall = "COMPLETED" if all(s == "COMPLETED" for s in all_statuses) else "IN_PROGRESS"
        if any(s == "FAILED" for s in all_statuses):
            overall = "FAILED"

        report = {
            "project_id": project_id,
            "overall_status": overall,
            "total_experiments": len(experiments),
            "phases": phase_summaries,
        }

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            click.echo(f"Wrote DS report to {out_path}")
        else:
            _echo_json(report)
    finally:
        _close_repo_context(ctx)


@cli.command("datasci-run")
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the dataset file (CSV, Parquet, etc.).",
)
@click.option("--target-column", required=True, help="Name of the target variable column.")
@click.option(
    "--problem-type",
    required=False,
    type=click.Choice(["classification", "regression"]),
    default="classification",
    show_default=True,
    help="ML problem type.",
)
@click.option(
    "--sensitive-columns",
    required=False,
    default="",
    help="Comma-separated list of sensitive columns for fairness checks.",
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional config directory for ComponentFactory.",
)
@click.option("--env", required=False, default=None, help="Optional config overlay environment.")
@click.option(
    "--api-key",
    required=False,
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="OpenRouter API key (falls back to OPENROUTER_API_KEY).",
)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional output file path for pipeline state (JSON).",
)
def datasci_run(
    dataset: Path,
    target_column: str,
    problem_type: str,
    sensitive_columns: str,
    config_dir: Path | None,
    env: str | None,
    api_key: str | None,
    out_path: Path | None,
) -> None:
    """Run the full 7-phase data science pipeline on a dataset."""
    component_factory = _load_component_factory()
    bundle = None
    try:
        bundle = component_factory.create(
            config_dir=config_dir, env=env, api_key=api_key,
        )

        if not bundle.config.datasci.enabled:
            raise click.ClickException(
                "Data science pipeline is not enabled. "
                "Set 'datasci.enabled: true' in config/default.yaml."
            )

        if bundle.ds_pipeline is None:
            raise click.ClickException(
                "DS pipeline could not be initialized. "
                "Install datasci dependencies: pip install -e '.[datasci]'"
            )

        # Create a project for this standalone DS run
        from src.core.models import Project
        project = Project(
            name=f"ds-run-{dataset.stem}",
            repo_path=str(dataset.parent.resolve()),
            tech_stack={"type": "datasci", "problem_type": problem_type},
            project_rules="",
        )
        bundle.repository.create_project(project)

        # Parse sensitive columns
        sens_cols = [c.strip() for c in sensitive_columns.split(",") if c.strip()]

        input_data = {
            "dataset_path": str(dataset.resolve()),
            "target_column": target_column,
            "problem_type": problem_type,
            "sensitive_columns": sens_cols,
            "project_id": project.id,
        }

        click.echo(click.style("Starting DS pipeline...", bold=True))
        click.echo(f"  Dataset:        {dataset}")
        click.echo(f"  Target column:  {target_column}")
        click.echo(f"  Problem type:   {problem_type}")
        if sens_cols:
            click.echo(f"  Sensitive cols: {', '.join(sens_cols)}")
        click.echo(f"  Project ID:     {project.id}")
        click.echo()

        state = bundle.ds_pipeline.run(input_data)

        # Summarize results
        phases_completed = []
        if state.audit_report is not None:
            phases_completed.append("data_audit")
        if state.eda_report is not None:
            phases_completed.append("eda")
        if state.feature_report is not None:
            phases_completed.append("feature_engineering")
        if state.training_report is not None:
            phases_completed.append("model_training")
        if state.ensemble_report is not None:
            phases_completed.append("ensemble")
        if state.evaluation_report is not None:
            phases_completed.append("evaluation")
        if state.deployment_package is not None:
            phases_completed.append("deployment")

        click.echo(click.style("DS pipeline complete.", bold=True))
        click.echo(f"  Phases completed: {len(phases_completed)}/7")
        for p in phases_completed:
            click.echo(click.style(f"    {p}", fg="green"))

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            click.echo(f"\nWrote pipeline state to {out_path}")
        else:
            click.echo(f"\n  Project ID: {project.id}")
            click.echo("  Use 'associate datasci-report --project-id <id>' for full report.")

    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"DS pipeline failed: {exc}") from exc
    finally:
        if bundle is not None:
            component_factory.close(bundle)


@cli.command("datasci-benchmark")
@click.option(
    "--tier",
    required=False,
    type=click.Choice(["mild", "moderate", "severe", "extreme", "all"]),
    default="all",
    show_default=True,
    help="Imbalance tier to benchmark (or 'all').",
)
@click.option(
    "--max-datasets",
    required=False,
    type=int,
    default=None,
    help="Maximum number of datasets to run.",
)
@click.option(
    "--cache-dir",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Cache directory for downloaded datasets.",
)
@click.option(
    "--out",
    "out_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file path for JSON report.",
)
@click.option(
    "--config-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Optional config directory for ComponentFactory.",
)
@click.option(
    "--env",
    required=False,
    default=None,
    help="Optional config overlay environment.",
)
@click.option(
    "--api-key",
    required=False,
    default=None,
    envvar="OPENROUTER_API_KEY",
    help="OpenRouter API key (falls back to OPENROUTER_API_KEY).",
)
def datasci_benchmark(
    tier: str,
    max_datasets: int | None,
    cache_dir: Path | None,
    out_path: Path | None,
    config_dir: Path | None,
    env: str | None,
    api_key: str | None,
) -> None:
    """Run the DS pipeline across standard imbalanced benchmark datasets."""
    component_factory = _load_component_factory()
    bundle = None
    try:
        bundle = component_factory.create(
            config_dir=config_dir, env=env, api_key=api_key,
        )

        if not bundle.config.datasci.enabled:
            raise click.ClickException(
                "Data science pipeline is not enabled. "
                "Set 'datasci.enabled: true' in config/default.yaml."
            )

        if bundle.ds_pipeline is None:
            raise click.ClickException(
                "DS pipeline could not be initialized. "
                "Install datasci dependencies: pip install -e '.[datasci]'"
            )

        from src.datasci.benchmark.loader import BenchmarkLoader
        from src.datasci.benchmark.report import BenchmarkReporter
        from src.datasci.benchmark.runner import BenchmarkRunner

        loader = BenchmarkLoader(cache_dir=cache_dir)
        runner = BenchmarkRunner(
            pipeline=bundle.ds_pipeline,
            loader=loader,
            repository=bundle.repository,
        )
        reporter = BenchmarkReporter()

        # Determine tier filter
        tiers = None if tier == "all" else [tier]

        click.echo(click.style("Starting DS benchmark run...", bold=True))
        click.echo(f"  Tier filter:   {tier}")
        if max_datasets:
            click.echo(f"  Max datasets:  {max_datasets}")
        click.echo(f"  Cache dir:     {loader.cache_dir}")
        click.echo()

        results = runner.run_all(
            tiers=tiers, max_datasets=max_datasets,
        )

        report = reporter.compile(results)

        # Display table
        table_output = reporter.to_table(report)
        click.echo(table_output)

        # Optionally write JSON report
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            json_output = reporter.to_json(report)
            out_path.write_text(json_output, encoding="utf-8")
            click.echo(f"\nWrote JSON report to {out_path}")

    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(
            f"Benchmark run failed: {exc}"
        ) from exc
    finally:
        if bundle is not None:
            component_factory.close(bundle)


@cli.command("doctor")
def doctor() -> None:
    """Check that all prerequisites are properly configured."""
    project_root = Path(__file__).resolve().parent.parent
    config_dir = project_root / "config"
    checks: list[tuple[str, bool, str]] = []  # (label, passed, detail)

    # 1. .env file exists
    env_path = project_root / ".env"
    if env_path.is_file():
        checks.append((".env file", True, str(env_path)))
    else:
        checks.append((".env file", False, f"Not found at {env_path}"))

    # 2. OPENROUTER_API_KEY is set and non-empty
    import os
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if api_key and not api_key.startswith("sk-or-v1-your"):
        masked = api_key[:12] + "..." + api_key[-4:]
        checks.append(("OPENROUTER_API_KEY", True, masked))
    elif api_key:
        checks.append(("OPENROUTER_API_KEY", False, "Set but appears to be placeholder value"))
    else:
        checks.append(("OPENROUTER_API_KEY", False, "Not set in environment"))

    # 3. DATABASE_URL
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        # Mask password in display
        from urllib.parse import urlparse
        try:
            parsed = urlparse(db_url)
            display = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}:{parsed.port}/{parsed.path.lstrip('/')}"
        except Exception:
            display = db_url[:30] + "..."
        checks.append(("DATABASE_URL", True, display))
    else:
        checks.append(("DATABASE_URL", False, "Not set — will use config/default.yaml defaults"))

    # 4. config/default.yaml parseable
    default_yaml_path = config_dir / "default.yaml"
    try:
        import yaml as _yaml
        with open(default_yaml_path) as f:
            data = _yaml.safe_load(f)
        if isinstance(data, dict):
            db_port = data.get("database", {}).get("port", "?")
            checks.append(("config/default.yaml", True, f"Parsed OK (db port: {db_port})"))
        else:
            checks.append(("config/default.yaml", False, "File parsed but is not a mapping"))
    except FileNotFoundError:
        checks.append(("config/default.yaml", False, f"Not found at {default_yaml_path}"))
    except Exception as exc:
        checks.append(("config/default.yaml", False, f"Parse error: {exc}"))

    # 5. config/models.yaml has all required roles
    models_yaml_path = config_dir / "models.yaml"
    try:
        with open(models_yaml_path) as f:
            models_data = _yaml.safe_load(f)
        if not isinstance(models_data, dict):
            checks.append(("config/models.yaml", False, "File parsed but is not a mapping"))
        else:
            roles = models_data.get("roles", {})
            missing = [r for r in _MODEL_ROLES if r not in roles]
            if missing:
                checks.append(("config/models.yaml", False, f"Missing roles: {', '.join(missing)}"))
            else:
                role_summary = ", ".join(f"{r}={roles[r].split('/')[-1]}" for r in _MODEL_ROLES)
                checks.append(("config/models.yaml", True, role_summary))
    except FileNotFoundError:
        checks.append(("config/models.yaml", False, f"Not found at {models_yaml_path}"))
    except Exception as exc:
        checks.append(("config/models.yaml", False, f"Parse error: {exc}"))

    # 6. PostgreSQL connectivity
    try:
        from src.core.config import load_config
        config = load_config(config_dir=config_dir)
        conn_str = config.database.connection_string
        import psycopg
        conn = psycopg.connect(conn_str, connect_timeout=5)
        server_version = conn.execute("SHOW server_version").fetchone()
        conn.close()
        ver = server_version[0] if server_version else "unknown"
        checks.append(("PostgreSQL", True, f"Connected on port {config.database.port} (v{ver})"))
    except ImportError:
        checks.append(("PostgreSQL", False, "psycopg not installed"))
    except Exception as exc:
        err_msg = str(exc).split("\n")[0][:80]
        checks.append(("PostgreSQL", False, f"Connection failed: {err_msg}"))

    # 7. pgvector extension
    try:
        conn = psycopg.connect(conn_str, connect_timeout=5)  # type: ignore[possibly-undefined]
        result = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        conn.close()
        if result:
            checks.append(("pgvector extension", True, f"v{result[0]}"))
        else:
            checks.append(("pgvector extension", False, "Extension not installed in database"))
    except Exception:
        checks.append(("pgvector extension", False, "Could not check (PostgreSQL not reachable)"))

    # Print results
    click.echo()
    click.echo(click.style("  Associate Doctor", bold=True))
    click.echo(click.style("  ================", bold=True))
    click.echo()

    passed = 0
    failed = 0
    for label, ok, detail in checks:
        if ok:
            icon = click.style("PASS", fg="green", bold=True)
            passed += 1
        else:
            icon = click.style("FAIL", fg="red", bold=True)
            failed += 1
        click.echo(f"  [{icon}] {label}")
        click.echo(f"         {detail}")

    click.echo()
    if failed == 0:
        click.echo(click.style(f"  All {passed} checks passed.", fg="green", bold=True))
    else:
        click.echo(click.style(f"  {failed} of {passed + failed} checks failed.", fg="red", bold=True))
    click.echo()


def main() -> None:
    """Entry point used by `associate` console script."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=True)
    cli()


if __name__ == "__main__":
    main()
