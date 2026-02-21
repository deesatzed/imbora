"""Component factory for The Associate.

Adapted from HMLR's ComponentFactory pattern. Creates and wires all
infrastructure components (database, LLM client, embeddings, model router)
and Phase 3 memory/search components so agents receive fully-initialized
dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core.config import (
    AppConfig,
    ModelRegistry,
    PromptLoader,
    load_config,
    load_model_registry,
)
from src.db.embeddings import EmbeddingEngine
from src.db.engine import DatabaseEngine
from src.db.repository import Repository
from src.llm.client import OpenRouterClient
from src.llm.router import ModelRouter
from src.memory.hybrid_search import HybridSearch
from src.memory.hypothesis_tracker import HypothesisTracker
from src.memory.methodology_store import MethodologyStore
from src.security.policy import AutonomyLevel, SecurityPolicy
from src.tools.search import TavilyClient

logger = logging.getLogger("associate.factory")


@dataclass
class ComponentBundle:
    """Container for all initialized infrastructure components.

    Agents receive the specific components they need from this bundle.
    The factory builds the bundle once; the orchestrator distributes
    references to each agent's __init__.
    """

    config: AppConfig
    model_registry: ModelRegistry
    db_engine: DatabaseEngine
    repository: Repository
    embedding_engine: EmbeddingEngine
    llm_client: OpenRouterClient
    model_router: ModelRouter
    security_policy: Optional[SecurityPolicy] = None
    # Phase 3 components
    hybrid_search: Optional[HybridSearch] = None
    methodology_store: Optional[MethodologyStore] = None
    hypothesis_tracker: Optional[HypothesisTracker] = None
    search_client: Optional[TavilyClient] = None
    # Data science pipeline orchestrator (optional, when datasci.enabled=True)
    ds_pipeline: Optional[object] = None


class ComponentFactory:
    """Factory for creating and wiring all Associate infrastructure.

    Usage:
        bundle = ComponentFactory.create(
            config_dir=Path("config"),
            api_key="sk-or-..."
        )
        # Pass bundle.repository, bundle.llm_client, etc. to agents
    """

    @staticmethod
    def create(
        config_dir: Optional[Path] = None,
        env: Optional[str] = None,
        api_key: Optional[str] = None,
        initialize_schema: bool = True,
        workspace_dir: Optional[Path] = None,
    ) -> ComponentBundle:
        """Create and wire all infrastructure components.

        Args:
            config_dir: Path to config/ directory. Default: project root/config.
            env: Environment name for config overlay (e.g., "test", "production").
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            initialize_schema: Whether to run schema.sql on startup. Default True.
            workspace_dir: Root directory for workspace boundary checks.
                Falls back to Path.cwd().resolve() when not specified.

        Returns:
            ComponentBundle with all components ready to use.
        """
        logger.info("Initializing components...")

        # --- Config ---
        config = load_config(config_dir=config_dir, env=env)
        model_registry = load_model_registry(config_dir=config_dir)
        logger.info("Config loaded (%d model roles)", len(model_registry.roles))

        # --- Database ---
        db_engine = DatabaseEngine(config.database)
        if initialize_schema:
            db_engine.initialize_schema()
            logger.info("Database schema initialized")

        repository = Repository(db_engine)

        # --- Embeddings ---
        embedding_engine = EmbeddingEngine(config.embeddings)

        # --- LLM ---
        llm_client = OpenRouterClient(config=config.llm, api_key=api_key)
        model_router = ModelRouter(model_registry)
        logger.info("LLM client configured (base_url=%s)", config.llm.base_url)

        # --- Security policy ---
        autonomy_name = config.security.autonomy_level.lower()
        if autonomy_name in ("readonly", "read_only"):
            autonomy = AutonomyLevel.READ_ONLY
        elif autonomy_name == "full":
            autonomy = AutonomyLevel.FULL
        else:
            autonomy = AutonomyLevel.SUPERVISED

        security_policy = SecurityPolicy(
            autonomy=autonomy,
            workspace_only=config.security.workspace_only,
            allowed_commands=list(config.security.allowed_commands),
            forbidden_paths=list(config.security.forbidden_paths),
            max_actions_per_hour=config.security.max_actions_per_hour,
            sanitize_env=config.security.sanitize_env,
            safe_env_vars=list(config.security.safe_env_vars),
            workspace_dir=(workspace_dir or Path.cwd()).resolve(),
        )

        # --- Phase 3: Memory & Search ---
        hybrid_search = HybridSearch(
            repository=repository,
            embedding_engine=embedding_engine,
        )
        methodology_store = MethodologyStore(
            repository=repository,
            embedding_engine=embedding_engine,
            hybrid_search=hybrid_search,
        )
        hypothesis_tracker = HypothesisTracker(repository=repository)
        search_client = TavilyClient(config=config.search)
        logger.info("Phase 3 memory/search components initialized")

        # --- Data science pipeline (optional) ---
        ds_pipeline = None
        if config.datasci.enabled:
            try:
                from src.datasci.pipeline import DSPipelineOrchestrator

                ds_pipeline = DSPipelineOrchestrator(
                    llm_client=llm_client,
                    model_router=model_router,
                    repository=repository,
                    ds_config=config.datasci,
                )
                logger.info("Data science pipeline initialized")
            except ImportError:
                logger.warning(
                    "datasci package not available; install with pip install -e '.[datasci]'"
                )

        logger.info("All components initialized")

        return ComponentBundle(
            config=config,
            model_registry=model_registry,
            db_engine=db_engine,
            repository=repository,
            embedding_engine=embedding_engine,
            llm_client=llm_client,
            model_router=model_router,
            security_policy=security_policy,
            hybrid_search=hybrid_search,
            methodology_store=methodology_store,
            hypothesis_tracker=hypothesis_tracker,
            search_client=search_client,
            ds_pipeline=ds_pipeline,
        )

    @staticmethod
    def close(bundle: ComponentBundle) -> None:
        """Cleanly shut down all components."""
        if bundle.search_client:
            bundle.search_client.close()
        bundle.llm_client.close()
        bundle.db_engine.close()
        logger.info("All components shut down")
