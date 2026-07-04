import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import uuid
from typing import Optional

from skydiscover.config import Config, build_output_dir, load_config
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.registry import create_database, get_program
from skydiscover.search.route import get_discovery_controller
from skydiscover.search.utils.logging_utils import setup_search_logging
from skydiscover.utils.async_utils import TaskPool
from skydiscover.utils.code_utils import extract_solution_language
from skydiscover.utils.metrics import format_metrics, get_score

logger = logging.getLogger(__name__)

DEFAULT_SEED_INGESTION_CONCURRENCY = max(1, min(8, os.cpu_count() or 1))
SEED_CHECKPOINT_ENV = "SKYDISCOVER_SEED_CHECKPOINT"
SEED_CHECKPOINT_INFO = "seed_checkpoint_info.json"
SEED_CHECKPOINT_VERSION = 1
SEED_CONCURRENCY_ENV = "SKYDISCOVER_SEED_INGESTION_CONCURRENCY"
SEED_DEDUP_ENV = "SKYDISCOVER_SEED_DEDUPLICATE"
SEED_TOP_K_ENV = "SKYDISCOVER_SEED_TOP_K"
SEED_FAST_MAX_NFEV_ENV = "SKYDISCOVER_SEED_FAST_MAX_NFEV"
SEED_CURVE_WORKERS_ENV = "SKYDISCOVER_SEED_CURVE_WORKERS"
SEED_MULTISTART_WORKERS_ENV = "SKYDISCOVER_SEED_MULTISTART_WORKERS"


class Runner:
    """Top-level entry point for a discovery run.

    Loads config, creates the database and discovery controller, runs the
    search loop, and saves checkpoints + best program.

    Args:
        initial_program_path: path to the starting solution file.
        evaluation_file: path to the user's evaluator script (must define evaluate()).
        config_path: optional YAML config file (ignored if config is provided).
        config: optional pre-built Config object (takes priority over config_path).
        output_dir: where to write logs, checkpoints, and best program.
            Auto-generated from search type + problem name if omitted.
    """

    def __init__(
        self,
        evaluation_file: str,
        initial_program_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
        evaluator_env_vars: Optional[dict[str, str]] = None,
    ):
        self.config = config if config is not None else load_config(config_path)
        self.config_path = config_path
        self.name = self.config.search.type
        self.output_dir = output_dir or build_output_dir(
            self.name, initial_program_path or "scratch"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

        # Load the initial program (can be optional)
        self.initial_program_path = initial_program_path
        self.initial_program_solution = (
            self._load_initial_program() if initial_program_path else None
        )
        if self.initial_program_solution and not self.config.language:
            self.config.language = extract_solution_language(self.initial_program_solution)
        if not self.config.language:
            self.config.language = "python"

        # Set the file extension
        ext = os.path.splitext(initial_program_path)[1] if initial_program_path else ".py"
        ext = ext or ".py"
        self.file_extension = ext if ext.startswith(".") else f".{ext}"
        if self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension
        self.additional_initial_programs = self._load_additional_initial_programs()

        # Create the database
        self.database = create_database(self.config.search.type, self.config.search.database)
        self.database.language = self.config.language or "python"
        self.evaluation_file = evaluation_file
        self.evaluator_env_vars = dict(evaluator_env_vars or {})

        # Initialize the discovery controller
        self.discovery_controller: Optional[DiscoveryController] = None

        logger.info(f"Runner ready: search={self.name}, program={self.initial_program_path}")

    @property
    def initial_score(self) -> Optional[float]:
        """Score of the seed program, or None if unavailable."""
        if not self.database or not self.database.programs or not self.initial_program_solution:
            return None

        seed_solution = self.initial_program_solution
        seed_prog = None
        for prog in self.database.programs.values():
            if prog.solution == seed_solution:
                seed_prog = prog
                break
        if seed_prog is None:
            for prog in self.database.programs.values():
                if prog.iteration_found == 0:
                    seed_prog = prog
                    break

        if seed_prog and seed_prog.metrics:
            return get_score(seed_prog.metrics)
        return None

    async def run(
        self,
        iterations: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """Entrypoint for the discovery process.

        Args:
            iterations: max iterations (uses config.max_iterations if None).
            checkpoint_path: resume from this checkpoint directory if provided.

        Returns:
            Best Program found, or None if no valid programs were produced.
        """
        max_iterations = iterations if iterations is not None else self.config.max_iterations

        start_iteration = 0
        loaded_seed_checkpoint = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from iteration {start_iteration}")
        else:
            seed_checkpoint = self._find_reusable_seed_checkpoint()
            if seed_checkpoint:
                self._clear_visible_run_history_snapshot()
                self._load_checkpoint(seed_checkpoint)
                start_iteration = max(self.database.last_iteration + 1, 1)
                loaded_seed_checkpoint = True
                logger.info(
                    "Reusing seed-evaluation checkpoint %s; starting discovery at iteration %s",
                    seed_checkpoint,
                    start_iteration,
                )
            else:
                start_iteration = self.database.last_iteration

        if start_iteration == 0 and len(self.database.programs) == 0:
            self._clear_visible_run_history_snapshot()

        # Create the discovery controller input
        controller_input = DiscoveryControllerInput(
            config=self.config,
            evaluation_file=self.evaluation_file,
            database=self.database,
            file_suffix=self.config.file_suffix,
            output_dir=self.output_dir,
            evaluator_env_vars=self.evaluator_env_vars,
        )

        # Get the discovery controller
        self.discovery_controller = get_discovery_controller(controller_input)

        # Add initial program to database if not resuming
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and self.initial_program_solution is not None
        )

        if should_add_initial:
            await self._add_initial_program(start_iteration)
            await self._add_additional_initial_programs(start_iteration)
            self._save_seed_checkpoint(start_iteration)
        else:
            logger.info(
                f"Resuming from iteration {start_iteration} with {len(self.database.programs)} programs"
            )

        # Start the monitor
        monitor_server = None
        try:
            monitor_server = self._start_monitor(max_iterations)
            self._setup_human_feedback(monitor_server)
            self._setup_monitor_summary(monitor_server)
            self._push_existing_to_monitor()
            self._install_signal_handlers()

            discovery_start = (
                start_iteration + 1
                if should_add_initial
                else max(start_iteration, 1)
                if loaded_seed_checkpoint
                else start_iteration
            )
            self.database.log_status()

            def checkpoint_cb(iteration: int) -> None:
                self._sync_database()
                self._save_checkpoint(iteration)

            # MAIN LOOP: Run the discovery
            from skydiscover.llm.cost_tracker import CostLimitExceeded

            try:
                await self.discovery_controller.run_discovery(
                    discovery_start,
                    max_iterations,
                    checkpoint_callback=checkpoint_cb,
                )
            except CostLimitExceeded as exc:
                logger.info("Cost limit reached, stopping discovery: %s", exc)
                print(f"\nCost limit reached: {exc}")
                print("Saving best result found so far...")

            self._sync_database()
            final_iteration = discovery_start + max_iterations - 1
            if final_iteration > 0:
                self._save_checkpoint(final_iteration)

            # Re-evaluate best program in test mode (authoritative score).
            best = self._get_best_program()
            if best:
                try:
                    test_result = await self.discovery_controller.evaluator.evaluate_program(
                        best.solution, best.id, mode="test"
                    )
                    for k, v in test_result.metrics.items():
                        best.metrics[f"test_{k}"] = v
                    logger.info(
                        f"Test evaluation for best program: {format_metrics(test_result.metrics)}"
                    )
                    # Persist test metrics to disk so they survive the run.
                    self._save_best_program(best)
                except Exception as e:
                    logger.warning(f"Test-mode re-evaluation failed: {e}")

        finally:
            # Stop the monitor
            early_stopped = (
                self.discovery_controller is not None
                and self.discovery_controller.early_stopping_triggered
            )
            if self.discovery_controller is not None:
                self.discovery_controller.close()
            self.discovery_controller = None

            if monitor_server:
                try:
                    reason = "early_stopping" if early_stopped else "completed"
                    monitor_server.push_event({"type": "discovery_complete", "reason": reason})
                    time.sleep(0.5)
                    monitor_server.stop()
                except Exception:
                    logger.debug("Failed to stop monitor server", exc_info=True)

        # Get the best program
        best_program = self._get_best_program()
        if best_program:
            status = "early stopping" if early_stopped else "completed"
            logger.info(f"Discovery {status}. Best: {format_metrics(best_program.metrics)}")
            self._save_best_program(best_program)
            return best_program

        logger.warning("No valid programs found")
        return None

    def _visible_run_history_snapshot_dir(self) -> Optional[Path]:
        """Return the benchmark-visible run-history snapshot directory, if any."""
        if not self.initial_program_path:
            return None

        path = Path(self.initial_program_path).resolve()
        for parent in path.parents:
            if parent.name == "pysr_symbolic":
                return parent / "reference" / "current_adaevolve_run"
        return None

    def _clear_visible_run_history_snapshot(self) -> None:
        """Clear stale visible run-history snapshots before a fresh run."""
        snapshot_dir = self._visible_run_history_snapshot_dir()
        if snapshot_dir is None:
            return

        snapshot_dir = snapshot_dir.resolve()
        if snapshot_dir.name != "current_adaevolve_run" or snapshot_dir.parent.name != "reference":
            logger.warning("Refusing to clear unexpected run-history path: %s", snapshot_dir)
            return

        try:
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared visible run-history snapshot: %s", snapshot_dir)
        except Exception as exc:
            logger.warning("Failed to clear visible run-history snapshot %s: %s", snapshot_dir, exc)

    def _default_seed_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoints", "seed_ingestion")

    def _configured_seed_checkpoint_path(self) -> Optional[str]:
        path = os.environ.get(SEED_CHECKPOINT_ENV)
        return path or None

    def _seed_checkpoint_candidates(self) -> list[str]:
        candidates: list[str] = []
        for path in (
            self._configured_seed_checkpoint_path(),
            self._default_seed_checkpoint_path(),
        ):
            if path and path not in candidates:
                candidates.append(path)
        return candidates

    def _file_digest(self, path: Optional[str]) -> Optional[str]:
        if not path or not os.path.exists(path):
            return None

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _seed_checkpoint_fingerprint(self) -> str:
        """Fingerprint inputs that make seed-evaluation metrics reusable."""
        payload = {
            "version": SEED_CHECKPOINT_VERSION,
            "search": self.name,
            "language": self.config.language,
            "file_suffix": self.config.file_suffix,
            "config": self.config.to_dict(),
            "evaluation_file": os.path.abspath(self.evaluation_file),
            "evaluation_file_sha256": self._file_digest(self.evaluation_file),
            "initial_program": os.path.abspath(self.initial_program_path)
            if self.initial_program_path
            else None,
            "initial_program_sha256": self._file_digest(self.initial_program_path),
            "additional_initial_programs": [
                {
                    "source_path": os.path.abspath(source_path),
                    "solution_sha256": hashlib.sha256(solution.encode("utf-8")).hexdigest(),
                }
                for source_path, solution in self.additional_initial_programs
            ],
            "evaluator_env_vars": self.evaluator_env_vars,
            "seed_relevant_env": {
                name: os.environ.get(name)
                for name in sorted(os.environ)
                if name.startswith("SKYDISCOVER_")
                and name not in {SEED_CHECKPOINT_ENV}
            },
        }
        encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _write_seed_checkpoint_info(self, path: str, fingerprint: str) -> None:
        from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

        info = {
            "version": SEED_CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "saved_at": time.time(),
            "search": self.name,
            "program_count": len(self.database.programs),
            "initial_program": self.initial_program_path,
            "evaluation_file": self.evaluation_file,
        }
        with open(os.path.join(path, SEED_CHECKPOINT_INFO), "w") as f:
            json.dump(info, f, indent=2, cls=SafeJSONEncoder)

    def _is_seed_checkpoint_reusable(self, path: str, fingerprint: str) -> bool:
        info_path = os.path.join(path, SEED_CHECKPOINT_INFO)
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(info_path) or not os.path.exists(metadata_path):
            return False

        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except Exception as exc:
            logger.warning("Could not read seed checkpoint metadata %s: %s", info_path, exc)
            return False

        if info.get("version") != SEED_CHECKPOINT_VERSION:
            logger.info("Ignoring seed checkpoint with old version: %s", path)
            return False
        if info.get("fingerprint") != fingerprint:
            logger.info("Ignoring stale seed checkpoint with mismatched fingerprint: %s", path)
            return False
        return True

    def _find_reusable_seed_checkpoint(self) -> Optional[str]:
        fingerprint = self._seed_checkpoint_fingerprint()
        for path in self._seed_checkpoint_candidates():
            if self._is_seed_checkpoint_reusable(path, fingerprint):
                return path
        return None

    def _save_seed_checkpoint(self, iteration: int) -> None:
        if not self.database.programs:
            return

        fingerprint = self._seed_checkpoint_fingerprint()
        for path in self._seed_checkpoint_candidates():
            try:
                os.makedirs(path, exist_ok=True)
                self.database.save(path, iteration)
                self._write_seed_checkpoint_info(path, fingerprint)
                logger.info("Seed-evaluation checkpoint saved to %s", path)
            except Exception as exc:
                logger.warning("Failed to save seed-evaluation checkpoint %s: %s", path, exc)

    def _int_env(self, name: str, default: int, *, minimum: int = 1) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return max(minimum, int(raw))
        except ValueError:
            logger.warning("Invalid integer for %s=%r; using %s", name, raw, default)
            return default

    def _bool_env(self, name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _seed_ingestion_concurrency(self) -> int:
        return self._int_env(
            SEED_CONCURRENCY_ENV,
            DEFAULT_SEED_INGESTION_CONCURRENCY,
        )

    def _seed_fast_max_nfev(self) -> Optional[int]:
        raw = os.environ.get(SEED_FAST_MAX_NFEV_ENV)
        if raw is None or raw == "":
            return 120
        if raw.strip().lower() in {"0", "none", "off", "false"}:
            return None
        return self._int_env(SEED_FAST_MAX_NFEV_ENV, 120)

    def _seed_top_k(self, n_seeds: int) -> int:
        default = min(24, n_seeds)
        raw = os.environ.get(SEED_TOP_K_ENV)
        if raw is None or raw == "":
            return default
        if raw.strip().lower() in {"0", "all", "none", "off", "false"}:
            return n_seeds
        return min(n_seeds, self._int_env(SEED_TOP_K_ENV, default))

    def _seed_evaluator_settings(self, *, max_nfev: Optional[int]) -> dict[str, int]:
        settings = {
            "ODE_CURVE_WORKERS": self._int_env(SEED_CURVE_WORKERS_ENV, 1),
            "ODE_MULTISTART_WORKERS": self._int_env(SEED_MULTISTART_WORKERS_ENV, 1),
        }
        if max_nfev is not None:
            settings["MAX_NFEV"] = max_nfev
        return settings

    def _canonical_seed_solution(self, solution: str) -> str:
        """Normalize superficial seed differences before deduplication."""
        lines = []
        in_docstring = False
        for line in solution.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') == 1 or stripped.count("'''") == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped:
                lines.append(stripped)
        return "\n".join(lines)

    def _unique_seed_specs(
        self,
        seed_specs: list[tuple[int, str, str, str]],
    ) -> list[tuple[int, str, str, str]]:
        if not self._bool_env(SEED_DEDUP_ENV, True):
            return seed_specs

        seen: dict[str, str] = {}
        unique_specs: list[tuple[int, str, str, str]] = []
        for spec in seed_specs:
            _, source_path, solution, _ = spec
            digest = hashlib.sha256(
                self._canonical_seed_solution(solution).encode("utf-8")
            ).hexdigest()
            if digest in seen:
                logger.info(
                    "Skipping duplicate seed program: %s (same template as %s)",
                    source_path,
                    seen[digest],
                )
                continue
            seen[digest] = source_path
            unique_specs.append(spec)

        if len(unique_specs) != len(seed_specs):
            logger.info(
                "Seed deduplication kept %d/%d programs",
                len(unique_specs),
                len(seed_specs),
            )
        return unique_specs

    def _patch_module_setting(self, module, name: str, value: int, old_values: list) -> None:
        if module is not None and hasattr(module, name):
            old_values.append((module, name, getattr(module, name)))
            setattr(module, name, value)

    def _patch_evaluate_ode_max_nfev(self, module, max_nfev: int, old_values: list) -> None:
        if module is None or not hasattr(module, "evaluate_ode_expression"):
            return

        original = getattr(module, "evaluate_ode_expression")

        def evaluate_ode_expression_with_seed_budget(*args, **kwargs):
            kwargs.setdefault("max_nfev", max_nfev)
            return original(*args, **kwargs)

        old_values.append((module, "evaluate_ode_expression", original))
        setattr(module, "evaluate_ode_expression", evaluate_ode_expression_with_seed_budget)

    async def _evaluate_seed_batch(
        self,
        seed_specs: list[tuple[int, str, str, str]],
        *,
        max_nfev: Optional[int],
        phase: str,
    ):
        if not seed_specs:
            return []

        evaluator = self.discovery_controller.evaluator
        original_pool = evaluator.task_pool
        original_env = {
            "SKYDISCOVER_ODE_CURVE_WORKERS": os.environ.get("SKYDISCOVER_ODE_CURVE_WORKERS"),
            "SKYDISCOVER_ODE_MULTISTART_WORKERS": os.environ.get(
                "SKYDISCOVER_ODE_MULTISTART_WORKERS"
            ),
            "SKYDISCOVER_INHIBITED_MAX_NFEV": os.environ.get(
                "SKYDISCOVER_INHIBITED_MAX_NFEV"
            ),
        }
        old_values: list[tuple[object, str, object]] = []
        settings = self._seed_evaluator_settings(max_nfev=max_nfev)
        concurrency = self._seed_ingestion_concurrency()

        try:
            evaluator.task_pool = TaskPool(max_concurrency=concurrency)
            os.environ["SKYDISCOVER_ODE_CURVE_WORKERS"] = str(settings["ODE_CURVE_WORKERS"])
            os.environ["SKYDISCOVER_ODE_MULTISTART_WORKERS"] = str(
                settings["ODE_MULTISTART_WORKERS"]
            )
            if max_nfev is not None:
                os.environ["SKYDISCOVER_INHIBITED_MAX_NFEV"] = str(max_nfev)

            module = getattr(evaluator, "_eval_module", None)
            base_module = getattr(module, "_base", None) if module is not None else None
            for target in (module, base_module):
                self._patch_module_setting(
                    target,
                    "ODE_CURVE_WORKERS",
                    settings["ODE_CURVE_WORKERS"],
                    old_values,
                )
                self._patch_module_setting(
                    target,
                    "ODE_MULTISTART_WORKERS",
                    settings["ODE_MULTISTART_WORKERS"],
                    old_values,
                )
                if max_nfev is not None:
                    self._patch_module_setting(target, "MAX_NFEV", max_nfev, old_values)
                    self._patch_evaluate_ode_max_nfev(target, max_nfev, old_values)

            logger.info(
                "Evaluating %d seed programs (%s pass, max_concurrent=%s, curve_workers=%s, multistart_workers=%s, max_nfev=%s)",
                len(seed_specs),
                phase,
                concurrency,
                settings["ODE_CURVE_WORKERS"],
                settings["ODE_MULTISTART_WORKERS"],
                max_nfev if max_nfev is not None else "default",
            )
            return await evaluator.evaluate_batch(
                [(solution, program_id) for _, _, solution, program_id in seed_specs]
            )
        finally:
            evaluator.task_pool = original_pool
            for module, name, old_value in reversed(old_values):
                setattr(module, name, old_value)
            for key, old_value in original_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    def _rank_seed_specs(
        self,
        seed_specs: list[tuple[int, str, str, str]],
        eval_results,
        top_k: int,
    ) -> list[tuple[int, str, str, str]]:
        scored_specs = []
        for order, (spec, eval_result) in enumerate(zip(seed_specs, eval_results)):
            score = get_score(eval_result.metrics or {})
            scored_specs.append((score, -order, spec))

        scored_specs.sort(reverse=True)
        selected = [spec for _, _, spec in scored_specs[:top_k]]
        if len(selected) != len(seed_specs):
            logger.info(
                "Seed fast pass selected top %d/%d programs for full evaluation",
                len(selected),
                len(seed_specs),
            )
        return selected

    # ------------------------------------------------------------------
    # Initial program
    # ------------------------------------------------------------------

    async def _add_initial_program(
        self,
        start_iteration: int,
        *,
        solution: Optional[str] = None,
        source_path: Optional[str] = None,
        target_island: Optional[int] = None,
    ) -> Optional[Program]:
        seed_solution = solution if solution is not None else self.initial_program_solution
        if seed_solution is None:
            return None

        seed_label = source_path or self.initial_program_path or "initial program"
        logger.info(f"Adding initial program to database: {seed_label}")
        program_id = str(uuid.uuid4())

        initial_image_path = None
        if self.config.language == "image":
            logger.info("Generating initial image from seed text...")
            img_dir = os.path.join(self.output_dir, "generated_images")
            try:
                result = await self.discovery_controller.llms.generate(
                    system_message="Generate an image based on the following description. Also provide brief reasoning about your creative choices.",
                    messages=[{"role": "user", "content": seed_solution}],
                    image_output=True,
                    output_dir=img_dir,
                    program_id=program_id,
                )
                initial_image_path = result.image_path
                logger.info(f"Initial image: {initial_image_path}")
            except Exception as e:
                logger.warning(f"Failed to generate initial image: {e}")

        eval_input = (
            initial_image_path
            if self.config.language == "image" and initial_image_path
            else seed_solution
        )
        eval_result = await self.discovery_controller.evaluator.evaluate_program(
            eval_input, program_id
        )
        metrics = eval_result.metrics

        if not initial_image_path and isinstance(metrics.get("image_path"), str):
            initial_image_path = metrics.pop("image_path")

        program = get_program(
            self.config, seed_solution, program_id, metrics, start_iteration
        )
        program.artifacts = eval_result.artifacts
        program.metadata = program.metadata or {}
        program.metadata["seed_source"] = seed_label

        if initial_image_path:
            program.metadata["image_path"] = initial_image_path

        self.database.add(program, iteration=start_iteration, target_island=target_island)
        try:
            if solution is None:
                self.database.initial_program_id = program.id
                self.database.initial_program_score = get_score(program.metrics or {})
        except Exception as e:
            logger.warning(f"Failed to set initial program score: {e}")
        return program

    async def _add_additional_initial_programs(self, start_iteration: int) -> None:
        if not self.additional_initial_programs:
            return

        logger.info(f"Adding {len(self.additional_initial_programs)} additional seed programs")
        if self.config.language == "image":
            for idx, (source_path, solution) in enumerate(self.additional_initial_programs):
                await self._add_initial_program(
                    start_iteration,
                    solution=solution,
                    source_path=source_path,
                    target_island=self._additional_seed_target_island(idx),
                )
            return

        seed_specs: list[tuple[int, str, str, str]] = []
        for idx, (source_path, solution) in enumerate(self.additional_initial_programs):
            program_id = str(uuid.uuid4())
            seed_specs.append((idx, source_path, solution, program_id))
            logger.info(f"Queueing seed program for evaluation: {source_path}")

        seed_specs = self._unique_seed_specs(seed_specs)
        fast_max_nfev = self._seed_fast_max_nfev()
        top_k = self._seed_top_k(len(seed_specs))
        if fast_max_nfev is not None and top_k < len(seed_specs):
            fast_results = await self._evaluate_seed_batch(
                seed_specs,
                max_nfev=fast_max_nfev,
                phase="fast",
            )
            seed_specs = self._rank_seed_specs(seed_specs, fast_results, top_k)

            # Give the selected seeds fresh IDs so full-budget results are distinct
            # from any temporary IDs used during the fast pass.
            seed_specs = [
                (idx, source_path, solution, str(uuid.uuid4()))
                for idx, source_path, solution, _ in seed_specs
            ]

        eval_results = await self._evaluate_seed_batch(
            seed_specs,
            max_nfev=None,
            phase="full",
        )

        for (idx, source_path, solution, program_id), eval_result in zip(
            seed_specs, eval_results
        ):
            metrics = eval_result.metrics
            initial_image_path = metrics.pop("image_path", None)

            program = get_program(
                self.config, solution, program_id, metrics, start_iteration
            )
            program.artifacts = eval_result.artifacts
            program.metadata = program.metadata or {}
            program.metadata["seed_source"] = source_path
            if initial_image_path:
                program.metadata["image_path"] = initial_image_path

            logger.info(f"Adding evaluated seed program to database: {source_path}")
            self.database.add(
                program,
                iteration=start_iteration,
                target_island=self._additional_seed_target_island(idx),
            )

    def _additional_seed_target_island(self, idx: int) -> Optional[int]:
        if not getattr(self.database, "num_islands", 0):
            return None

        # The primary seed starts on island 0; offset variants to keep
        # an even 80-seed spread across four islands.
        return (idx + 1) % self.database.num_islands

    # ------------------------------------------------------------------
    # Monitor and feedback setup
    # ------------------------------------------------------------------

    def _start_monitor(self, max_iterations: int):
        if not self.config.monitor.enabled:
            return None
        try:
            from skydiscover.extras.monitor import MonitorServer, create_monitor_callback

            server = MonitorServer(
                host=self.config.monitor.host,
                port=self.config.monitor.port,
                max_solution_length=self.config.monitor.max_solution_length,
            )
            server.set_config_summary(f"{self.name} | max_iter={max_iterations}")
            server.start()

            callback = create_monitor_callback(server, self.database, time.time())
            self.discovery_controller.monitor_callback = callback

            url = f"http://localhost:{server.port}/"
            print(f"\n  Live monitor: {url}\n", flush=True)
            logger.info(f"Live monitor: {url}")
            return server
        except Exception as e:
            logger.warning(f"Failed to start monitor: {e}")
            return None

    def _setup_human_feedback(self, monitor_server) -> None:
        if not (self.config.human_feedback_enabled or monitor_server):
            return
        try:
            from skydiscover.context_builder import HumanFeedbackReader

            path = self.config.human_feedback_file or os.path.join(
                self.output_dir, "human_feedback.md"
            )
            mode = getattr(self.config, "human_feedback_mode", "append")
            reader = HumanFeedbackReader(path, mode=mode)
            self.discovery_controller.feedback_reader = reader
            if monitor_server:
                monitor_server.set_feedback_reader(reader)
            logger.info(f"Human feedback: {path}")
        except Exception as e:
            logger.warning(f"Failed to set up human feedback: {e}")

    def _setup_monitor_summary(self, monitor_server) -> None:
        if not (monitor_server and self.config.monitor.summary_model):
            return
        try:
            monitor_server.configure_summary(
                model=self.config.monitor.summary_model,
                api_key=self.config.monitor.summary_api_key or "",
                api_base=self.config.monitor.summary_api_base,
                top_k=self.config.monitor.summary_top_k,
                interval=self.config.monitor.summary_interval,
            )
        except Exception as e:
            logger.warning(f"Failed to configure AI summary: {e}")

    def _push_existing_to_monitor(self) -> None:
        if not (self.discovery_controller.monitor_callback and self.database.programs):
            return
        for prog in self.database.programs.values():
            try:
                self.discovery_controller.monitor_callback(
                    prog, getattr(prog, "iteration_found", 0)
                )
            except Exception:
                logger.debug("Monitor callback failed for program %s", prog.id, exc_info=True)
        logger.info(f"Pushed {len(self.database.programs)} existing program(s) to monitor")

    def _install_signal_handlers(self) -> None:
        def on_signal(signum, frame):
            logger.info(f"Signal {signum} received, shutting down...")
            if self.discovery_controller is not None:
                self.discovery_controller.request_shutdown()

            def force_exit(signum, frame):
                sys.exit(128 + signum)

            # After the first termination signal, ensure subsequent SIGINT/SIGTERM
            # cause an immediate exit instead of re-running the soft handler.
            signal.signal(signal.SIGINT, force_exit)
            signal.signal(signal.SIGTERM, force_exit)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    # ------------------------------------------------------------------
    # Checkpointing and saving
    # ------------------------------------------------------------------

    def _sync_database(self) -> None:
        """Ensure we have the controller's latest database"""
        db = getattr(self.discovery_controller, "database", None)
        if db is not None and db is not self.database:
            self.database = db

    def _setup_logging(self) -> None:
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        setup_search_logging(log_level=self.config.log_level, log_dir=log_dir, name=self.name)

    def _load_initial_program(self) -> str:
        with open(self.initial_program_path, "r") as f:
            return f.read()

    def _load_additional_initial_programs(self) -> list[tuple[str, str]]:
        params = getattr(getattr(self.config, "benchmark", None), "params", {}) or {}
        seed_dir = params.get("seed_programs_dir")
        if not seed_dir:
            return []

        seed_path = Path(seed_dir)
        if not seed_path.is_absolute():
            base_path = Path(self.initial_program_path).resolve().parent if self.initial_program_path else Path.cwd()
            seed_path = base_path / seed_path

        if not seed_path.exists():
            logger.warning(f"Configured seed_programs_dir does not exist: {seed_path}")
            return []

        programs: list[tuple[str, str]] = []
        for path in sorted(seed_path.glob(f"*{self.file_extension}")):
            if path.resolve() == Path(self.initial_program_path).resolve():
                continue
            with path.open("r") as f:
                programs.append((str(path), f.read()))
        return programs

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.database.save(checkpoint_path, iteration)

        best = self._get_best_program()
        if best:
            with open(
                os.path.join(checkpoint_path, f"best_program{self.file_extension}"), "w"
            ) as f:
                f.write(best.solution)
            with open(os.path.join(checkpoint_path, "best_program_info.json"), "w") as f:
                from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

                json.dump(
                    {
                        "id": best.id,
                        "generation": best.generation,
                        "iteration": best.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best.metrics,
                        "language": best.language,
                        "timestamp": best.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                    cls=SafeJSONEncoder,
                )
            logger.info(f"Checkpoint {iteration}: best={format_metrics(best.metrics)}")

        current_iteration_best = self._get_current_iteration_best_program(iteration)
        if current_iteration_best:
            self._save_validation_curve_plot(
                current_iteration_best,
                checkpoint_path,
                label=f"iteration_{iteration}_best",
            )

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Loaded checkpoint (iteration {self.database.last_iteration})")

    def _get_best_program(self) -> Optional[Program]:
        if self.database.best_program_id:
            prog = self.database.get(self.database.best_program_id)
            if prog:
                return prog
        return self.database.get_best_program()

    def _get_current_iteration_best_program(self, iteration: int) -> Optional[Program]:
        """Return the best scored program first found in this iteration."""
        programs = [
            program
            for program in self.database.programs.values()
            if program.iteration_found == iteration and program.metrics
        ]
        if not programs:
            return None
        return max(programs, key=lambda program: get_score(program.metrics))

    def _save_validation_curve_plot(
        self,
        program: Program,
        checkpoint_path: str,
        *,
        label: str,
    ) -> None:
        """Save validation points plus predicted ODE curves for checkpoint inspection."""
        equation = str(program.metrics.get("equation") or "").strip()
        if not equation:
            return
        try:
            plot_path = self._build_validation_curve_plot(program, checkpoint_path, label, equation)
            if plot_path:
                logger.info("Saved validation curve plot for %s to %s", program.id, plot_path)
        except Exception as exc:
            logger.warning("Failed to save validation curve plot for %s: %s", program.id, exc)

    def _build_validation_curve_plot(
        self,
        program: Program,
        checkpoint_path: str,
        label: str,
        equation: str,
    ) -> Optional[str]:
        """Build a validation curve plot using ODE benchmark helpers when available."""
        import importlib.util

        import numpy as np
        import sympy as sp

        evaluator_path = Path(self.evaluation_file).resolve()
        benchmark_root = evaluator_path.parent.parent
        if str(benchmark_root) not in sys.path:
            sys.path.insert(0, str(benchmark_root))
        from pysr_harness import equation_session

        module_name = f"_skydiscover_checkpoint_plot_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            return None
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)

        load_all_datasets = getattr(evaluator, "load_all_datasets", None)
        if load_all_datasets is None:
            return None
        datasets = load_all_datasets()
        if not datasets:
            return None

        dataset_name, _X_train, X_val, _y_train, y_val = datasets[0]
        if X_val.shape[0] == 0:
            return None

        ode_owner = getattr(evaluator, "_base", evaluator)
        ode_predictions = getattr(ode_owner, "_ode_predictions", None)
        if ode_predictions is None:
            return None

        rhs_text = equation.split("=", 1)[1].strip() if equation.startswith("d(c)/dt") else equation
        feature_names = [f"x{i}" for i in range(X_val.shape[1])]
        expression = equation_session._as_sympy_expr(rhs_text, feature_names)
        feature_symbols = equation_session.feature_symbols(X_val.shape[1])
        rhs_fn = sp.lambdify(feature_symbols, expression, modules=["numpy"])
        y_pred = ode_predictions(rhs_fn, X_val, y_val, np.asarray([], dtype=float))

        curve_ids = self._validation_curve_ids(ode_owner, X_val)
        unique_curve_ids = list(dict.fromkeys(np.asarray(curve_ids, dtype=int)))
        panels = []
        for curve_id in unique_curve_ids:
            idx = np.flatnonzero(curve_ids == curve_id)
            if idx.size == 0:
                continue
            ordered = idx[np.argsort(X_val[idx, 0])]
            panels.append(
                {
                    "curve_id": int(curve_id),
                    "time": X_val[ordered, 0],
                    "true": y_val[ordered],
                    "pred": y_pred[ordered],
                    "features": X_val[ordered[0], 1:-1],
                }
            )
        if not panels:
            return None

        base_name = f"{label}_validation_curves"
        title = (
            f"{label}: {program.id} | score={get_score(program.metrics):.6g} | "
            f"dataset={dataset_name}"
        )
        try:
            return self._save_matplotlib_validation_plot(
                checkpoint_path,
                base_name,
                panels,
                title,
                program,
                dataset_name,
            )
        except ImportError:
            return self._save_svg_validation_plot(
                checkpoint_path,
                base_name,
                panels,
                title,
                program,
                dataset_name,
            )

    @staticmethod
    def _validation_curve_ids(ode_owner: object, X_val) -> object:
        """Return curve IDs remembered by the benchmark loader."""
        import numpy as np

        array_curve_ids = getattr(ode_owner, "_ARRAY_CURVE_IDS", {})
        curve_ids = array_curve_ids.get(id(X_val)) if isinstance(array_curve_ids, dict) else None
        if curve_ids is None or curve_ids.shape[0] != X_val.shape[0]:
            curve_ids = np.zeros(X_val.shape[0], dtype=int)
        return curve_ids

    def _save_matplotlib_validation_plot(
        self,
        checkpoint_path: str,
        base_name: str,
        panels: list[dict],
        title: str,
        program: Program,
        dataset_name: str,
    ) -> str:
        """Save a PNG validation curve plot with matplotlib."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_curves = len(panels)
        n_cols = min(4, max(1, math.ceil(math.sqrt(n_curves))))
        n_rows = math.ceil(n_curves / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.2 * n_cols, 3.2 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()

        for ax_idx, panel in enumerate(panels):
            ax = axes_flat[ax_idx]
            ax.scatter(
                panel["time"],
                panel["true"],
                s=7,
                alpha=0.55,
                label="validation points",
                color="#1f77b4",
            )
            ax.plot(
                panel["time"],
                panel["pred"],
                linewidth=1.7,
                label="ODE prediction",
                color="#d62728",
            )
            feature_text = ", ".join(f"x{i + 1}={value:g}" for i, value in enumerate(panel["features"]))
            ax.set_title(f"curve {panel['curve_id']} | {feature_text}", fontsize=8)
            ax.grid(True, alpha=0.25)
            if ax_idx % n_cols == 0:
                ax.set_ylabel("normalized concentration")
            if ax_idx // n_cols == n_rows - 1:
                ax.set_xlabel("normalized time")

        for ax in axes_flat[n_curves:]:
            ax.axis("off")

        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.98, 0.96])

        plot_path = os.path.join(checkpoint_path, f"{base_name}.png")
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        self._write_validation_plot_info(checkpoint_path, base_name, program, dataset_name, plot_path)
        return plot_path

    def _save_svg_validation_plot(
        self,
        checkpoint_path: str,
        base_name: str,
        panels: list[dict],
        title: str,
        program: Program,
        dataset_name: str,
    ) -> str:
        """Save a dependency-free SVG validation curve plot."""
        n_curves = len(panels)
        n_cols = min(4, max(1, math.ceil(math.sqrt(n_curves))))
        n_rows = math.ceil(n_curves / n_cols)
        panel_w = 260
        panel_h = 200
        margin = 42
        title_h = 44
        width = n_cols * panel_w
        height = title_h + n_rows * panel_h

        def scale_points(panel: dict, panel_idx: int, values_key: str) -> str:
            points = []
            xs = panel["time"]
            ys = panel[values_key]
            x_min, x_max = float(min(xs)), float(max(xs))
            y_min, y_max = 0.0, 1.0
            x_span = x_max - x_min if x_max > x_min else 1.0
            y_span = y_max - y_min
            col = panel_idx % n_cols
            row = panel_idx // n_cols
            x0 = col * panel_w + margin
            y0 = title_h + row * panel_h + 16
            plot_w = panel_w - margin - 16
            plot_h = panel_h - margin - 18
            for x_val, y_val in zip(xs, ys):
                px = x0 + (float(x_val) - x_min) / x_span * plot_w
                py = y0 + plot_h - (float(y_val) - y_min) / y_span * plot_h
                points.append(f"{px:.2f},{py:.2f}")
            return " ".join(points)

        elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="12" y="24" font-family="sans-serif" font-size="14">'
            f"{html.escape(title)}</text>",
        ]

        for idx, panel in enumerate(panels):
            col = idx % n_cols
            row = idx // n_cols
            x0 = col * panel_w + margin
            y0 = title_h + row * panel_h + 16
            plot_w = panel_w - margin - 16
            plot_h = panel_h - margin - 18
            feature_text = ", ".join(f"x{i + 1}={value:g}" for i, value in enumerate(panel["features"]))
            panel_title = html.escape(f"curve {panel['curve_id']} | {feature_text}")
            elements.extend(
                [
                    f'<text x="{col * panel_w + 8}" y="{title_h + row * panel_h + 12}" '
                    f'font-family="sans-serif" font-size="9">{panel_title}</text>',
                    f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" '
                    'fill="none" stroke="#cccccc"/>',
                    f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" '
                    f'y2="{y0 + plot_h}" stroke="#777777"/>',
                    f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" '
                    'stroke="#777777"/>',
                    f'<polyline points="{scale_points(panel, idx, "pred")}" fill="none" '
                    'stroke="#d62728" stroke-width="1.8"/>',
                    *[
                        f'<circle cx="{point.split(",")[0]}" cy="{point.split(",")[1]}" '
                        'r="1.4" fill="#1f77b4" fill-opacity="0.55"/>'
                        for point in scale_points(panel, idx, "true").split()
                    ],
                ]
            )
        elements.append("</svg>")

        plot_path = os.path.join(checkpoint_path, f"{base_name}.svg")
        with open(plot_path, "w") as f:
            f.write("\n".join(elements))
        self._write_validation_plot_info(checkpoint_path, base_name, program, dataset_name, plot_path)
        return plot_path

    @staticmethod
    def _write_validation_plot_info(
        checkpoint_path: str,
        base_name: str,
        program: Program,
        dataset_name: str,
        plot_path: str,
    ) -> None:
        info_path = os.path.join(checkpoint_path, f"{base_name}.json")
        with open(info_path, "w") as f:
            from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

            json.dump(
                {
                    "program_id": program.id,
                    "iteration_found": program.iteration_found,
                    "dataset": dataset_name,
                    "plot": plot_path,
                    "metrics": program.metrics,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
                cls=SafeJSONEncoder,
            )

    def _save_best_program(self, program: Program) -> None:
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        code_path = os.path.join(best_dir, f"best_program{self.file_extension}")
        with open(code_path, "w") as f:
            f.write(program.solution)

        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
                cls=SafeJSONEncoder,
            )

        if self.config.language == "image" and program.metadata:
            img = program.metadata.get("image_path")
            if img and os.path.exists(img):
                import shutil

                shutil.copy2(img, os.path.join(best_dir, "best_image" + os.path.splitext(img)[1]))

        logger.info(f"Best program saved to {best_dir}")
