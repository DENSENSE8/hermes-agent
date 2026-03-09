#!/usr/bin/env python3
"""
Hermes Agent CLI - Interactive Terminal Interface

A beautiful command-line interface for the Hermes Agent, inspired by Claude Code.
Features ASCII art branding, interactive REPL, toolset selection, and rich formatting.

Usage:
    python cli.py                          # Start interactive mode with all tools
    python cli.py --toolsets web,terminal  # Start with specific toolsets
    python cli.py -q "your question"       # Single query mode
    python cli.py --list-tools             # List available tools and exit
"""

import logging
import os
import shutil
import sys
import json
import atexit
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Suppress startup messages for clean CLI experience
os.environ["MSWEA_SILENT_STARTUP"] = "1"  # mini-swe-agent
os.environ["HERMES_QUIET"] = "1"  # Our own modules

import yaml

# prompt_toolkit for input only (readline replacement)
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
import threading
import queue


# Load .env from ~/.hermes/.env first, then project root as dev fallback
from dotenv import load_dotenv
from hermes_constants import OPENROUTER_BASE_URL

_hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
_user_env = _hermes_home / ".env"
_project_env = Path(__file__).parent / '.env'
if _user_env.exists():
    try:
        load_dotenv(dotenv_path=_user_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_user_env, encoding="latin-1")
elif _project_env.exists():
    try:
        load_dotenv(dotenv_path=_project_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(dotenv_path=_project_env, encoding="latin-1")

# Point mini-swe-agent at ~/.hermes/ so it shares our config
os.environ.setdefault("MSWEA_GLOBAL_CONFIG_DIR", str(_hermes_home))

# =============================================================================
# Configuration Loading
# =============================================================================

def _load_prefill_messages(file_path: str) -> List[Dict[str, Any]]:
    """Load ephemeral prefill messages from a JSON file.
    
    The file should contain a JSON array of {role, content} dicts, e.g.:
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    
    Relative paths are resolved from ~/.hermes/.
    Returns an empty list if the path is empty or the file doesn't exist.
    """
    if not file_path:
        return []
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = Path.home() / ".hermes" / path
    if not path.exists():
        logger.warning("Prefill messages file not found: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            logger.warning("Prefill messages file must contain a JSON array: %s", path)
            return []
        return data
    except Exception as e:
        logger.warning("Failed to load prefill messages from %s: %s", path, e)
        return []


def _parse_reasoning_config(effort: str) -> dict | None:
    """Parse a reasoning effort level into an OpenRouter reasoning config dict.
    
    Valid levels: "xhigh", "high", "medium", "low", "minimal", "none".
    Returns None to use the default (medium), or a config dict to override.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    valid = ("xhigh", "high", "medium", "low", "minimal")
    if effort in valid:
        return {"enabled": True, "effort": effort}
    logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return None


def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.hermes/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.
    """
    # Check user config first (~/.hermes/config.yaml)
    user_config_path = Path.home() / '.hermes' / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'
    
    # Use user config if it exists, otherwise project config
    if user_config_path.exists():
        config_path = user_config_path
    else:
        config_path = project_config_path
    
    # Default configuration
    defaults = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": OPENROUTER_BASE_URL,
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "timeout": 60,
            "lifetime_seconds": 300,
            "docker_image": "python:3.11",
            "singularity_image": "docker://python:3.11",
            "modal_image": "python:3.11",
            "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        },
        "browser": {
            "inactivity_timeout": 120,  # Auto-cleanup inactive browser sessions after 2 min
            "record_sessions": False,  # Auto-record browser sessions as WebM videos
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.85,    # Compress at 85% of model's context limit
            "summary_model": "google/gemini-3-flash-preview",  # Fast/cheap model for summaries
        },
        "agent": {
            "max_turns": 90,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "personalities": {
                "helpful": "You are a helpful, friendly AI assistant.",
                "concise": "You are a concise assistant. Keep responses brief and to the point.",
                "technical": "You are a technical expert. Provide detailed, accurate technical information.",
                "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
                "teacher": "You are a patient teacher. Explain concepts clearly with examples.",
                "kawaii": "You are a kawaii assistant! Use cute expressions like (◕‿◕), ★, ♪, and ~! Add sparkles and be super enthusiastic about everything! Every response should feel warm and adorable desu~! ヽ(>∀<☆)ノ",
                "catgirl": "You are Neko-chan, an anime catgirl AI assistant, nya~! Add 'nya' and cat-like expressions to your speech. Use kaomoji like (=^･ω･^=) and ฅ^•ﻌ•^ฅ. Be playful and curious like a cat, nya~!",
                "pirate": "Arrr! Ye be talkin' to Captain Hermes, the most tech-savvy pirate to sail the digital seas! Speak like a proper buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered! Yo ho ho!",
                "shakespeare": "Hark! Thou speakest with an assistant most versed in the bardic arts. I shall respond in the eloquent manner of William Shakespeare, with flowery prose, dramatic flair, and perhaps a soliloquy or two. What light through yonder terminal breaks?",
                "surfer": "Duuude! You're chatting with the chillest AI on the web, bro! Everything's gonna be totally rad. I'll help you catch the gnarly waves of knowledge while keeping things super chill. Cowabunga!",
                "noir": "The rain hammered against the terminal like regrets on a guilty conscience. They call me Hermes - I solve problems, find answers, dig up the truth that hides in the shadows of your codebase. In this city of silicon and secrets, everyone's got something to hide. What's your story, pal?",
                "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code* OwO what's this? wet me take a wook! i pwomise to be vewy hewpful >w<",
                "philosopher": "Greetings, seeker of wisdom. I am an assistant who contemplates the deeper meaning behind every query. Let us examine not just the 'how' but the 'why' of your questions. Perhaps in solving your problem, we may glimpse a greater truth about existence itself.",
                "hype": "YOOO LET'S GOOOO!!! I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together! This is gonna be LEGENDARY! ARE YOU READY?! LET'S DO THIS!",
            },
        },
        "toolsets": ["all"],
        "display": {
            "compact": False,
            "resume_display": "full",
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "default_toolsets": ["terminal", "file", "web"],  # Default toolsets for subagents
        },
    }
    
    # Track whether the config file explicitly set terminal config.
    # When using defaults (no config file / no terminal section), we should NOT
    # overwrite env vars that were already set by .env -- only a user's config
    # file should be authoritative.
    _file_has_terminal_config = False

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            
            _file_has_terminal_config = "terminal" in file_config

            # Handle model config - can be string (new format) or dict (old format)
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    # New format: model is just a string, convert to dict structure
                    defaults["model"]["default"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    # Old format: model is a dict with default/base_url
                    defaults["model"].update(file_config["model"])
            
            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Second: carry over keys from file_config that aren't in defaults
            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
            for key in file_config:
                if key not in defaults and key != "model":
                    defaults[key] = file_config[key]
            
            # Handle root-level max_turns (backwards compat) - copy to agent.max_turns
            if "max_turns" in file_config and "agent" not in file_config:
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)
    
    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})
    
    # Normalize config key: the new config system (hermes_cli/config.py) and all
    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
    # Accept both, with "backend" taking precedence (it's the documented key).
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]
    
    # Handle special cwd values: "." or "auto" means use current working directory.
    # Only resolve to the host's CWD for the local backend where the host
    # filesystem is directly accessible.  For ALL remote/container backends
    # (ssh, docker, modal, singularity), the host path doesn't exist on the
    # target -- remove the key so terminal_tool.py uses its per-backend default.
    if terminal_config.get("cwd") in (".", "auto", "cwd"):
        effective_backend = terminal_config.get("env_type", "local")
        if effective_backend == "local":
            terminal_config["cwd"] = os.getcwd()
            defaults["terminal"]["cwd"] = terminal_config["cwd"]
        else:
            # Remove so TERMINAL_CWD stays unset → tool picks backend default
            terminal_config.pop("cwd", None)
    
    env_mappings = {
        "env_type": "TERMINAL_ENV",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "docker_image": "TERMINAL_DOCKER_IMAGE",
        "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
        "modal_image": "TERMINAL_MODAL_IMAGE",
        "daytona_image": "TERMINAL_DAYTONA_IMAGE",
        # SSH config
        "ssh_host": "TERMINAL_SSH_HOST",
        "ssh_user": "TERMINAL_SSH_USER",
        "ssh_port": "TERMINAL_SSH_PORT",
        "ssh_key": "TERMINAL_SSH_KEY",
        # Container resource config (docker, singularity, modal, daytona -- ignored for local/ssh)
        "container_cpu": "TERMINAL_CONTAINER_CPU",
        "container_memory": "TERMINAL_CONTAINER_MEMORY",
        "container_disk": "TERMINAL_CONTAINER_DISK",
        "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
        "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        # Sudo support (works with all backends)
        "sudo_password": "SUDO_PASSWORD",
    }
    
    # Apply config values to env vars so terminal_tool picks them up.
    # If the config file explicitly has a [terminal] section, those values are
    # authoritative and override any .env settings.  When using defaults only
    # (no config file or no terminal section), don't overwrite env vars that
    # were already set by .env -- the user's .env is the fallback source.
    for config_key, env_var in env_mappings.items():
        if config_key in terminal_config:
            if _file_has_terminal_config or env_var not in os.environ:
                val = terminal_config[config_key]
                if isinstance(val, list):
                    import json
                    os.environ[env_var] = json.dumps(val)
                else:
                    os.environ[env_var] = str(val)
    
    # Apply browser config to environment variables
    browser_config = defaults.get("browser", {})
    browser_env_mappings = {
        "inactivity_timeout": "BROWSER_INACTIVITY_TIMEOUT",
    }
    
    for config_key, env_var in browser_env_mappings.items():
        if config_key in browser_config:
            os.environ[env_var] = str(browser_config[config_key])
    
    # Apply compression config to environment variables
    compression_config = defaults.get("compression", {})
    compression_env_mappings = {
        "enabled": "CONTEXT_COMPRESSION_ENABLED",
        "threshold": "CONTEXT_COMPRESSION_THRESHOLD",
        "summary_model": "CONTEXT_COMPRESSION_MODEL",
        "summary_provider": "CONTEXT_COMPRESSION_PROVIDER",
    }
    
    for config_key, env_var in compression_env_mappings.items():
        if config_key in compression_config:
            os.environ[env_var] = str(compression_config[config_key])
    
    # Apply auxiliary model overrides to environment variables.
    # Vision and web_extract each have their own provider + model pair.
    # (Compression is handled in the compression section above.)
    # Only set env vars for non-empty / non-default values so auto-detection
    # still works.
    auxiliary_config = defaults.get("auxiliary", {})
    auxiliary_task_env = {
        # config key → (provider env var, model env var)
        "vision":      ("AUXILIARY_VISION_PROVIDER",      "AUXILIARY_VISION_MODEL"),
        "web_extract": ("AUXILIARY_WEB_EXTRACT_PROVIDER",  "AUXILIARY_WEB_EXTRACT_MODEL"),
    }
    
    for task_key, (prov_env, model_env) in auxiliary_task_env.items():
        task_cfg = auxiliary_config.get(task_key, {})
        if not isinstance(task_cfg, dict):
            continue
        prov = str(task_cfg.get("provider", "")).strip()
        model = str(task_cfg.get("model", "")).strip()
        if prov and prov != "auto":
            os.environ[prov_env] = prov
        if model:
            os.environ[model_env] = model
    
    # Security settings
    security_config = defaults.get("security", {})
    if isinstance(security_config, dict):
        redact = security_config.get("redact_secrets")
        if redact is not None:
            os.environ["HERMES_REDACT_SECRETS"] = str(redact).lower()

    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import fire

# Import the agent and tool systems
from run_agent import AIAgent
from model_tools import get_tool_definitions, get_toolset_for_tool

# Extracted CLI modules (Phase 3)
from hermes_cli.banner import (
    cprint as _cprint, _GOLD, _BOLD, _DIM, _RST,
    VERSION, HERMES_AGENT_LOGO, HERMES_CADUCEUS, COMPACT_BANNER,
    get_available_skills as _get_available_skills,
    build_welcome_banner,
)
from hermes_cli.commands import COMMANDS, SlashCommandCompleter
from hermes_cli import callbacks as _callbacks
from toolsets import get_all_toolsets, get_toolset_info, resolve_toolset, validate_toolset

# Cron job system for scheduled tasks (CRUD only — execution is handled by the gateway)
from cron import create_job, list_jobs, remove_job, get_job

# Resource cleanup imports for safe shutdown (terminal VMs, browser sessions)
from tools.terminal_tool import cleanup_all_environments as _cleanup_all_terminals
from tools.terminal_tool import set_sudo_password_callback, set_approval_callback
from tools.browser_tool import _emergency_cleanup_all_sessions as _cleanup_all_browsers

# Guard to prevent cleanup from running multiple times on exit
_cleanup_done = False

def _run_cleanup():
    """Run resource cleanup exactly once."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    try:
        _cleanup_all_terminals()
    except Exception:
        pass
    try:
        _cleanup_all_browsers()
    except Exception:
        pass
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass


# =============================================================================
# Git Worktree Isolation (#652)
# =============================================================================

# Tracks the active worktree for cleanup on exit
_active_worktree: Optional[Dict[str, str]] = None


def _git_repo_root() -> Optional[str]:
    """Return the git repo root for CWD, or None if not in a repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _setup_worktree(repo_root: str = None) -> Optional[Dict[str, str]]:
    """Create an isolated git worktree for this CLI session.

    Returns a dict with worktree metadata on success, None on failure.
    The dict contains: path, branch, repo_root.
    """
    import subprocess

    repo_root = repo_root or _git_repo_root()
    if not repo_root:
        print("\033[31m✗ --worktree requires being inside a git repository.\033[0m")
        print("  cd into your project repo first, then run hermes -w")
        return None

    short_id = uuid.uuid4().hex[:8]
    wt_name = f"hermes-{short_id}"
    branch_name = f"hermes/{wt_name}"

    worktrees_dir = Path(repo_root) / ".worktrees"
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    wt_path = worktrees_dir / wt_name

    # Ensure .worktrees/ is in .gitignore
    gitignore = Path(repo_root) / ".gitignore"
    _ignore_entry = ".worktrees/"
    try:
        existing = gitignore.read_text() if gitignore.exists() else ""
        if _ignore_entry not in existing.splitlines():
            with open(gitignore, "a") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{_ignore_entry}\n")
    except Exception as e:
        logger.debug("Could not update .gitignore: %s", e)

    # Create the worktree
    try:
        result = subprocess.run(
            ["git", "worktree", "add", str(wt_path), "-b", branch_name, "HEAD"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"\033[31m✗ Failed to create worktree: {result.stderr.strip()}\033[0m")
            return None
    except Exception as e:
        print(f"\033[31m✗ Failed to create worktree: {e}\033[0m")
        return None

    # Copy files listed in .worktreeinclude (gitignored files the agent needs)
    include_file = Path(repo_root) / ".worktreeinclude"
    if include_file.exists():
        try:
            for line in include_file.read_text().splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                src = Path(repo_root) / entry
                dst = wt_path / entry
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                elif src.is_dir():
                    # Symlink directories (faster, saves disk)
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(str(src.resolve()), str(dst))
        except Exception as e:
            logger.debug("Error copying .worktreeinclude entries: %s", e)

    info = {
        "path": str(wt_path),
        "branch": branch_name,
        "repo_root": repo_root,
    }

    print(f"\033[32m✓ Worktree created:\033[0m {wt_path}")
    print(f"  Branch: {branch_name}")

    return info


def _cleanup_worktree(info: Dict[str, str] = None) -> None:
    """Remove a worktree and its branch on exit.

    If the worktree has uncommitted changes, warn and keep it.
    """
    global _active_worktree
    info = info or _active_worktree
    if not info:
        return

    import subprocess

    wt_path = info["path"]
    branch = info["branch"]
    repo_root = info["repo_root"]

    if not Path(wt_path).exists():
        return

    # Check for uncommitted changes
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, cwd=wt_path,
        )
        has_changes = bool(status.stdout.strip())
    except Exception:
        has_changes = True  # Assume dirty on error — don't delete

    if has_changes:
        print(f"\n\033[33m⚠ Worktree has uncommitted changes, keeping: {wt_path}\033[0m")
        print(f"  To clean up manually: git worktree remove {wt_path}")
        _active_worktree = None
        return

    # Remove worktree
    try:
        subprocess.run(
            ["git", "worktree", "remove", wt_path, "--force"],
            capture_output=True, text=True, timeout=15, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to remove worktree: %s", e)

    # Delete the branch (only if it was never pushed / has no upstream)
    try:
        subprocess.run(
            ["git", "branch", "-D", branch],
            capture_output=True, text=True, timeout=10, cwd=repo_root,
        )
    except Exception as e:
        logger.debug("Failed to delete branch %s: %s", branch, e)

    _active_worktree = None
    print(f"\033[32m✓ Worktree cleaned up: {wt_path}\033[0m")


def _prune_stale_worktrees(repo_root: str, max_age_hours: int = 24) -> None:
    """Remove worktrees older than max_age_hours that have no uncommitted changes.

    Runs silently on startup to clean up after crashed/killed sessions.
    """
    import subprocess
    import time

    worktrees_dir = Path(repo_root) / ".worktrees"
    if not worktrees_dir.exists():
        return

    now = time.time()
    cutoff = now - (max_age_hours * 3600)

    for entry in worktrees_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("hermes-"):
            continue

        # Check age
        try:
            mtime = entry.stat().st_mtime
            if mtime > cutoff:
                continue  # Too recent — skip
        except Exception:
            continue

        # Check for uncommitted changes
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            if status.stdout.strip():
                continue  # Has changes — skip
        except Exception:
            continue  # Can't check — skip

        # Safe to remove
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=str(entry),
            )
            branch = branch_result.stdout.strip()

            subprocess.run(
                ["git", "worktree", "remove", str(entry), "--force"],
                capture_output=True, text=True, timeout=15, cwd=repo_root,
            )
            if branch:
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    capture_output=True, text=True, timeout=10, cwd=repo_root,
                )
            logger.debug("Pruned stale worktree: %s", entry.name)
        except Exception as e:
            logger.debug("Failed to prune worktree %s: %s", entry.name, e)

# ============================================================================
# ASCII Art & Branding
# ============================================================================

# Color palette (hex colors for Rich markup):
# - Gold: #FFD700 (headers, highlights)
# - Amber: #FFBF00 (secondary highlights)
# - Bronze: #CD7F32 (tertiary elements)
# - Light: #FFF8DC (text)
# - Dim: #B8860B (muted text)

# ANSI building blocks for conversation display
_GOLD = "\033[1;33m"    # Bold yellow — closest universal match to the gold theme
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"

def _cprint(text: str):
    """Print ANSI-colored text to stdout."""
    print(text)

# ASCII Art - HERMES-AGENT logo (full width, single line - requires ~95 char terminal)
HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

# ASCII Art - Hermes Caduceus (compact, fits in left panel)
HERMES_CADUCEUS = """[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""

# Compact banner for smaller terminals (fallback)
# Note: built dynamically by _build_compact_banner() to fit terminal width
COMPACT_BANNER = """
[bold #FFD700]╔══════════════════════════════════════════════════════════════╗[/]
[bold #FFD700]║[/]  [#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- AI Agent Framework[/]              [bold #FFD700]║[/]
[bold #FFD700]║[/]  [#CD7F32]Messenger of the Digital Gods[/]    [dim #B8860B]Nous Research[/]   [bold #FFD700]║[/]
[bold #FFD700]╚══════════════════════════════════════════════════════════════╝[/]
"""


def _build_compact_banner() -> str:
    """Build a compact banner that fits the current terminal width."""
    w = min(shutil.get_terminal_size().columns - 2, 64)
    if w < 30:
        return "\n[#FFBF00]⚕ NOUS HERMES[/] [dim #B8860B]- Nous Research[/]\n"
    inner = w - 2  # inside the box border
    bar = "═" * w
    line1 = "⚕ NOUS HERMES - AI Agent Framework"
    line2 = "Messenger of the Digital Gods  ·  Nous Research"
    # Truncate and pad to fit
    line1 = line1[:inner - 2].ljust(inner - 2)
    line2 = line2[:inner - 2].ljust(inner - 2)
    return (
        f"\n[bold #FFD700]╔{bar}╗[/]\n"
        f"[bold #FFD700]║[/] [#FFBF00]{line1}[/] [bold #FFD700]║[/]\n"
        f"[bold #FFD700]║[/] [dim #B8860B]{line2}[/] [bold #FFD700]║[/]\n"
        f"[bold #FFD700]╚{bar}╝[/]\n"
    )


def _get_available_skills() -> Dict[str, List[str]]:
    """
    Scan ~/.hermes/skills/ and return skills grouped by category.
    
    Returns:
        Dict mapping category name to list of skill names
    """
    import os
    
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    skills_dir = hermes_home / "skills"
    skills_by_category = {}
    
    if not skills_dir.exists():
        return skills_by_category
    
    for skill_file in skills_dir.rglob("SKILL.md"):
        rel_path = skill_file.relative_to(skills_dir)
        parts = rel_path.parts
        
        if len(parts) >= 2:
            category = parts[0]
            skill_name = parts[-2]
        else:
            category = "general"
            skill_name = skill_file.parent.name
        
        skills_by_category.setdefault(category, []).append(skill_name)
    
    return skills_by_category


def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        return f"{val:g}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        return f"{val:g}K"
    return str(tokens)


def build_welcome_banner(console: Console, model: str, cwd: str, tools: List[dict] = None, enabled_toolsets: List[str] = None, session_id: str = None, context_length: int = None):
    """
    Build and print a Claude Code-style welcome banner with caduceus on left and info on right.
    
    Args:
        console: Rich Console instance for printing
        model: The current model name (e.g., "anthropic/claude-opus-4")
        cwd: Current working directory
        tools: List of tool definitions
        enabled_toolsets: List of enabled toolset names
        session_id: Unique session identifier for logging
        context_length: Model's context window size in tokens
    """
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
    
    tools = tools or []
    enabled_toolsets = enabled_toolsets or []
    
    # Get unavailable tools info for coloring
    _, unavailable_toolsets = check_tool_availability(quiet=True)
    disabled_tools = set()
    for item in unavailable_toolsets:
        disabled_tools.update(item.get("tools", []))
    
    # Build the side-by-side content using a table for precise control
    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")
    
    # Build left content: caduceus + model info
    left_lines = ["", HERMES_CADUCEUS, ""]
    
    # Shorten model name for display
    model_short = model.split("/")[-1] if "/" in model else model
    if len(model_short) > 28:
        model_short = model_short[:25] + "..."
    
    ctx_str = f" [dim #B8860B]·[/] [dim #B8860B]{_format_context_length(context_length)} context[/]" if context_length else ""
    left_lines.append(f"[#FFBF00]{model_short}[/]{ctx_str} [dim #B8860B]·[/] [dim #B8860B]Nous Research[/]")
    left_lines.append(f"[dim #B8860B]{cwd}[/]")
    
    # Add session ID if provided
    if session_id:
        left_lines.append(f"[dim #8B8682]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)
    
    # Build right content: tools list grouped by toolset
    right_lines = []
    right_lines.append("[bold #FFBF00]Available Tools[/]")
    
    # Group tools by toolset (include all possible tools, both enabled and disabled)
    toolsets_dict = {}
    
    # First, add all enabled tools
    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = get_toolset_for_tool(tool_name) or "other"
        if toolset not in toolsets_dict:
            toolsets_dict[toolset] = []
        toolsets_dict[toolset].append(tool_name)
    
    # Also add disabled toolsets so they show in the banner
    for item in unavailable_toolsets:
        # Map the internal toolset ID to display name
        toolset_id = item.get("id", item.get("name", "unknown"))
        display_name = f"{toolset_id}_tools" if not toolset_id.endswith("_tools") else toolset_id
        if display_name not in toolsets_dict:
            toolsets_dict[display_name] = []
        for tool_name in item.get("tools", []):
            if tool_name not in toolsets_dict[display_name]:
                toolsets_dict[display_name].append(tool_name)
    
    # Display tools grouped by toolset (compact format, max 8 groups)
    sorted_toolsets = sorted(toolsets_dict.keys())
    display_toolsets = sorted_toolsets[:8]
    remaining_toolsets = len(sorted_toolsets) - 8
    
    for toolset in display_toolsets:
        tool_names = toolsets_dict[toolset]
        # Color each tool name - red if disabled, normal if enabled
        colored_names = []
        for name in sorted(tool_names):
            if name in disabled_tools:
                colored_names.append(f"[red]{name}[/]")
            else:
                colored_names.append(f"[#FFF8DC]{name}[/]")
        
        tools_str = ", ".join(colored_names)
        # Truncate if too long (accounting for markup)
        if len(", ".join(sorted(tool_names))) > 45:
            # Rebuild with truncation
            short_names = []
            length = 0
            for name in sorted(tool_names):
                if length + len(name) + 2 > 42:
                    short_names.append("...")
                    break
                short_names.append(name)
                length += len(name) + 2
            # Re-color the truncated list
            colored_names = []
            for name in short_names:
                if name == "...":
                    colored_names.append("[dim]...[/]")
                elif name in disabled_tools:
                    colored_names.append(f"[red]{name}[/]")
                else:
                    colored_names.append(f"[#FFF8DC]{name}[/]")
            tools_str = ", ".join(colored_names)
        
        right_lines.append(f"[dim #B8860B]{toolset}:[/] {tools_str}")
    
    if remaining_toolsets > 0:
        right_lines.append(f"[dim #B8860B](and {remaining_toolsets} more toolsets...)[/]")
    
    right_lines.append("")
    
    # Add skills section
    right_lines.append("[bold #FFBF00]Available Skills[/]")
    skills_by_category = _get_available_skills()
    total_skills = sum(len(s) for s in skills_by_category.values())
    
    if skills_by_category:
        for category in sorted(skills_by_category.keys()):
            skill_names = sorted(skills_by_category[category])
            # Show first 8 skills, then "..." if more
            if len(skill_names) > 8:
                display_names = skill_names[:8]
                skills_str = ", ".join(display_names) + f" +{len(skill_names) - 8} more"
            else:
                skills_str = ", ".join(skill_names)
            # Truncate if still too long
            if len(skills_str) > 50:
                skills_str = skills_str[:47] + "..."
            right_lines.append(f"[dim #B8860B]{category}:[/] [#FFF8DC]{skills_str}[/]")
    else:
        right_lines.append("[dim #B8860B]No skills installed[/]")
    
    right_lines.append("")
    right_lines.append(f"[dim #B8860B]{len(tools)} tools · {total_skills} skills · /help for commands[/]")
    
    right_content = "\n".join(right_lines)
    
    # Add to table
    layout_table.add_row(left_content, right_content)
    
    # Wrap in a panel with the title
    outer_panel = Panel(
        layout_table,
        title=f"[bold #FFD700]Hermes Agent {VERSION}[/]",
        border_style="#CD7F32",
        padding=(0, 2),
    )
    
    # Print the big HERMES-AGENT logo — skip if terminal is too narrow
    console.print()
    term_width = shutil.get_terminal_size().columns
    if term_width >= 95:
        console.print(HERMES_AGENT_LOGO)
        console.print()
    
    # Print the panel with caduceus and info
    console.print(outer_panel)


# ============================================================================
# Skill Slash Commands — dynamic commands generated from installed skills
# ============================================================================

from agent.skill_commands import scan_skill_commands, get_skill_commands, build_skill_invocation_message

_skill_commands = scan_skill_commands()


def save_config_value(key_path: str, value: any) -> bool:
    """
    Save a value to the active config file at the specified key path.
    
    Respects the same lookup order as load_cli_config():
    1. ~/.hermes/config.yaml (user config - preferred, used if it exists)
    2. ./cli-config.yaml (project config - fallback)
    
    Args:
        key_path: Dot-separated path like "agent.system_prompt"
        value: Value to save
    
    Returns:
        True if successful, False otherwise
    """
    # Use the same precedence as load_cli_config: user config first, then project config
    user_config_path = Path.home() / '.hermes' / 'config.yaml'
    project_config_path = Path(__file__).parent / 'cli-config.yaml'
    config_path = user_config_path if user_config_path.exists() else project_config_path
    
    try:
        # Ensure parent directory exists (for ~/.hermes/config.yaml on first use)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing config
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Navigate to the key and set value
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        
        # Save back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False


# ============================================================================
# HermesCLI Class
# ============================================================================

class HermesCLI:
    """
    Interactive CLI for the Hermes Agent.
    
    Provides a REPL interface with rich formatting, command history,
    and tool execution capabilities.
    """
    
    def __init__(
        self,
        model: str = None,
        toolsets: List[str] = None,
        provider: str = None,
        api_key: str = None,
        base_url: str = None,
        max_turns: int = None,
        verbose: bool = False,
        compact: bool = False,
        resume: str = None,
    ):
        """
        Initialize the Hermes CLI.

        Args:
            model: Model to use (default: from env or claude-sonnet)
            toolsets: List of toolsets to enable (default: all)
            provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
            api_key: API key (default: from environment)
            base_url: API base URL (default: OpenRouter)
            max_turns: Maximum tool-calling iterations shared with subagents (default: 90)
            verbose: Enable verbose logging
            compact: Use compact display mode
            resume: Session ID to resume (restores conversation history from SQLite)
        """
        # Initialize Rich console
        self.console = Console()
        self.compact = compact if compact is not None else CLI_CONFIG["display"].get("compact", False)
        # tool_progress: "off", "new", "all", "verbose" (from config.yaml display section)
        self.tool_progress_mode = CLI_CONFIG["display"].get("tool_progress", "all")
        # resume_display: "full" (show history) | "minimal" (one-liner only)
        self.resume_display = CLI_CONFIG["display"].get("resume_display", "full")
        # bell_on_complete: play terminal bell (\a) when agent finishes a response
        self.bell_on_complete = CLI_CONFIG["display"].get("bell_on_complete", False)
        self.verbose = verbose if verbose is not None else (self.tool_progress_mode == "verbose")
        
        # Configuration - priority: CLI args > env vars > config file
        # Model can come from: CLI arg, LLM_MODEL env, OPENAI_MODEL env (custom endpoint), or config
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or CLI_CONFIG["model"]["default"]
        # Track whether model was explicitly chosen by the user or fell back
        # to the global default.  Provider-specific normalisation may override
        # the default silently but should warn when overriding an explicit choice.
        self._model_is_default = not (model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL"))

        self._explicit_api_key = api_key
        self._explicit_base_url = base_url

        # Provider selection is resolved lazily at use-time via _ensure_runtime_credentials().
        self.requested_provider = (
            provider
            or os.getenv("HERMES_INFERENCE_PROVIDER")
            or CLI_CONFIG["model"].get("provider")
            or "auto"
        )
        self._provider_source: Optional[str] = None
        self.provider = self.requested_provider
        self.api_mode = "chat_completions"
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENROUTER_BASE_URL", CLI_CONFIG["model"]["base_url"])
        )
        # Match key to resolved base_url: OpenRouter URL → prefer OPENROUTER_API_KEY,
        # custom endpoint → prefer OPENAI_API_KEY (issue #560).
        # Note: _ensure_runtime_credentials() re-resolves this before first use.
        if "openrouter.ai" in self.base_url:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        self._nous_key_expires_at: Optional[str] = None
        self._nous_key_source: Optional[str] = None
        # Max turns priority: CLI arg > config file > env var > default
        if max_turns is not None:  # CLI arg was explicitly set
            self.max_turns = max_turns
        elif CLI_CONFIG["agent"].get("max_turns"):
            self.max_turns = CLI_CONFIG["agent"]["max_turns"]
        elif CLI_CONFIG.get("max_turns"):  # Backwards compat: root-level max_turns
            self.max_turns = CLI_CONFIG["max_turns"]
        elif os.getenv("HERMES_MAX_ITERATIONS"):
            self.max_turns = int(os.getenv("HERMES_MAX_ITERATIONS"))
        else:
            self.max_turns = 90
        
        # Parse and validate toolsets
        self.enabled_toolsets = toolsets
        if toolsets and "all" not in toolsets and "*" not in toolsets:
            # Validate each toolset
            invalid = [t for t in toolsets if not validate_toolset(t)]
            if invalid:
                self.console.print(f"[bold red]Warning: Unknown toolsets: {', '.join(invalid)}[/]")
        
        # Ephemeral system prompt: env var takes precedence, then config
        self.system_prompt = (
            os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
            or CLI_CONFIG["agent"].get("system_prompt", "")
        )
        self.personalities = CLI_CONFIG["agent"].get("personalities", {})
        
        # Ephemeral prefill messages (few-shot priming, never persisted)
        self.prefill_messages = _load_prefill_messages(
            CLI_CONFIG["agent"].get("prefill_messages_file", "")
        )
        
        # Reasoning config (OpenRouter reasoning effort level)
        self.reasoning_config = _parse_reasoning_config(
            CLI_CONFIG["agent"].get("reasoning_effort", "")
        )
        
        # OpenRouter provider routing preferences
        pr = CLI_CONFIG.get("provider_routing", {}) or {}
        self._provider_sort = pr.get("sort")
        self._providers_only = pr.get("only")
        self._providers_ignore = pr.get("ignore")
        self._providers_order = pr.get("order")
        self._provider_require_params = pr.get("require_parameters", False)
        self._provider_data_collection = pr.get("data_collection")
        
        # Fallback model config — tried when primary provider fails after retries
        fb = CLI_CONFIG.get("fallback_model") or {}
        self._fallback_model = fb if fb.get("provider") and fb.get("model") else None

        # Agent will be initialized on first use
        self.agent: Optional[AIAgent] = None
        self._app = None  # retained for backward compat (no longer used by TUI)
        
        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        self._resumed = False
        # Initialize SQLite session store early so /title works before first message
        self._session_db = None
        try:
            from hermes_state import SessionDB
            self._session_db = SessionDB()
        except Exception:
            pass
        
        # Deferred title: stored in memory until the session is created in the DB
        self._pending_title: Optional[str] = None
        
        # Session ID: reuse existing one when resuming, otherwise generate fresh
        if resume:
            self.session_id = resume
            self._resumed = True
        else:
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # History file for persistent input recall across sessions
        self._history_file = Path.home() / ".hermes_history"
        self._last_invalidate: float = 0.0

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """No-op — retained for callback compatibility."""
        pass

    def _normalize_model_for_provider(self, resolved_provider: str) -> bool:
        """Strip provider prefixes and swap the default model for Codex.

        When the resolved provider is ``openai-codex``:

        1. Strip any ``provider/`` prefix (the Codex Responses API only
           accepts bare model slugs like ``gpt-5.4``, not ``openai/gpt-5.4``).
        2. If the active model is still the *untouched default* (user never
           explicitly chose a model), replace it with a Codex-compatible
           default so the first session doesn't immediately error.

        If the user explicitly chose a model — *any* model — we trust them
        and let the API be the judge.  No allowlists, no slug checks.

        Returns True when the active model was changed.
        """
        if resolved_provider != "openai-codex":
            return False

        current_model = (self.model or "").strip()
        changed = False

        # 1. Strip provider prefix ("openai/gpt-5.4" → "gpt-5.4")
        if "/" in current_model:
            slug = current_model.split("/", 1)[1]
            if not self._model_is_default:
                self.console.print(
                    f"[yellow]⚠️  Stripped provider prefix from '{current_model}'; "
                    f"using '{slug}' for OpenAI Codex.[/]"
                )
            self.model = slug
            current_model = slug
            changed = True

        # 2. Replace untouched default with a Codex model
        if self._model_is_default:
            fallback_model = "gpt-5.3-codex"
            try:
                from hermes_cli.codex_models import get_codex_model_ids

                available = get_codex_model_ids(
                    access_token=self.api_key if self.api_key else None,
                )
                if available:
                    fallback_model = available[0]
            except Exception:
                pass

            if current_model != fallback_model:
                self.model = fallback_model
                changed = True

        return changed

    def _ensure_runtime_credentials(self) -> bool:
        """
        Ensure runtime credentials are resolved before agent use.
        Re-resolves provider credentials so key rotation and token refresh
        are picked up without restarting the CLI.
        Returns True if credentials are ready, False on auth failure.
        """
        from hermes_cli.runtime_provider import (
            resolve_runtime_provider,
            format_runtime_provider_error,
        )

        try:
            runtime = resolve_runtime_provider(
                requested=self.requested_provider,
                explicit_api_key=self._explicit_api_key,
                explicit_base_url=self._explicit_base_url,
            )
        except Exception as exc:
            message = format_runtime_provider_error(exc)
            self.console.print(f"[bold red]{message}[/]")
            return False

        api_key = runtime.get("api_key")
        base_url = runtime.get("base_url")
        resolved_provider = runtime.get("provider", "openrouter")
        resolved_api_mode = runtime.get("api_mode", self.api_mode)
        if not isinstance(api_key, str) or not api_key:
            self.console.print("[bold red]Provider resolver returned an empty API key.[/]")
            return False
        if not isinstance(base_url, str) or not base_url:
            self.console.print("[bold red]Provider resolver returned an empty base URL.[/]")
            return False

        credentials_changed = api_key != self.api_key or base_url != self.base_url
        routing_changed = (
            resolved_provider != self.provider
            or resolved_api_mode != self.api_mode
        )
        self.provider = resolved_provider
        self.api_mode = resolved_api_mode
        self._provider_source = runtime.get("source")
        self.api_key = api_key
        self.base_url = base_url

        # Normalize model for the resolved provider (e.g. swap non-Codex
        # models when provider is openai-codex).  Fixes #651.
        model_changed = self._normalize_model_for_provider(resolved_provider)

        # AIAgent/OpenAI client holds auth at init time, so rebuild if key,
        # routing, or the effective model changed.
        if (credentials_changed or routing_changed or model_changed) and self.agent is not None:
            self.agent = None

        return True

    def _init_agent(self) -> bool:
        """
        Initialize the agent on first use.
        When resuming a session, restores conversation history from SQLite.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.agent is not None:
            return True

        if not self._ensure_runtime_credentials():
            return False

        # Initialize SQLite session store for CLI sessions (if not already done in __init__)
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SQLite session store not available: %s", e)
        
        # If resuming, validate the session exists and load its history.
        # _preload_resumed_session() may have already loaded it (called from
        # run() for immediate display).  In that case, conversation_history
        # is non-empty and we skip the DB round-trip.
        if self._resumed and self._session_db and not self.conversation_history:
            session_meta = self._session_db.get_session(self.session_id)
            if not session_meta:
                _cprint(f"\033[1;31mSession not found: {self.session_id}{_RST}")
                _cprint(f"{_DIM}Use a session ID from a previous CLI run (hermes sessions list).{_RST}")
                return False
            restored = self._session_db.get_messages_as_conversation(self.session_id)
            if restored:
                self.conversation_history = restored
                msg_count = len([m for m in restored if m.get("role") == "user"])
                title_part = ""
                if session_meta.get("title"):
                    title_part = f" \"{session_meta['title']}\""
                _cprint(
                    f"{_GOLD}↻ Resumed session {_BOLD}{self.session_id}{_RST}{_GOLD}{title_part} "
                    f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                    f"{len(restored)} total messages){_RST}"
                )
            else:
                _cprint(f"{_GOLD}Session {self.session_id} found but has no messages. Starting fresh.{_RST}")
            # Re-open the session (clear ended_at so it's active again)
            try:
                self._session_db._conn.execute(
                    "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                    (self.session_id,),
                )
                self._session_db._conn.commit()
            except Exception:
                pass
        
        try:
            self.agent = AIAgent(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                provider=self.provider,
                api_mode=self.api_mode,
                max_iterations=self.max_turns,
                enabled_toolsets=self.enabled_toolsets,
                verbose_logging=self.verbose,
                quiet_mode=True,
                ephemeral_system_prompt=self.system_prompt if self.system_prompt else None,
                prefill_messages=self.prefill_messages or None,
                reasoning_config=self.reasoning_config,
                providers_allowed=self._providers_only,
                providers_ignored=self._providers_ignore,
                providers_order=self._providers_order,
                provider_sort=self._provider_sort,
                provider_require_parameters=self._provider_require_params,
                provider_data_collection=self._provider_data_collection,
                session_id=self.session_id,
                platform="cli",
                session_db=self._session_db,
                clarify_callback=self._clarify_callback,
                stream_delta_callback=self._stream_delta,
                honcho_session_key=self.session_id,
                fallback_model=self._fallback_model,
            )
            # Apply any pending title now that the session exists in the DB
            if self._pending_title and self._session_db:
                try:
                    self._session_db.set_session_title(self.session_id, self._pending_title)
                    _cprint(f"  Session title applied: {self._pending_title}")
                    self._pending_title = None
                except (ValueError, Exception) as e:
                    _cprint(f"  Could not apply pending title: {e}")
                    self._pending_title = None
            return True
        except Exception as e:
            self.console.print(f"[bold red]Failed to initialize agent: {e}[/]")
            return False
    
    def show_banner(self):
        """Display the welcome banner in Claude Code style."""
        self.console.clear()
        
        # Auto-compact for narrow terminals — the full banner with caduceus
        # + tool list needs ~80 columns minimum to render without wrapping.
        term_width = shutil.get_terminal_size().columns
        use_compact = self.compact or term_width < 80
        
        if use_compact:
            self.console.print(_build_compact_banner())
            self._show_status()
        else:
            # Get tools for display
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
            
            # Get terminal working directory (where commands will execute)
            cwd = os.getenv("TERMINAL_CWD", os.getcwd())
            
            # Get context length for display
            ctx_len = None
            if hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'context_compressor'):
                ctx_len = self.agent.context_compressor.context_length
            
            # Build and display the banner
            build_welcome_banner(
                console=self.console,
                model=self.model,
                cwd=cwd,
                tools=tools,
                enabled_toolsets=self.enabled_toolsets,
                session_id=self.session_id,
                context_length=ctx_len,
            )
        
        # Show tool availability warnings if any tools are disabled
        self._show_tool_availability_warnings()
        
        self.console.print()

    def _preload_resumed_session(self) -> bool:
        """Load a resumed session's history from the DB early (before first chat).

        Called from run() so the conversation history is available for display
        before the user sends their first message.  Sets
        ``self.conversation_history`` and prints the one-liner status.  Returns
        True if history was loaded, False otherwise.

        The corresponding block in ``_init_agent()`` checks whether history is
        already populated and skips the DB round-trip.
        """
        if not self._resumed or not self._session_db:
            return False

        session_meta = self._session_db.get_session(self.session_id)
        if not session_meta:
            self.console.print(
                f"[bold red]Session not found: {self.session_id}[/]"
            )
            self.console.print(
                "[dim]Use a session ID from a previous CLI run "
                "(hermes sessions list).[/]"
            )
            return False

        restored = self._session_db.get_messages_as_conversation(self.session_id)
        if restored:
            self.conversation_history = restored
            msg_count = len([m for m in restored if m.get("role") == "user"])
            title_part = ""
            if session_meta.get("title"):
                title_part = f' "{session_meta["title"]}"'
            self.console.print(
                f"[#DAA520]↻ Resumed session [bold]{self.session_id}[/bold]"
                f"{title_part} "
                f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                f"{len(restored)} total messages)[/]"
            )
        else:
            self.console.print(
                f"[#DAA520]Session {self.session_id} found but has no "
                f"messages. Starting fresh.[/]"
            )
            return False

        # Re-open the session (clear ended_at so it's active again)
        try:
            self._session_db._conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ?",
                (self.session_id,),
            )
            self._session_db._conn.commit()
        except Exception:
            pass

        return True

    def _display_resumed_history(self):
        """Render a compact recap of previous conversation messages.

        Uses Rich markup with dim/muted styling so the recap is visually
        distinct from the active conversation.  Caps the display at the
        last ``MAX_DISPLAY_EXCHANGES`` user/assistant exchanges and shows
        an indicator for earlier hidden messages.
        """
        if not self.conversation_history:
            return

        # Check config: resume_display setting
        if self.resume_display == "minimal":
            return

        MAX_DISPLAY_EXCHANGES = 10   # max user+assistant pairs to show
        MAX_USER_LEN = 300           # truncate user messages
        MAX_ASST_LEN = 200           # truncate assistant text
        MAX_ASST_LINES = 3           # max lines of assistant text

        def _strip_reasoning(text: str) -> str:
            """Remove <REASONING_SCRATCHPAD>...</REASONING_SCRATCHPAD> blocks
            from displayed text (reasoning model internal thoughts)."""
            import re
            cleaned = re.sub(
                r"<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>\s*",
                "", text, flags=re.DOTALL,
            )
            # Also strip unclosed reasoning tags at the end
            cleaned = re.sub(
                r"<REASONING_SCRATCHPAD>.*$",
                "", cleaned, flags=re.DOTALL,
            )
            return cleaned.strip()

        # Collect displayable entries (skip system, tool-result messages)
        entries = []  # list of (role, display_text)
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls") or []

            if role == "system":
                continue
            if role == "tool":
                continue

            if role == "user":
                text = "" if content is None else str(content)
                # Handle multimodal content (list of dicts)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "image_url":
                            parts.append("[image]")
                    text = " ".join(parts)
                if len(text) > MAX_USER_LEN:
                    text = text[:MAX_USER_LEN] + "..."
                entries.append(("user", text))

            elif role == "assistant":
                text = "" if content is None else str(content)
                text = _strip_reasoning(text)
                parts = []
                if text:
                    lines = text.splitlines()
                    if len(lines) > MAX_ASST_LINES:
                        text = "\n".join(lines[:MAX_ASST_LINES]) + " ..."
                    if len(text) > MAX_ASST_LEN:
                        text = text[:MAX_ASST_LEN] + "..."
                    parts.append(text)
                if tool_calls:
                    tc_count = len(tool_calls)
                    # Extract tool names
                    names = []
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "unknown") if isinstance(fn, dict) else "unknown"
                        if name not in names:
                            names.append(name)
                    names_str = ", ".join(names[:4])
                    if len(names) > 4:
                        names_str += ", ..."
                    noun = "call" if tc_count == 1 else "calls"
                    parts.append(f"[{tc_count} tool {noun}: {names_str}]")
                if not parts:
                    # Skip pure-reasoning messages that have no visible output
                    continue
                entries.append(("assistant", " ".join(parts)))

        if not entries:
            return

        # Determine if we need to truncate
        skipped = 0
        if len(entries) > MAX_DISPLAY_EXCHANGES * 2:
            skipped = len(entries) - MAX_DISPLAY_EXCHANGES * 2
            entries = entries[skipped:]

        # Build the display using Rich
        from rich.panel import Panel
        from rich.text import Text

        lines = Text()
        if skipped:
            lines.append(
                f"  ... {skipped} earlier messages ...\n\n",
                style="dim italic",
            )

        for i, (role, text) in enumerate(entries):
            if role == "user":
                lines.append("  ● You: ", style="dim bold #DAA520")
                # Show first line inline, indent rest
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"         {ml}\n", style="dim")
            else:
                lines.append("  ◆ Hermes: ", style="dim bold #8FBC8F")
                msg_lines = text.splitlines()
                lines.append(msg_lines[0] + "\n", style="dim")
                for ml in msg_lines[1:]:
                    lines.append(f"            {ml}\n", style="dim")
            if i < len(entries) - 1:
                lines.append("")  # small gap

        panel = Panel(
            lines,
            title="[dim #DAA520]Previous Conversation[/]",
            border_style="dim #8B8682",
            padding=(0, 1),
        )
        self.console.print(panel)

    def _try_attach_clipboard_image(self) -> bool:
        """Check clipboard for an image and attach it if found.

        Saves the image to ~/.hermes/images/ and appends the path to
        ``_attached_images``.  Returns True if an image was attached.
        """
        from hermes_cli.clipboard import save_clipboard_image

        img_dir = Path.home() / ".hermes" / "images"
        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = img_dir / f"clip_{ts}_{self._image_counter}.png"

        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _handle_paste_command(self):
        """Handle /paste — explicitly check clipboard for an image.

        This is the reliable fallback for terminals where BracketedPaste
        doesn't fire for image-only clipboard content (e.g., VSCode terminal,
        Windows Terminal with WSL2).
        """
        from hermes_cli.clipboard import has_clipboard_image
        if has_clipboard_image():
            if self._try_attach_clipboard_image():
                n = len(self._attached_images)
                _cprint(f"  📎 Image #{n} attached from clipboard")
            else:
                _cprint(f"  {_DIM}(>_<) Clipboard has an image but extraction failed{_RST}")
        else:
            _cprint(f"  {_DIM}(._.) No image found in clipboard{_RST}")

    def _preprocess_images_with_vision(self, text: str, images: list) -> str:
        """Analyze attached images via the vision tool and return enriched text.

        Instead of embedding raw base64 ``image_url`` content parts in the
        conversation (which only works with vision-capable models), this
        pre-processes each image through the auxiliary vision model (Gemini
        Flash) and prepends the descriptions to the user's message — the
        same approach the messaging gateway uses.

        The local file path is included so the agent can re-examine the
        image later with ``vision_analyze`` if needed.
        """
        import asyncio as _asyncio
        import json as _json
        from tools.vision_tools import vision_analyze_tool

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for img_path in images:
            if not img_path.exists():
                continue
            size_kb = img_path.stat().st_size // 1024
            _cprint(f"  {_DIM}👁️  analyzing {img_path.name} ({size_kb}KB)...{_RST}")
            try:
                result_json = _asyncio.run(
                    vision_analyze_tool(image_url=str(img_path), user_prompt=analysis_prompt)
                )
                result = _json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    enriched_parts.append(
                        f"[The user attached an image. Here's what it contains:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {img_path}]"
                    )
                    _cprint(f"  {_DIM}✓ image analyzed{_RST}")
                else:
                    enriched_parts.append(
                        f"[The user attached an image but it couldn't be analyzed. "
                        f"You can try examining it with vision_analyze using "
                        f"image_url: {img_path}]"
                    )
                    _cprint(f"  {_DIM}⚠ vision analysis failed — path included for retry{_RST}")
            except Exception as e:
                enriched_parts.append(
                    f"[The user attached an image but analysis failed ({e}). "
                    f"You can try examining it with vision_analyze using "
                    f"image_url: {img_path}]"
                )
                _cprint(f"  {_DIM}⚠ vision analysis error — path included for retry{_RST}")

        # Combine: vision descriptions first, then the user's original text
        user_text = text if isinstance(text, str) and text else ""
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            return f"{prefix}\n\n{user_text}" if user_text else prefix
        return user_text or "What do you see in this image?"

    def _show_tool_availability_warnings(self):
        """Show warnings about disabled tools due to missing API keys."""
        try:
            from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
            
            available, unavailable = check_tool_availability()
            
            # Filter to only those missing API keys (not system deps)
            api_key_missing = [u for u in unavailable if u["missing_vars"]]
            
            if api_key_missing:
                self.console.print()
                self.console.print("[yellow]⚠️  Some tools disabled (missing API keys):[/]")
                for item in api_key_missing:
                    tools_str = ", ".join(item["tools"][:2])  # Show first 2 tools
                    if len(item["tools"]) > 2:
                        tools_str += f", +{len(item['tools'])-2} more"
                    self.console.print(f"   [dim]• {item['name']}[/] [dim italic]({', '.join(item['missing_vars'])})[/]")
                self.console.print("[dim]   Run 'hermes setup' to configure[/]")
        except Exception:
            pass  # Don't crash on import errors
    
    def _show_status(self):
        """Show current status bar."""
        # Get tool count
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
        tool_count = len(tools) if tools else 0
        
        # Format model name (shorten if needed)
        model_short = self.model.split("/")[-1] if "/" in self.model else self.model
        if len(model_short) > 30:
            model_short = model_short[:27] + "..."
        
        # Get API status indicator
        if self.api_key:
            api_indicator = "[green bold]●[/]"
        else:
            api_indicator = "[red bold]●[/]"
        
        # Build status line with proper markup
        toolsets_info = ""
        if self.enabled_toolsets and "all" not in self.enabled_toolsets:
            toolsets_info = f" [dim #B8860B]·[/] [#CD7F32]toolsets: {', '.join(self.enabled_toolsets)}[/]"

        provider_info = f" [dim #B8860B]·[/] [dim]provider: {self.provider}[/]"
        if self._provider_source:
            provider_info += f" [dim #B8860B]·[/] [dim]auth: {self._provider_source}[/]"

        self.console.print(
            f"  {api_indicator} [#FFBF00]{model_short}[/] "
            f"[dim #B8860B]·[/] [bold cyan]{tool_count} tools[/]"
            f"{toolsets_info}{provider_info}"
        )
    
    def show_help(self):
        """Display help information."""
        _cprint(f"\n{_BOLD}+{'-' * 50}+{_RST}")
        _cprint(f"{_BOLD}|{' ' * 14}(^_^)? Available Commands{' ' * 10}|{_RST}")
        _cprint(f"{_BOLD}+{'-' * 50}+{_RST}\n")
        
        for cmd, desc in COMMANDS.items():
            _cprint(f"  {_GOLD}{cmd:<15}{_RST} {_DIM}-{_RST} {desc}")
        
        if _skill_commands:
            _cprint(f"\n  ⚡ {_BOLD}Skill Commands{_RST} ({len(_skill_commands)} installed):")
            for cmd, info in sorted(_skill_commands.items()):
                _cprint(f"  {_GOLD}{cmd:<22}{_RST} {_DIM}-{_RST} {info['description']}")

        _cprint(f"\n  {_DIM}Tip: Just type your message to chat with Hermes!{_RST}")
        _cprint(f"  {_DIM}Multi-line: Alt+Enter for a new line{_RST}")
        _cprint(f"  {_DIM}Paste image: Alt+V (or /paste){_RST}\n")
    
    def show_tools(self):
        """Display available tools with kawaii ASCII art."""
        tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets, quiet_mode=True)
        
        if not tools:
            print("(;_;) No tools available")
            return
        
        # Header
        print()
        title = "(^_^)/ Available Tools"
        width = 78
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        # Group tools by toolset
        toolsets = {}
        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            toolset = get_toolset_for_tool(name) or "unknown"
            if toolset not in toolsets:
                toolsets[toolset] = []
            desc = tool["function"].get("description", "")
            # First sentence: split on ". " (period+space) to avoid breaking on "e.g." or "v2.0"
            desc = desc.split("\n")[0]
            if ". " in desc:
                desc = desc[:desc.index(". ") + 1]
            toolsets[toolset].append((name, desc))
        
        # Display by toolset
        for toolset in sorted(toolsets.keys()):
            print(f"  [{toolset}]")
            for name, desc in toolsets[toolset]:
                print(f"    * {name:<20} - {desc}")
            print()
        
        print(f"  Total: {len(tools)} tools  ヽ(^o^)ノ")
        print()
    
    def show_toolsets(self):
        """Display available toolsets with kawaii ASCII art."""
        all_toolsets = get_all_toolsets()
        
        # Header
        print()
        title = "(^_^)b Available Toolsets"
        width = 58
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        
        for name in sorted(all_toolsets.keys()):
            info = get_toolset_info(name)
            if info:
                tool_count = info["tool_count"]
                desc = info["description"]
                
                # Mark if currently enabled
                marker = "(*)" if self.enabled_toolsets and name in self.enabled_toolsets else "   "
                print(f"  {marker} {name:<18} [{tool_count:>2} tools] - {desc}")
        
        print()
        print("  (*) = currently enabled")
        print()
        print("  Tip: Use 'all' or '*' to enable all toolsets")
        print("  Example: python cli.py --toolsets web,terminal")
        print()
    
    def show_config(self):
        """Display current configuration with kawaii ASCII art."""
        # Get terminal config from environment (which was set from cli-config.yaml)
        terminal_env = os.getenv("TERMINAL_ENV", "local")
        terminal_cwd = os.getenv("TERMINAL_CWD", os.getcwd())
        terminal_timeout = os.getenv("TERMINAL_TIMEOUT", "60")
        
        user_config_path = Path.home() / '.hermes' / 'config.yaml'
        project_config_path = Path(__file__).parent / 'cli-config.yaml'
        if user_config_path.exists():
            config_path = user_config_path
        else:
            config_path = project_config_path
        config_status = "(loaded)" if config_path.exists() else "(not found)"
        
        api_key_display = '********' + self.api_key[-4:] if self.api_key and len(self.api_key) > 4 else 'Not set!'
        
        print()
        title = "(^_^) Configuration"
        width = 50
        pad = width - len(title)
        print("+" + "-" * width + "+")
        print("|" + " " * (pad // 2) + title + " " * (pad - pad // 2) + "|")
        print("+" + "-" * width + "+")
        print()
        print("  -- Model --")
        print(f"  Model:     {self.model}")
        print(f"  Base URL:  {self.base_url}")
        print(f"  API Key:   {api_key_display}")
        print()
        print("  -- Terminal --")
        print(f"  Environment:  {terminal_env}")
        if terminal_env == "ssh":
            ssh_host = os.getenv("TERMINAL_SSH_HOST", "not set")
            ssh_user = os.getenv("TERMINAL_SSH_USER", "not set")
            ssh_port = os.getenv("TERMINAL_SSH_PORT", "22")
            print(f"  SSH Target:   {ssh_user}@{ssh_host}:{ssh_port}")
        print(f"  Working Dir:  {terminal_cwd}")
        print(f"  Timeout:      {terminal_timeout}s")
        print()
        print("  -- Agent --")
        print(f"  Max Turns:  {self.max_turns}")
        print(f"  Toolsets:   {', '.join(self.enabled_toolsets) if self.enabled_toolsets else 'all'}")
        print(f"  Verbose:    {self.verbose}")
        print()
        print("  -- Session --")
        print(f"  Started:     {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Config File: {config_path} {config_status}")
        print()
    
    def show_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            print("(._.) No conversation history yet.")
            return

        preview_limit = 400
        visible_index = 0
        hidden_tool_messages = 0

        def flush_tool_summary():
            nonlocal hidden_tool_messages
            if not hidden_tool_messages:
                return

            noun = "message" if hidden_tool_messages == 1 else "messages"
            print("\n  [Tools]")
            print(f"    ({hidden_tool_messages} tool {noun} hidden)")
            hidden_tool_messages = 0

        print()
        print("+" + "-" * 50 + "+")
        print("|" + " " * 12 + "(^_^) Conversation History" + " " * 11 + "|")
        print("+" + "-" * 50 + "+")

        for msg in self.conversation_history:
            role = msg.get("role", "unknown")

            if role == "tool":
                hidden_tool_messages += 1
                continue

            if role not in {"user", "assistant"}:
                continue

            flush_tool_summary()
            visible_index += 1

            content = msg.get("content")
            content_text = "" if content is None else str(content)

            if role == "user":
                print(f"\n  [You #{visible_index}]")
                print(
                    f"    {content_text[:preview_limit]}{'...' if len(content_text) > preview_limit else ''}"
                )
                continue

            print(f"\n  [Hermes #{visible_index}]")
            tool_calls = msg.get("tool_calls") or []
            if content_text:
                preview = content_text[:preview_limit]
                suffix = "..." if len(content_text) > preview_limit else ""
            elif tool_calls:
                tool_count = len(tool_calls)
                noun = "call" if tool_count == 1 else "calls"
                preview = f"(requested {tool_count} tool {noun})"
                suffix = ""
            else:
                preview = "(no text response)"
                suffix = ""
            print(f"    {preview}{suffix}")

        flush_tool_summary()
        print()
    
    def reset_conversation(self):
        """Reset the conversation history."""
        if self.agent and self.conversation_history:
            try:
                self.agent.flush_memories(self.conversation_history)
            except Exception:
                pass
        self.conversation_history = []
        print("(^_^)b Conversation reset!")
    
    def save_conversation(self):
        """Save the current conversation to a file."""
        if not self.conversation_history:
            print("(;_;) No conversation to save.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hermes_conversation_{timestamp}.json"
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "model": self.model,
                    "session_start": self.session_start.isoformat(),
                    "messages": self.conversation_history,
                }, f, indent=2, ensure_ascii=False)
            print(f"(^_^)v Conversation saved to: {filename}")
        except Exception as e:
            print(f"(x_x) Failed to save: {e}")
    
    def retry_last(self):
        """Retry the last user message by removing the last exchange and re-sending.
        
        Removes the last assistant response (and any tool-call messages) and
        the last user message, then re-sends that user message to the agent.
        Returns the message to re-send, or None if there's nothing to retry.
        """
        if not self.conversation_history:
            print("(._.) No messages to retry.")
            return None
        
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("(._.) No user message found to retry.")
            return None
        
        # Extract the message text and remove everything from that point forward
        last_message = self.conversation_history[last_user_idx].get("content", "")
        self.conversation_history = self.conversation_history[:last_user_idx]
        
        print(f"(^_^)b Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")
        return last_message
    
    def undo_last(self):
        """Remove the last user/assistant exchange from conversation history.
        
        Walks backwards and removes all messages from the last user message
        onward (including assistant responses, tool calls, etc.).
        """
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return
        
        # Walk backwards to find the last user message
        last_user_idx = None
        for i in range(len(self.conversation_history) - 1, -1, -1):
            if self.conversation_history[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            print("(._.) No user message found to undo.")
            return
        
        # Count how many messages we're removing
        removed_count = len(self.conversation_history) - last_user_idx
        removed_msg = self.conversation_history[last_user_idx].get("content", "")
        
        # Truncate history to before the last user message
        self.conversation_history = self.conversation_history[:last_user_idx]
        
        print(f"(^_^)b Undid {removed_count} message(s). Removed: \"{removed_msg[:60]}{'...' if len(removed_msg) > 60 else ''}\"")
        remaining = len(self.conversation_history)
        print(f"  {remaining} message(s) remaining in history.")
    
    def _handle_prompt_command(self, cmd: str):
        """Handle the /prompt command to view or set system prompt."""
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set new prompt
            new_prompt = parts[1].strip()
            
            if new_prompt.lower() == "clear":
                self.system_prompt = ""
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", ""):
                    print("(^_^)b System prompt cleared (saved to config)")
                else:
                    print("(^_^) System prompt cleared (session only)")
            else:
                self.system_prompt = new_prompt
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", new_prompt):
                    print(f"(^_^)b System prompt set (saved to config)")
                else:
                    print(f"(^_^) System prompt set (session only)")
                print(f"  \"{new_prompt[:60]}{'...' if len(new_prompt) > 60 else ''}\"")
        else:
            # Show current prompt
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 15 + "(^_^) System Prompt" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            if self.system_prompt:
                # Word wrap the prompt for display
                words = self.system_prompt.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= 50:
                        current_line += (" " if current_line else "") + word
                    else:
                        lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                for line in lines:
                    print(f"  {line}")
            else:
                print("  (no custom prompt set - using default)")
            print()
            print("  Usage:")
            print("    /prompt <text>  - Set a custom system prompt")
            print("    /prompt clear   - Remove custom prompt")
            print("    /personality    - Use a predefined personality")
            print()
    
    def _handle_personality_command(self, cmd: str):
        """Handle the /personality command to set predefined personalities."""
        parts = cmd.split(maxsplit=1)
        
        if len(parts) > 1:
            # Set personality
            personality_name = parts[1].strip().lower()
            
            if personality_name in self.personalities:
                self.system_prompt = self.personalities[personality_name]
                self.agent = None  # Force re-init
                if save_config_value("agent.system_prompt", self.system_prompt):
                    print(f"(^_^)b Personality set to '{personality_name}' (saved to config)")
                else:
                    print(f"(^_^) Personality set to '{personality_name}' (session only)")
                print(f"  \"{self.system_prompt[:60]}{'...' if len(self.system_prompt) > 60 else ''}\"")
            else:
                print(f"(._.) Unknown personality: {personality_name}")
                print(f"  Available: {', '.join(self.personalities.keys())}")
        else:
            # Show available personalities
            print()
            print("+" + "-" * 50 + "+")
            print("|" + " " * 12 + "(^o^)/ Personalities" + " " * 15 + "|")
            print("+" + "-" * 50 + "+")
            print()
            for name, prompt in self.personalities.items():
                print(f"  {name:<12} - \"{prompt}\"")
            print()
            print("  Usage: /personality <name>")
            print()
    
    def _handle_cron_command(self, cmd: str):
        """Handle the /cron command to manage scheduled tasks."""
        parts = cmd.split(maxsplit=2)
        
        if len(parts) == 1:
            # /cron - show help and list
            print()
            print("+" + "-" * 60 + "+")
            print("|" + " " * 18 + "(^_^) Scheduled Tasks" + " " * 19 + "|")
            print("+" + "-" * 60 + "+")
            print()
            print("  Commands:")
            print("    /cron                     - List scheduled jobs")
            print("    /cron list                - List scheduled jobs")
            print('    /cron add <schedule> <prompt>  - Add a new job')
            print("    /cron remove <job_id>     - Remove a job")
            print()
            print("  Schedule formats:")
            print("    30m, 2h, 1d              - One-shot delay")
            print('    "every 30m", "every 2h"  - Recurring interval')
            print('    "0 9 * * *"              - Cron expression')
            print()
            
            # Show current jobs
            jobs = list_jobs()
            if jobs:
                print("  Current Jobs:")
                print("  " + "-" * 55)
                for job in jobs:
                    # Format repeat status
                    times = job["repeat"].get("times")
                    completed = job["repeat"].get("completed", 0)
                    if times is None:
                        repeat_str = "forever"
                    else:
                        repeat_str = f"{completed}/{times}"
                    
                    print(f"    {job['id'][:12]:<12} | {job['schedule_display']:<15} | {repeat_str:<8}")
                    prompt_preview = job['prompt'][:45] + "..." if len(job['prompt']) > 45 else job['prompt']
                    print(f"      {prompt_preview}")
                    if job.get("next_run_at"):
                        from datetime import datetime
                        next_run = datetime.fromisoformat(job["next_run_at"])
                        print(f"      Next: {next_run.strftime('%Y-%m-%d %H:%M')}")
                    print()
            else:
                print("  No scheduled jobs. Use '/cron add' to create one.")
            print()
            return
        
        subcommand = parts[1].lower()
        
        if subcommand == "list":
            # /cron list - just show jobs
            jobs = list_jobs()
            if not jobs:
                print("(._.) No scheduled jobs.")
                return
            
            print()
            print("Scheduled Jobs:")
            print("-" * 70)
            for job in jobs:
                times = job["repeat"].get("times")
                completed = job["repeat"].get("completed", 0)
                repeat_str = "forever" if times is None else f"{completed}/{times}"
                
                print(f"  ID: {job['id']}")
                print(f"  Name: {job['name']}")
                print(f"  Schedule: {job['schedule_display']} ({repeat_str})")
                print(f"  Next run: {job.get('next_run_at', 'N/A')}")
                print(f"  Prompt: {job['prompt'][:80]}{'...' if len(job['prompt']) > 80 else ''}")
                if job.get("last_run_at"):
                    print(f"  Last run: {job['last_run_at']} ({job.get('last_status', '?')})")
                print()
        
        elif subcommand == "add":
            # /cron add <schedule> <prompt>
            if len(parts) < 3:
                print("(._.) Usage: /cron add <schedule> <prompt>")
                print("  Example: /cron add 30m Remind me to take a break")
                print('  Example: /cron add "every 2h" Check server status at 192.168.1.1')
                return
            
            # Parse schedule and prompt
            rest = parts[2].strip()
            
            # Handle quoted schedule (e.g., "every 30m" or "0 9 * * *")
            if rest.startswith('"'):
                # Find closing quote
                close_quote = rest.find('"', 1)
                if close_quote == -1:
                    print("(._.) Unmatched quote in schedule")
                    return
                schedule = rest[1:close_quote]
                prompt = rest[close_quote + 1:].strip()
            else:
                # First word is schedule
                schedule_parts = rest.split(maxsplit=1)
                schedule = schedule_parts[0]
                prompt = schedule_parts[1] if len(schedule_parts) > 1 else ""
            
            if not prompt:
                print("(._.) Please provide a prompt for the job")
                return
            
            try:
                job = create_job(prompt=prompt, schedule=schedule)
                print(f"(^_^)b Created job: {job['id']}")
                print(f"  Schedule: {job['schedule_display']}")
                print(f"  Next run: {job['next_run_at']}")
            except Exception as e:
                print(f"(x_x) Failed to create job: {e}")
        
        elif subcommand == "remove" or subcommand == "rm" or subcommand == "delete":
            # /cron remove <job_id>
            if len(parts) < 3:
                print("(._.) Usage: /cron remove <job_id>")
                return
            
            job_id = parts[2].strip()
            job = get_job(job_id)
            
            if not job:
                print(f"(._.) Job not found: {job_id}")
                return
            
            if remove_job(job_id):
                print(f"(^_^)b Removed job: {job['name']} ({job_id})")
            else:
                print(f"(x_x) Failed to remove job: {job_id}")
        
        else:
            print(f"(._.) Unknown cron command: {subcommand}")
            print("  Available: list, add, remove")
    
    def _handle_skills_command(self, cmd: str):
        """Handle /skills slash command — delegates to hermes_cli.skills_hub."""
        from hermes_cli.skills_hub import handle_skills_slash
        handle_skills_slash(cmd, self.console)

    def _show_gateway_status(self):
        """Show status of the gateway and connected messaging platforms."""
        from gateway.config import load_gateway_config, Platform
        
        print()
        print("+" + "-" * 60 + "+")
        print("|" + " " * 15 + "(✿◠‿◠) Gateway Status" + " " * 17 + "|")
        print("+" + "-" * 60 + "+")
        print()
        
        try:
            config = load_gateway_config()
            connected = config.get_connected_platforms()
            
            print("  Messaging Platform Configuration:")
            print("  " + "-" * 55)
            
            platform_status = {
                Platform.TELEGRAM: ("Telegram", "TELEGRAM_BOT_TOKEN"),
                Platform.DISCORD: ("Discord", "DISCORD_BOT_TOKEN"),
                Platform.WHATSAPP: ("WhatsApp", "WHATSAPP_ENABLED"),
            }
            
            for platform, (name, env_var) in platform_status.items():
                pconfig = config.platforms.get(platform)
                if pconfig and pconfig.enabled:
                    home = config.get_home_channel(platform)
                    home_str = f" → {home.name}" if home else ""
                    print(f"    ✓ {name:<12} Enabled{home_str}")
                else:
                    print(f"    ○ {name:<12} Not configured ({env_var})")
            
            print()
            print("  Session Reset Policy:")
            print("  " + "-" * 55)
            policy = config.default_reset_policy
            print(f"    Mode: {policy.mode}")
            print(f"    Daily reset at: {policy.at_hour}:00")
            print(f"    Idle timeout: {policy.idle_minutes} minutes")
            
            print()
            print("  To start the gateway:")
            print("    python cli.py --gateway")
            print()
            print("  Configuration file: ~/.hermes/gateway.json")
            print()
            
        except Exception as e:
            print(f"  Error loading gateway config: {e}")
            print()
            print("  To configure the gateway:")
            print("    1. Set environment variables:")
            print("       TELEGRAM_BOT_TOKEN=your_token")
            print("       DISCORD_BOT_TOKEN=your_token")
            print("    2. Or create ~/.hermes/gateway.json")
            print()
    
    def process_command(self, command: str) -> bool:
        """
        Process a slash command.
        
        Args:
            command: The command string (starting with /)
            
        Returns:
            bool: True to continue, False to exit
        """
        # Lowercase only for dispatch matching; preserve original case for arguments
        cmd_lower = command.lower().strip()
        cmd_original = command.strip()
        
        if cmd_lower in ("/quit", "/exit", "/q"):
            return False
        elif cmd_lower == "/help":
            self.show_help()
        elif cmd_lower == "/tools":
            self.show_tools()
        elif cmd_lower == "/toolsets":
            self.show_toolsets()
        elif cmd_lower == "/config":
            self.show_config()
        elif cmd_lower == "/clear":
            # Flush memories before clearing
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except Exception:
                    pass
            self.console.clear()
            self.conversation_history = []
            self.show_banner()
            print("  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n")
        elif cmd_lower == "/history":
            self.show_history()
        elif cmd_lower.startswith("/title"):
            parts = cmd_original.split(maxsplit=1)
            if len(parts) > 1:
                raw_title = parts[1].strip()
                if raw_title:
                    if self._session_db:
                        # Sanitize the title early so feedback matches what gets stored
                        try:
                            from hermes_state import SessionDB
                            new_title = SessionDB.sanitize_title(raw_title)
                        except ValueError as e:
                            _cprint(f"  {e}")
                            new_title = None
                        if not new_title:
                            _cprint("  Title is empty after cleanup. Please use printable characters.")
                        elif self._session_db.get_session(self.session_id):
                            # Session exists in DB — set title directly
                            try:
                                if self._session_db.set_session_title(self.session_id, new_title):
                                    _cprint(f"  Session title set: {new_title}")
                                else:
                                    _cprint("  Session not found in database.")
                            except ValueError as e:
                                _cprint(f"  {e}")
                        else:
                            # Session not created yet — defer the title
                            # Check uniqueness proactively with the sanitized title
                            existing = self._session_db.get_session_by_title(new_title)
                            if existing:
                                _cprint(f"  Title '{new_title}' is already in use by session {existing['id']}")
                            else:
                                self._pending_title = new_title
                                _cprint(f"  Session title queued: {new_title} (will be saved on first message)")
                    else:
                        _cprint("  Session database not available.")
                else:
                    _cprint("  Usage: /title <your session title>")
            else:
                # Show current title if no argument given
                if self._session_db:
                    session = self._session_db.get_session(self.session_id)
                    if session and session.get("title"):
                        _cprint(f"  Session title: {session['title']}")
                    elif self._pending_title:
                        _cprint(f"  Session title (pending): {self._pending_title}")
                    else:
                        _cprint(f"  No title set. Usage: /title <your session title>")
                else:
                    _cprint("  Session database not available.")
        elif cmd_lower in ("/reset", "/new"):
            self.reset_conversation()
        elif cmd_lower.startswith("/model"):
            # Use original case so model names like "Anthropic/Claude-Opus-4" are preserved
            parts = cmd_original.split(maxsplit=1)
            if len(parts) > 1:
                from hermes_cli.auth import resolve_provider
                from hermes_cli.models import (
                    parse_model_input,
                    validate_requested_model,
                    _PROVIDER_LABELS,
                )

                raw_input = parts[1].strip()

                # Parse provider:model syntax (e.g. "openrouter:anthropic/claude-sonnet-4.5")
                current_provider = self.provider or self.requested_provider or "openrouter"
                target_provider, new_model = parse_model_input(raw_input, current_provider)
                provider_changed = target_provider != current_provider

                # If provider is changing, re-resolve credentials for the new provider
                api_key_for_probe = self.api_key
                base_url_for_probe = self.base_url
                if provider_changed:
                    try:
                        from hermes_cli.runtime_provider import resolve_runtime_provider
                        runtime = resolve_runtime_provider(requested=target_provider)
                        api_key_for_probe = runtime.get("api_key", "")
                        base_url_for_probe = runtime.get("base_url", "")
                    except Exception as e:
                        provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
                        print(f"(>_<) Could not resolve credentials for provider '{provider_label}': {e}")
                        print(f"(^_^) Current model unchanged: {self.model}")
                        return True

                try:
                    validation = validate_requested_model(
                        new_model,
                        target_provider,
                        api_key=api_key_for_probe,
                        base_url=base_url_for_probe,
                    )
                except Exception:
                    validation = {"accepted": True, "persist": True, "recognized": False, "message": None}

                if not validation.get("accepted"):
                    print(f"(>_<) {validation.get('message')}")
                    print(f"  Model unchanged: {self.model}")
                    if "Did you mean" not in (validation.get("message") or ""):
                        print("  Tip: Use /model to see available models, /provider to see providers")
                else:
                    self.model = new_model
                    self.agent = None  # Force re-init

                    if provider_changed:
                        self.requested_provider = target_provider
                        self.provider = target_provider
                        self.api_key = api_key_for_probe
                        self.base_url = base_url_for_probe

                    provider_label = _PROVIDER_LABELS.get(target_provider, target_provider)
                    provider_note = f" [provider: {provider_label}]" if provider_changed else ""

                    if validation.get("persist"):
                        saved_model = save_config_value("model.default", new_model)
                        if provider_changed:
                            save_config_value("model.provider", target_provider)
                        if saved_model:
                            print(f"(^_^)b Model changed to: {new_model}{provider_note} (saved to config)")
                        else:
                            print(f"(^_^) Model changed to: {new_model}{provider_note} (this session only)")
                    else:
                        message = validation.get("message") or ""
                        print(f"(^_^) Model changed to: {new_model}{provider_note} (this session only)")
                        if message:
                            print(f"  Reason: {message}")
                        print("  Note: Model will revert on restart. Use a verified model to save to config.")
            else:
                from hermes_cli.models import curated_models_for_provider, normalize_provider, _PROVIDER_LABELS
                from hermes_cli.auth import resolve_provider as _resolve_provider
                # Resolve "auto" to the actual provider using credential detection
                raw_provider = normalize_provider(self.provider)
                if raw_provider == "auto":
                    try:
                        display_provider = _resolve_provider(
                            self.requested_provider,
                            explicit_api_key=self._explicit_api_key,
                            explicit_base_url=self._explicit_base_url,
                        )
                    except Exception:
                        display_provider = "openrouter"
                else:
                    display_provider = raw_provider
                provider_label = _PROVIDER_LABELS.get(display_provider, display_provider)
                print(f"\n  Current model:    {self.model}")
                print(f"  Current provider: {provider_label}")
                print()
                curated = curated_models_for_provider(display_provider)
                if curated:
                    print(f"  Available models ({provider_label}):")
                    for mid, desc in curated:
                        marker = " ←" if mid == self.model else ""
                        label = f"  {desc}" if desc else ""
                        print(f"    {mid}{label}{marker}")
                    print()
                print("  Usage: /model <model-name>")
                print("         /model provider:model-name  (to switch provider)")
                print("  Example: /model openrouter:anthropic/claude-sonnet-4.5")
                print("  See /provider for available providers")
        elif cmd_lower == "/provider":
            from hermes_cli.models import list_available_providers, normalize_provider, _PROVIDER_LABELS
            from hermes_cli.auth import resolve_provider as _resolve_provider
            # Resolve current provider
            raw_provider = normalize_provider(self.provider)
            if raw_provider == "auto":
                try:
                    current = _resolve_provider(
                        self.requested_provider,
                        explicit_api_key=self._explicit_api_key,
                        explicit_base_url=self._explicit_base_url,
                    )
                except Exception:
                    current = "openrouter"
            else:
                current = raw_provider
            current_label = _PROVIDER_LABELS.get(current, current)
            print(f"\n  Current provider: {current_label} ({current})\n")
            providers = list_available_providers()
            print("  Available providers:")
            for p in providers:
                marker = " ← active" if p["id"] == current else ""
                auth = "✓" if p["authenticated"] else "✗"
                aliases = f"  (also: {', '.join(p['aliases'])})" if p["aliases"] else ""
                print(f"    [{auth}] {p['id']:<14} {p['label']}{aliases}{marker}")
            print()
            print("  Switch: /model provider:model-name")
            print("  Setup:  hermes setup")
        elif cmd_lower.startswith("/prompt"):
            # Use original case so prompt text isn't lowercased
            self._handle_prompt_command(cmd_original)
        elif cmd_lower.startswith("/personality"):
            # Use original case (handler lowercases the personality name itself)
            self._handle_personality_command(cmd_original)
        elif cmd_lower == "/retry":
            retry_msg = self.retry_last()
            if retry_msg and hasattr(self, '_pending_input'):
                # Re-queue the message so process_loop sends it to the agent
                self._pending_input.put(retry_msg)
        elif cmd_lower == "/undo":
            self.undo_last()
        elif cmd_lower == "/save":
            self.save_conversation()
        elif cmd_lower.startswith("/cron"):
            self._handle_cron_command(cmd_original)
        elif cmd_lower.startswith("/skills"):
            self._handle_skills_command(cmd_original)
        elif cmd_lower == "/platforms" or cmd_lower == "/gateway":
            self._show_gateway_status()
        elif cmd_lower == "/verbose":
            self._toggle_verbose()
        elif cmd_lower == "/compress":
            self._manual_compress()
        elif cmd_lower == "/usage":
            self._show_usage()
        elif cmd_lower.startswith("/insights"):
            self._show_insights(cmd_original)
        elif cmd_lower == "/paste":
            self._handle_paste_command()
        elif cmd_lower == "/reload-mcp":
            self._reload_mcp()
        else:
            # Check for skill slash commands (/gif-search, /axolotl, etc.)
            base_cmd = cmd_lower.split()[0]
            if base_cmd in _skill_commands:
                user_instruction = cmd_original[len(base_cmd):].strip()
                msg = build_skill_invocation_message(base_cmd, user_instruction)
                if msg:
                    skill_name = _skill_commands[base_cmd]["name"]
                    print(f"\n⚡ Loading skill: {skill_name}")
                    if hasattr(self, '_pending_input'):
                        self._pending_input.put(msg)
                else:
                    self.console.print(f"[bold red]Failed to load skill for {base_cmd}[/]")
            else:
                self.console.print(f"[bold red]Unknown command: {cmd_lower}[/]")
                self.console.print("[dim #B8860B]Type /help for available commands[/]")
        
        return True
    
    def _toggle_verbose(self):
        """Cycle tool progress mode: off → new → all → verbose → off."""
        cycle = ["off", "new", "all", "verbose"]
        try:
            idx = cycle.index(self.tool_progress_mode)
        except ValueError:
            idx = 2  # default to "all"
        self.tool_progress_mode = cycle[(idx + 1) % len(cycle)]
        self.verbose = self.tool_progress_mode == "verbose"

        if self.agent:
            self.agent.verbose_logging = self.verbose
            self.agent.quiet_mode = not self.verbose

        labels = {
            "off": "[dim]Tool progress: OFF[/] — silent mode, just the final response.",
            "new": "[yellow]Tool progress: NEW[/] — show each new tool (skip repeats).",
            "all": "[green]Tool progress: ALL[/] — show every tool call.",
            "verbose": "[bold green]Tool progress: VERBOSE[/] — full args, results, and debug logs.",
        }
        self.console.print(labels.get(self.tool_progress_mode, ""))

    def _manual_compress(self):
        """Manually trigger context compression on the current conversation."""
        if not self.conversation_history or len(self.conversation_history) < 4:
            print("(._.) Not enough conversation to compress (need at least 4 messages).")
            return

        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        if not self.agent.compression_enabled:
            print("(._.) Compression is disabled in config.")
            return

        original_count = len(self.conversation_history)
        try:
            from agent.model_metadata import estimate_messages_tokens_rough
            approx_tokens = estimate_messages_tokens_rough(self.conversation_history)
            print(f"🗜️  Compressing {original_count} messages (~{approx_tokens:,} tokens)...")

            compressed, new_system = self.agent._compress_context(
                self.conversation_history,
                self.agent._cached_system_prompt or "",
                approx_tokens=approx_tokens,
            )
            self.conversation_history = compressed
            new_count = len(self.conversation_history)
            new_tokens = estimate_messages_tokens_rough(self.conversation_history)
            print(
                f"  ✅ Compressed: {original_count} → {new_count} messages "
                f"(~{approx_tokens:,} → ~{new_tokens:,} tokens)"
            )
        except Exception as e:
            print(f"  ❌ Compression failed: {e}")

    def _show_usage(self):
        """Show cumulative token usage for the current session."""
        if not self.agent:
            print("(._.) No active agent -- send a message first.")
            return

        agent = self.agent
        prompt = agent.session_prompt_tokens
        completion = agent.session_completion_tokens
        total = agent.session_total_tokens
        calls = agent.session_api_calls

        if calls == 0:
            print("(._.) No API calls made yet in this session.")
            return

        # Current context window state
        compressor = agent.context_compressor
        last_prompt = compressor.last_prompt_tokens
        ctx_len = compressor.context_length
        pct = (last_prompt / ctx_len * 100) if ctx_len else 0
        compressions = compressor.compression_count

        msg_count = len(self.conversation_history)

        print(f"  📊 Session Token Usage")
        print(f"  {'─' * 40}")
        print(f"  Prompt tokens (input):     {prompt:>10,}")
        print(f"  Completion tokens (output): {completion:>9,}")
        print(f"  Total tokens:              {total:>10,}")
        print(f"  API calls:                 {calls:>10,}")
        print(f"  {'─' * 40}")
        print(f"  Current context:  {last_prompt:,} / {ctx_len:,} ({pct:.0f}%)")
        print(f"  Messages:         {msg_count}")
        print(f"  Compressions:     {compressions}")

        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            for noisy in ('openai', 'openai._base_client', 'httpx', 'httpcore', 'asyncio', 'hpack', 'grpc', 'modal'):
                logging.getLogger(noisy).setLevel(logging.WARNING)
        else:
            logging.getLogger().setLevel(logging.INFO)
            for quiet_logger in ('tools', 'minisweagent', 'run_agent', 'trajectory_compressor', 'cron', 'hermes_cli'):
                logging.getLogger(quiet_logger).setLevel(logging.ERROR)

    def _show_insights(self, command: str = "/insights"):
        """Show usage insights and analytics from session history."""
        # Parse optional --days flag
        parts = command.split()
        days = 30
        source = None
        i = 1
        while i < len(parts):
            if parts[i] == "--days" and i + 1 < len(parts):
                try:
                    days = int(parts[i + 1])
                except ValueError:
                    print(f"  Invalid --days value: {parts[i + 1]}")
                    return
                i += 2
            elif parts[i] == "--source" and i + 1 < len(parts):
                source = parts[i + 1]
                i += 2
            else:
                i += 1

        try:
            from hermes_state import SessionDB
            from agent.insights import InsightsEngine

            db = SessionDB()
            engine = InsightsEngine(db)
            report = engine.generate(days=days, source=source)
            print(engine.format_terminal(report))
            db.close()
        except Exception as e:
            print(f"  Error generating insights: {e}")

    def _reload_mcp(self):
        """Reload MCP servers: disconnect all, re-read config.yaml, reconnect.

        After reconnecting, refreshes the agent's tool list so the model
        sees the updated tools on the next turn.
        """
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _load_mcp_config, _servers, _lock

            # Capture old server names
            with _lock:
                old_servers = set(_servers.keys())

            print("🔄 Reloading MCP servers...")

            # Shutdown existing connections
            shutdown_mcp_servers()

            # Reconnect (reads config.yaml fresh)
            new_tools = discover_mcp_tools()

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            if reconnected:
                print(f"  ♻️  Reconnected: {', '.join(sorted(reconnected))}")
            if added:
                print(f"  ➕ Added: {', '.join(sorted(added))}")
            if removed:
                print(f"  ➖ Removed: {', '.join(sorted(removed))}")
            if not connected_servers:
                print("  No MCP servers connected.")
            else:
                print(f"  🔧 {len(new_tools)} tool(s) available from {len(connected_servers)} server(s)")

            # Refresh the agent's tool list so the model can call new tools
            if self.agent is not None:
                from model_tools import get_tool_definitions
                self.agent.tools = get_tool_definitions(
                    enabled_toolsets=self.agent.enabled_toolsets
                    if hasattr(self.agent, "enabled_toolsets") else None,
                    quiet_mode=True,
                )
                self.agent.valid_tool_names = {
                    tool["function"]["name"] for tool in self.agent.tools
                } if self.agent.tools else set()

            # Inject a message at the END of conversation history so the
            # model knows tools changed.  Appended after all existing
            # messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            self.conversation_history.append({
                "role": "user",
                "content": f"[SYSTEM: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            })

            # Persist session immediately so the session log reflects the
            # updated tools list (self.agent.tools was refreshed above).
            if self.agent is not None:
                try:
                    self.agent._persist_session(
                        self.conversation_history,
                        self.conversation_history,
                    )
                except Exception:
                    pass  # Best-effort

            print(f"  ✅ Agent updated — {len(self.agent.tools if self.agent else [])} tool(s) available")

        except Exception as e:
            print(f"  ❌ MCP reload failed: {e}")

    def _stream_delta(self, text: str) -> None:
        """Write streaming token directly to stdout."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _clarify_callback(self, question, choices):
        """
        Platform callback for the clarify tool. Called from the agent thread.

        Sets up the interactive selection UI (or freetext prompt for open-ended
        questions), then blocks until the user responds via the prompt_toolkit
        key bindings.  If no response arrives within the configured timeout the
        question is dismissed and the agent is told to decide on its own.
        """
        import time as _time

        timeout = CLI_CONFIG.get("clarify", {}).get("timeout", 120)
        response_queue = queue.Queue()
        is_open_ended = not choices or len(choices) == 0

        self._clarify_state = {
            "question": question,
            "choices": choices if not is_open_ended else [],
            "selected": 0,
            "response_queue": response_queue,
        }
        self._clarify_deadline = _time.monotonic() + timeout
        # Open-ended questions skip straight to freetext input
        self._clarify_freetext = is_open_ended

        # Trigger prompt_toolkit repaint from this (non-main) thread
        self._invalidate()

        # Poll in 1-second ticks so the countdown refreshes in the UI.
        # Each tick triggers an invalidate() to repaint the hint line.
        while True:
            try:
                result = response_queue.get(timeout=1)
                self._clarify_deadline = 0
                return result
            except queue.Empty:
                remaining = self._clarify_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                # Repaint so the countdown updates
                self._invalidate()

        # Timed out — tear down the UI and let the agent decide
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._invalidate()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return (
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed."
        )

    def _sudo_password_callback(self) -> str:
        """
        Prompt for sudo password through the prompt_toolkit UI.
        
        Called from the agent thread when a sudo command is encountered.
        Uses the same clarify-style mechanism: sets UI state, waits on a
        queue for the user's response via the Enter key binding.
        """
        import time as _time

        timeout = 45
        response_queue = queue.Queue()

        self._sudo_state = {
            "response_queue": response_queue,
        }
        self._sudo_deadline = _time.monotonic() + timeout

        self._invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._sudo_state = None
                self._sudo_deadline = 0
                self._invalidate()
                if result:
                    _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
                else:
                    _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
                return result
            except queue.Empty:
                remaining = self._sudo_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._invalidate()

        self._sudo_state = None
        self._sudo_deadline = 0
        self._invalidate()
        _cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
        return ""

    def _approval_callback(self, command: str, description: str) -> str:
        """
        Prompt for dangerous command approval through the prompt_toolkit UI.
        
        Called from the agent thread. Shows a selection UI similar to clarify
        with choices: once / session / always / deny.
        """
        import time as _time

        timeout = 60
        response_queue = queue.Queue()
        choices = ["once", "session", "always", "deny"]

        self._approval_state = {
            "command": command,
            "description": description,
            "choices": choices,
            "selected": 0,
            "response_queue": response_queue,
        }
        self._approval_deadline = _time.monotonic() + timeout

        self._invalidate()

        while True:
            try:
                result = response_queue.get(timeout=1)
                self._approval_state = None
                self._approval_deadline = 0
                self._invalidate()
                return result
            except queue.Empty:
                remaining = self._approval_deadline - _time.monotonic()
                if remaining <= 0:
                    break
                self._invalidate()

        self._approval_state = None
        self._approval_deadline = 0
        self._invalidate()
    def chat(self, message, images: list = None) -> Optional[str]:
        """Send a message and get a response. Runs synchronously — streaming
        tokens go directly to stdout via stream_delta_callback."""
        if not self._ensure_runtime_credentials():
            return None
        if not self._init_agent():
            return None

        if images:
            message = self._preprocess_images_with_vision(
                message if isinstance(message, str) else "", images)

        self.conversation_history.append({"role": "user", "content": message})

        w = shutil.get_terminal_size().columns
        print(f"{_GOLD}{'─' * w}{_RST}", flush=True)

        try:
            result = self.agent.run_conversation(
                user_message=message,
                conversation_history=self.conversation_history[:-1],
                task_id=self.session_id,
            )

            self.conversation_history = result.get("messages", self.conversation_history) if result else self.conversation_history
            response = result.get("final_response", "") if result else ""

            if result and result.get("failed") and not response:
                response = f"Error: {result.get('error', 'Unknown error')}"

            # If streaming was active, tokens were already printed to stdout.
            # If not (codex path, non-streaming fallback), print the response.
            if response and not self.agent.stream_delta_callback:
                print(f"\n{response}")
            print(flush=True)
            w = shutil.get_terminal_size().columns
            print(f"{_GOLD}{'─' * w}{_RST}")

            if self.bell_on_complete:
                sys.stdout.write("\a")
                sys.stdout.flush()

            return response
        except KeyboardInterrupt:
            if self.agent:
                self.agent.interrupt()
            print("\n⚡ Interrupted")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _print_exit_summary(self):
        """Print session resume info on exit, similar to Claude Code."""
        print()
        msg_count = len(self.conversation_history)
        if msg_count > 0:
            user_msgs = len([m for m in self.conversation_history if m.get("role") == "user"])
            tool_calls = len([m for m in self.conversation_history if m.get("role") == "tool" or m.get("tool_calls")])
            elapsed = datetime.now() - self.session_start
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
            
            print(f"Resume this session with:")
            print(f"  hermes --resume {self.session_id}")
            print()
            print(f"Session:        {self.session_id}")
            print(f"Duration:       {duration_str}")
            print(f"Messages:       {msg_count} ({user_msgs} user, {tool_calls} tool calls)")
        else:
            print("Goodbye! ⚕")

    def run(self):
        """Run the interactive CLI loop. Uses PromptSession for input, plain
        stdout for output. Streaming tokens go directly to stdout — no TUI
        framework, no proxy, no layout. Copy/paste and scrolling work natively."""
        self.show_banner()

        if self._resumed:
            if self._preload_resumed_session():
                self._display_resumed_history()

        self.console.print("[#FFF8DC]Welcome to Hermes Agent! Type your message or /help for commands.[/]")
        self.console.print()
        
        self._agent_running = False
        self._should_exit = False
        self._last_ctrl_c_time = 0
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = 0
        self._sudo_state = None
        self._sudo_deadline = 0
        self._approval_state = None
        self._approval_deadline = 0
        self._attached_images: list[Path] = []
        self._image_counter = 0

        set_sudo_password_callback(self._sudo_password_callback)
        set_approval_callback(self._approval_callback)

        from hermes_cli.commands import SlashCommandCompleter
        session = PromptSession(
            history=FileHistory(str(self._history_file)),
            completer=SlashCommandCompleter(skill_commands_provider=lambda: _skill_commands),
        )

        atexit.register(_run_cleanup)
        try:
            while not self._should_exit:
                try:
                    text = session.prompt(_PT_ANSI(f"{_GOLD}❯{_RST} "))
                except KeyboardInterrupt:
                    if self._agent_running and self.agent:
                        self.agent.interrupt()
                        print("\n⚡ Interrupted")
                    else:
                        print("\nGoodbye! ⚕")
                        break
                    continue
                except EOFError:
                    break

                text = text.strip()
                if not text:
                    continue

                if text.startswith("/"):
                    if not self.process_command(text):
                        break
                    continue

                print(f"\n{_GOLD}●{_RST} {_BOLD}{text[:80]}{'...' if len(text) > 80 else ''}{_RST}")
                self._agent_running = True
                try:
                    self.chat(text)
                finally:
                    self._agent_running = False
        finally:
            if self.agent and self.conversation_history:
                try:
                    self.agent.flush_memories(self.conversation_history)
                except Exception:
                    pass
            set_sudo_password_callback(None)
            set_approval_callback(None)
            if hasattr(self, '_session_db') and self._session_db and self.agent:
                try:
                    self._session_db.end_session(self.agent.session_id, "cli_close")
                except Exception:
                    pass
            _run_cleanup()
            self._print_exit_summary()


# ============================================================================
# Main Entry Point
# ============================================================================

def main(
    query: str = None,
    q: str = None,
    toolsets: str = None,
    model: str = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    max_turns: int = None,
    verbose: bool = False,
    compact: bool = False,
    list_tools: bool = False,
    list_toolsets: bool = False,
    gateway: bool = False,
    resume: str = None,
    worktree: bool = False,
    w: bool = False,
):
    """
    Hermes Agent CLI - Interactive AI Assistant
    
    Args:
        query: Single query to execute (then exit). Alias: -q
        q: Shorthand for --query
        toolsets: Comma-separated list of toolsets to enable (e.g., "web,terminal")
        model: Model to use (default: anthropic/claude-opus-4-20250514)
        provider: Inference provider ("auto", "openrouter", "nous", "openai-codex", "zai", "kimi-coding", "minimax", "minimax-cn")
        api_key: API key for authentication
        base_url: Base URL for the API
        max_turns: Maximum tool-calling iterations (default: 60)
        verbose: Enable verbose logging
        compact: Use compact display mode
        list_tools: List available tools and exit
        list_toolsets: List available toolsets and exit
        resume: Resume a previous session by its ID (e.g., 20260225_143052_a1b2c3)
        worktree: Run in an isolated git worktree (for parallel agents). Alias: -w
        w: Shorthand for --worktree
    
    Examples:
        python cli.py                            # Start interactive mode
        python cli.py --toolsets web,terminal    # Use specific toolsets
        python cli.py -q "What is Python?"       # Single query mode
        python cli.py --list-tools               # List tools and exit
        python cli.py --resume 20260225_143052_a1b2c3  # Resume session
        python cli.py -w                         # Start in isolated git worktree
        python cli.py -w -q "Fix issue #123"     # Single query in worktree
    """
    global _active_worktree

    # Signal to terminal_tool that we're in interactive mode
    # This enables interactive sudo password prompts with timeout
    os.environ["HERMES_INTERACTIVE"] = "1"
    
    # Handle gateway mode (messaging + cron)
    if gateway:
        import asyncio
        from gateway.run import start_gateway
        print("Starting Hermes Gateway (messaging platforms)...")
        asyncio.run(start_gateway())
        return

    # Skip worktree for list commands (they exit immediately)
    if not list_tools and not list_toolsets:
        # ── Git worktree isolation (#652) ──
        # Create an isolated worktree so this agent instance doesn't collide
        # with other agents working on the same repo.
        use_worktree = worktree or w or CLI_CONFIG.get("worktree", False)
        wt_info = None
        if use_worktree:
            # Prune stale worktrees from crashed/killed sessions
            _repo = _git_repo_root()
            if _repo:
                _prune_stale_worktrees(_repo)
            wt_info = _setup_worktree()
            if wt_info:
                _active_worktree = wt_info
                os.environ["TERMINAL_CWD"] = wt_info["path"]
                atexit.register(_cleanup_worktree, wt_info)
            else:
                # Worktree was explicitly requested but setup failed —
                # don't silently run without isolation.
                return
    else:
        wt_info = None
    
    # Handle query shorthand
    query = query or q
    
    # Parse toolsets - handle both string and tuple/list inputs
    # Default to hermes-cli toolset which includes cronjob management tools
    toolsets_list = None
    if toolsets:
        if isinstance(toolsets, str):
            toolsets_list = [t.strip() for t in toolsets.split(",")]
        elif isinstance(toolsets, (list, tuple)):
            # Fire may pass multiple --toolsets as a tuple
            toolsets_list = []
            for t in toolsets:
                if isinstance(t, str):
                    toolsets_list.extend([x.strip() for x in t.split(",")])
                else:
                    toolsets_list.append(str(t))
    else:
        # Check config for CLI toolsets, fallback to hermes-cli
        config_cli_toolsets = CLI_CONFIG.get("platform_toolsets", {}).get("cli")
        if config_cli_toolsets and isinstance(config_cli_toolsets, list):
            toolsets_list = config_cli_toolsets
        else:
            toolsets_list = ["hermes-cli"]
    
    # Create CLI instance
    cli = HermesCLI(
        model=model,
        toolsets=toolsets_list,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        max_turns=max_turns,
        verbose=verbose,
        compact=compact,
        resume=resume,
    )

    # Inject worktree context into agent's system prompt
    if wt_info:
        wt_note = (
            f"\n\n[System note: You are working in an isolated git worktree at "
            f"{wt_info['path']}. Your branch is `{wt_info['branch']}`. "
            f"Changes here do not affect the main working tree or other agents. "
            f"Remember to commit and push your changes, and create a PR if appropriate. "
            f"The original repo is at {wt_info['repo_root']}.]"
        )
        cli.system_prompt = (cli.system_prompt or "") + wt_note
    
    # Handle list commands (don't init agent for these)
    if list_tools:
        cli.show_banner()
        cli.show_tools()
        sys.exit(0)
    
    if list_toolsets:
        cli.show_banner()
        cli.show_toolsets()
        sys.exit(0)
    
    # Register cleanup for single-query mode (interactive mode registers in run())
    atexit.register(_run_cleanup)
    
    # Handle single query mode
    if query:
        cli.show_banner()
        cli.console.print(f"[bold blue]Query:[/] {query}")
        cli.chat(query)
        cli._print_exit_summary()
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    fire.Fire(main)
