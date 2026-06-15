"""
UMRS Authentication and Authorization Module.

Implements:
- U-501: Admin role enforcement for privileged commands
- U-502: Per-user resource isolation
- U-503: UUID-based user identification
- U-504: Config identity validation
- U-505: XDG config directory standard
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# U-501: Admin Role Enforcement
# =============================================================================


class UserRole(str, Enum):
    """User roles for authorization."""

    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"  # For internal service-to-service calls


# Commands that require admin role
ADMIN_ONLY_COMMANDS: set[str] = frozenset({
    "RELOAD",       # Reload server configuration (global, reprovisioning all users)
    "RECOMPUTE",    # Recompute embeddings across configured buckets
    "ENROLL_USER",  # Admin-only: register a new user on the server
})

# Commands that users can run on their own data
USER_ALLOWED_COMMANDS: set[str] = frozenset({
    "PROCESS_OBJECTS",
    "WAIT_OUTPUT",
    "FILE",
    "STATUS",
    "CREATE_DATA_QUEUE",
    "HANDSHAKE",
    "EXPORT_EMBEDDINGS",
    "MODEL_REGISTER",
    "MODEL_LIST",
    "MODEL_DOWNLOAD",
    "JOB_SUBMIT",
    "JOB_CANCEL",
    "JOB_LIST",
    "UPLOAD_WORKFLOW",
    "LIST_WORKFLOWS",
    "RUN_WORKFLOW",
    "UPDATE",       # Update embeddings across configured buckets (only client files)
    "CHECK",        # Global consistency check across all users' buckets (only client files)
})


@dataclass
class AuthContext:
    """
    Authentication context for a request.

    Contains user identity and authorization information.
    """

    user_id: str  # UUID-based stable identifier
    username: str  # Human-readable username (e.g., s3_username)
    role: UserRole = UserRole.USER
    project_id: str | None = None
    email: str | None = None
    allowed_buckets: set[str] = field(default_factory=set)

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN

    @property
    def is_service(self) -> bool:
        """Check if this is a service account."""
        return self.role == UserRole.SERVICE

    def can_execute_command(
        self,
        command: str,
        target_user: str | None = None,
        *,
        log_denial: bool = True,
    ) -> bool:
        """
        Check if user can execute a command.

        Args:
            command: The command to execute (e.g., RELOAD, RECOMPUTE)
            target_user: The user being targeted (for scoped commands)

        Returns:
            True if authorized, False otherwise
        """
        cmd_upper = command.upper()

        # Admins and services can do everything
        if self.is_admin or self.is_service:
            return True

        # Check if command requires admin
        if cmd_upper in ADMIN_ONLY_COMMANDS:
            if log_denial:
                logger.warning(
                    f"User {self.username} attempted admin command {cmd_upper}"
                )
            return False

        # For user-allowed commands, check scope
        if cmd_upper in USER_ALLOWED_COMMANDS:
            # Users can only operate on their own data
            if target_user and target_user != self.username:
                if log_denial:
                    logger.warning(
                        f"User {self.username} attempted to access {target_user}'s data"
                    )
                return False
            return True

        # Unknown command - deny by default
        if log_denial:
            logger.warning(f"Unknown command {cmd_upper} - denying access")
        return False

    def filter_by_ownership(self, data: dict) -> dict:
        """
        Filter data dictionary to only include user's own entries.

        U-502: Per-user resource isolation.
        """
        if self.is_admin or self.is_service:
            return data  # Admins see everything

        return {
            k: v
            for k, v in data.items()
            if isinstance(v, dict) and v.get("s3_username") == self.username
        }


class AuthorizationError(Exception):
    """Raised when authorization fails."""

    def __init__(self, message: str, command: str = "", user: str = ""):
        self.message = message
        self.command = command
        self.user = user
        super().__init__(message)


def require_admin(auth: AuthContext, command: str) -> None:
    """
    Require admin role for a command.

    Raises:
        AuthorizationError: If user is not an admin.
    """
    if not auth.is_admin and not auth.is_service:
        raise AuthorizationError(
            f"Command '{command}' requires admin role",
            command=command,
            user=auth.username,
        )


def require_ownership(auth: AuthContext, target_user: str) -> None:
    """
    Require user owns the target resource.

    Raises:
        AuthorizationError: If user doesn't own the resource.
    """
    if auth.is_admin or auth.is_service:
        return

    if auth.username != target_user:
        raise AuthorizationError(
            f"User '{auth.username}' cannot access '{target_user}'s resources",
            user=auth.username,
        )


# =============================================================================
# U-503: UUID-based User Identification
# =============================================================================


def generate_user_uuid(username: str, namespace: str = "umrs") -> str:
    """
    Generate a deterministic UUID for a username.

    Uses UUID5 (SHA-1 based) for reproducibility.
    The same username always gets the same UUID.

    Args:
        username: The username to generate UUID for
        namespace: Namespace for UUID generation (default: "umrs")

    Returns:
        UUID string (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    # Create namespace UUID from the namespace string
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{namespace}.local")
    # Generate deterministic UUID for username
    return str(uuid.uuid5(namespace_uuid, username))


def generate_random_user_id() -> str:
    """
    Generate a random user ID.

    For cases where deterministic UUID is not desired.
    """
    return str(uuid.uuid4())


# =============================================================================
# U-504: Config Identity Validation
# =============================================================================


@dataclass
class IdentityValidationResult:
    """Result of identity validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_config_identity(
    config: dict,
    expected_username: str,
    expected_user_id: str | None = None,
) -> IdentityValidationResult:
    """
    Validate that config identity matches expected values.

    U-504: Reject mismatched username/user_id uploads.

    Args:
        config: Configuration dictionary to validate
        expected_username: Expected username
        expected_user_id: Expected user ID (optional)

    Returns:
        IdentityValidationResult with validation status
    """
    result = IdentityValidationResult(valid=True)

    # Check username
    config_username = config.get("username") or config.get("s3_username")
    if config_username and config_username != expected_username:
        result.valid = False
        result.errors.append(
            f"Username mismatch: config has '{config_username}', "
            f"expected '{expected_username}'"
        )

    # Check user_id if provided
    config_user_id = config.get("user_id")
    if expected_user_id and config_user_id and config_user_id != expected_user_id:
        result.valid = False
        result.errors.append(
            f"User ID mismatch: config has '{config_user_id}', "
            f"expected '{expected_user_id}'"
        )

    # Check for suspicious fields
    if config.get("is_admin") or config.get("role") == "admin":
        result.warnings.append(
            "Config contains admin privilege claims - these will be ignored"
        )

    return result


def sanitize_username(username: str) -> str:
    """
    Sanitize username for safe use in paths and identifiers.

    Args:
        username: Raw username

    Returns:
        Sanitized username (alphanumeric + underscore/hyphen only)
    """
    # Allow only alphanumeric, underscore, and hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", username)
    # Ensure it doesn't start with a number or special char
    if sanitized and not sanitized[0].isalpha():
        sanitized = "u_" + sanitized
    return sanitized[:64]  # Limit length


# =============================================================================
# U-505: XDG Config Directory Standard
# =============================================================================


class ConfigDirectory:
    """
    XDG-compliant configuration directory management.

    Priority order:
    1. UMRS_CONFIG environment variable
    2. $XDG_CONFIG_HOME/umrs
    3. ~/.config/umrs (fallback)
    """

    DEFAULT_DIR_NAME = "umrs"

    def __init__(self, app_name: str = DEFAULT_DIR_NAME):
        self.app_name = app_name
        self._config_dir: Path | None = None
        self._data_dir: Path | None = None
        self._cache_dir: Path | None = None

    @property
    def config_dir(self) -> Path:
        """Get the configuration directory."""
        if self._config_dir is None:
            self._config_dir = self._resolve_config_dir()
        return self._config_dir

    @property
    def data_dir(self) -> Path:
        """Get the data directory (XDG_DATA_HOME)."""
        if self._data_dir is None:
            self._data_dir = self._resolve_data_dir()
        return self._data_dir

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory (XDG_CACHE_HOME)."""
        if self._cache_dir is None:
            self._cache_dir = self._resolve_cache_dir()
        return self._cache_dir

    def _resolve_config_dir(self) -> Path:
        """Resolve config directory following XDG standard."""
        # 1. Check UMRS_CONFIG env var
        umrs_config = os.environ.get("UMRS_CONFIG")
        if umrs_config:
            return Path(umrs_config)

        # 2. Check XDG_CONFIG_HOME
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / self.app_name

        # 3. Fallback to ~/.config/umrs
        return Path.home() / ".config" / self.app_name

    def _resolve_data_dir(self) -> Path:
        """Resolve data directory following XDG standard."""
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            return Path(xdg_data) / self.app_name
        return Path.home() / ".local" / "share" / self.app_name

    def _resolve_cache_dir(self) -> Path:
        """Resolve cache directory following XDG standard."""
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / self.app_name
        return Path.home() / ".cache" / self.app_name

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for dir_path in [self.config_dir, self.data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_config_file(self, filename: str) -> Path:
        """Get path to a config file."""
        return self.config_dir / filename

    def get_data_file(self, filename: str) -> Path:
        """Get path to a data file."""
        return self.data_dir / filename

    def get_cache_file(self, filename: str) -> Path:
        """Get path to a cache file."""
        return self.cache_dir / filename


# Global config directory instance
_config_directory: ConfigDirectory | None = None


def get_config_directory() -> ConfigDirectory:
    """Get the global config directory instance."""
    global _config_directory
    if _config_directory is None:
        _config_directory = ConfigDirectory()
    return _config_directory


# =============================================================================
# User Identity Store
# =============================================================================


class UserIdentityStore:
    """
    Persistent storage for user identity mappings.

    Maps usernames to stable UUIDs and stores user metadata.
    """

    def __init__(self, config_dir: ConfigDirectory | None = None):
        self._config = config_dir or get_config_directory()
        self._identity_file = self._config.get_data_file("user_identities.json")
        self._cache: dict | None = None

    def get_or_create_user_id(self, username: str) -> str:
        """
        Get existing user ID or create a new one.

        Args:
            username: The username

        Returns:
            UUID string for the user
        """
        identities = self._load()

        if username in identities:
            return identities[username]["user_id"]

        # Create new identity
        user_id = generate_user_uuid(username)
        identities[username] = {
            "user_id": user_id,
            "username": username,
            "created_at": self._now_iso(),
        }
        self._save(identities)

        logger.info(f"Created new user identity: {username} -> {user_id}")
        return user_id

    def get_user_id(self, username: str) -> str | None:
        """Get user ID for username, or None if not found."""
        identities = self._load()
        entry = identities.get(username)
        return entry["user_id"] if entry else None

    def get_username(self, user_id: str) -> str | None:
        """Get username for user ID, or None if not found."""
        identities = self._load()
        for username, entry in identities.items():
            if entry.get("user_id") == user_id:
                return username
        return None

    def list_users(self) -> list[dict]:
        """List all registered users."""
        identities = self._load()
        return [
            {"username": k, **v}
            for k, v in identities.items()
        ]

    def _load(self) -> dict:
        """Load identities from file."""
        if self._cache is not None:
            return self._cache

        if not self._identity_file.exists():
            return {}

        try:
            self._cache = json.loads(self._identity_file.read_text())
            return self._cache
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load user identities: {e}")
            return {}

    def _save(self, identities: dict) -> None:
        """Save identities to file."""
        self._config.data_dir.mkdir(parents=True, exist_ok=True)
        self._identity_file.write_text(json.dumps(identities, indent=2))
        self._identity_file.chmod(0o600)  # Restrict permissions
        self._cache = identities

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# Global identity store instance
_identity_store: UserIdentityStore | None = None


def get_identity_store() -> UserIdentityStore:
    """Get the global identity store instance."""
    global _identity_store
    if _identity_store is None:
        _identity_store = UserIdentityStore()
    return _identity_store


# =============================================================================
# Admin Token Management
# =============================================================================

WEAK_ADMIN_TOKEN_LITERALS = {
    "",
    "admin",
    "password",
    "changeme",
    "change-me",
    "change_me",
    "umrs-admin",
    "umrs-admin-token",
    "umrs-admin-change-me",
}


def is_weak_admin_token(token: str) -> bool:
    """Return True when an admin token is a documented weak/default literal."""
    return token.strip() in WEAK_ADMIN_TOKEN_LITERALS


class AdminTokenManager:
    """
    Manage admin API tokens.

    Tokens are stored securely in the config directory.
    """

    def __init__(self, config_dir: ConfigDirectory | None = None):
        self._config = config_dir or get_config_directory()
        self._tokens_file = self._config.get_config_file("admin_tokens.json")

    def generate_token(self, name: str, description: str = "") -> str:
        """
        Generate a new admin token.

        Args:
            name: Token name (for identification)
            description: Optional description

        Returns:
            The generated token string
        """
        token = f"umrs-admin-{secrets.token_urlsafe(32)}"

        tokens = self._load_tokens()
        token_hash, token_salt = self._hash_token(token)
        tokens[name] = {
            "token_hash": token_hash,
            "token_salt": token_salt,
            "description": description,
            "created_at": self._now_iso(),
        }
        self._save_tokens(tokens)

        logger.info(f"Generated admin token: {name}")
        return token

    def validate_token(self, token: str) -> str | None:
        """
        Validate an admin token.

        Args:
            token: The token to validate

        Returns:
            Token name if valid, None otherwise
        """
        if is_weak_admin_token(token):
            logger.warning("Rejected weak admin token literal")
            return None

        tokens = self._load_tokens()

        for name, entry in tokens.items():
            stored_salt = entry.get("token_salt")
            if stored_salt:
                # New salted format
                computed_hash, _ = self._hash_token(token, salt=stored_salt)
                if computed_hash == entry.get("token_hash"):
                    return name
            else:
                # Legacy unsalted SHA-256 — validate and upgrade in-place
                import hashlib
                legacy_hash = hashlib.sha256(token.encode()).hexdigest()
                if legacy_hash == entry.get("token_hash"):
                    # Upgrade to salted hash
                    new_hash, new_salt = self._hash_token(token)
                    entry["token_hash"] = new_hash
                    entry["token_salt"] = new_salt
                    self._save_tokens(tokens)
                    logger.info("Upgraded legacy token %r to salted hash", name)
                    return name

        return None

    def revoke_token(self, name: str) -> bool:
        """Revoke a token by name."""
        tokens = self._load_tokens()
        if name in tokens:
            del tokens[name]
            self._save_tokens(tokens)
            logger.info(f"Revoked admin token: {name}")
            return True
        return False

    def list_tokens(self) -> list[dict]:
        """List all tokens (without hashes)."""
        tokens = self._load_tokens()
        return [
            {
                "name": name,
                "description": entry.get("description", ""),
                "created_at": entry.get("created_at", ""),
                "expires_at": self._format_token_expiration(entry),
            }
            for name, entry in tokens.items()
        ]

    def _format_token_expiration(self, entry: dict) -> str:
        """Return a display-friendly expiration timestamp for token metadata."""
        expires_at = entry.get("expires_at")
        if isinstance(expires_at, str) and expires_at.strip():
            return expires_at

        exp = entry.get("exp")
        if isinstance(exp, str):
            try:
                exp = int(exp)
            except ValueError:
                exp = None

        if isinstance(exp, (int, float)):
            from datetime import datetime, timezone

            return datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()

        return "never"

    def _hash_token(self, token: str, salt: str | None = None) -> tuple:
        """Hash a token for storage using PBKDF2 with a per-token salt.

        Returns:
            (hex_hash, hex_salt) tuple.
        """
        import hashlib
        if salt is None:
            salt = os.urandom(16).hex()
        h = hashlib.pbkdf2_hmac("sha256", token.encode(), bytes.fromhex(salt), 100_000)
        return h.hex(), salt

    def _load_tokens(self) -> dict:
        """Load tokens from file."""
        if not self._tokens_file.exists():
            return {}
        try:
            return json.loads(self._tokens_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_tokens(self, tokens: dict) -> None:
        """Save tokens to file."""
        self._config.config_dir.mkdir(parents=True, exist_ok=True)
        self._tokens_file.write_text(json.dumps(tokens, indent=2))
        self._tokens_file.chmod(0o600)

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Factory Functions
# =============================================================================


def create_auth_context_from_request(
    username: str,
    token: str | None = None,
    project_id: str | None = None,
) -> AuthContext:
    """
    Create an AuthContext from a request.

    Args:
        username: The requesting username
        token: Optional admin token
        project_id: Optional project ID

    Returns:
        AuthContext with appropriate role
    """
    identity_store = get_identity_store()
    user_id = identity_store.get_or_create_user_id(username)

    # Check for admin token
    role = UserRole.USER
    if token:
        token_manager = AdminTokenManager()
        if token_manager.validate_token(token):
            role = UserRole.ADMIN
            logger.info(f"Admin access granted for {username}")

    return AuthContext(
        user_id=user_id,
        username=username,
        role=role,
        project_id=project_id,
    )


def create_service_auth_context(service_name: str) -> AuthContext:
    """
    Create an AuthContext for a service account.

    Args:
        service_name: Name of the service

    Returns:
        AuthContext with service role
    """
    return AuthContext(
        user_id=f"service:{service_name}",
        username=service_name,
        role=UserRole.SERVICE,
    )
