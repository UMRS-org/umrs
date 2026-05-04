"""ResolverRegistry — dispatches resolution by ModelRefType."""

from __future__ import annotations

from umrs.model_registry.models import ExecutionAuthority, ModelRef, ModelRefType

from .base import ModelRefResolver, ResolutionError, ResolvedModel
from .filesystem import FilesystemResolver
from .huggingface import HuggingFaceResolver
from .pvc import PVCResolver
from .s3 import S3Resolver


class ResolverRegistry:
    """
    Registry of ModelRef resolvers.

    Dispatches resolution to the appropriate resolver based on ModelRefType.
    """

    def __init__(self):
        self._resolvers: dict[ModelRefType, ModelRefResolver] = {}

    def register(self, resolver: ModelRefResolver) -> None:
        """Register a resolver for its ref type."""
        self._resolvers[resolver.ref_type] = resolver

    def get(self, ref_type: ModelRefType) -> ModelRefResolver | None:
        """Get resolver for a ref type."""
        return self._resolvers.get(ref_type)

    async def resolve(
        self,
        ref: ModelRef,
        authority: ExecutionAuthority,
    ) -> ResolvedModel:
        """Resolve a ModelRef using the appropriate resolver."""
        resolver = self._resolvers.get(ref.type)

        if not resolver:
            raise ResolutionError(f"No resolver registered for ref type '{ref.type}'")

        return await resolver.resolve(ref, authority)

    @classmethod
    def default(cls) -> ResolverRegistry:
        """Create registry with default resolvers."""
        registry = cls()
        registry.register(FilesystemResolver())
        registry.register(PVCResolver())
        registry.register(HuggingFaceResolver())
        registry.register(S3Resolver())
        return registry
