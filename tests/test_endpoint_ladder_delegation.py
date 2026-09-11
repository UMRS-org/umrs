"""WP-06 round 2 leg-d: endpoint_registry delegates to the scope ladder (pins)."""

from __future__ import annotations

from umrs.orchestrator.endpoint_registry import EndpointRegistry


class TestAssetScope:
    def _registry(self, pid="rivault"):
        registry = EndpointRegistry()
        registry.register(project_id=pid, model="m", job_id="j1", host="h", port=1, backend="slurm")
        return registry

    def test_platform_pool_project_normalizes_system(self):
        ep = list(self._registry("rivault")._endpoints.values())[0]
        scope = ep.asset_scope()
        assert scope.kind == "system"
        assert scope.project_id == "rivault"

    def test_user_project_normalizes_user(self):
        ep = list(self._registry("myproj")._endpoints.values())[0]
        ep.metadata["owner_type"] = "user"
        ep.metadata["owner_id"] = "u-9"
        scope = ep.asset_scope()
        assert scope.kind == "user"
        assert scope.owner_id == "u-9"
        assert scope.project_id == "myproj"

    def test_metadata_preserved_byte_identical_tier(self):
        ep = list(self._registry("myproj")._endpoints.values())[0]
        ep.metadata["owner_type"] = "user"
        d = ep.to_dict()
        assert d["tier"] == "user"
        assert d["owner_type"] == "user"
        assert d["asset_scope"]["kind"] == "user"
        assert d["asset_scope"]["project_id"] == "myproj"

    def test_ladder_readable_delegates(self):
        from umrs.shared.scope_grammar import scope_readable

        ep = list(self._registry("myproj")._endpoints.values())[0]
        ep.metadata["owner_type"] = "user"
        ep.metadata["owner_id"] = "u-9"
        scope = ep.asset_scope()
        assert scope_readable(scope, user_id="u-9", project_id="myproj") is True
        assert scope_readable(scope, user_id="u-10", project_id="myproj") is False
        assert scope_readable(scope, user_id="u-10", project_id="myproj", is_admin=True) is True
