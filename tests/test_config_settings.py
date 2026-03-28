from __future__ import annotations

from config.settings import AppConfig


def test_from_sources_reads_state_root_override_from_environment(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    state_root = tmp_path / "isolated-state"

    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("STATE_ROOT_OVERRIDE", str(state_root))

    config = AppConfig.from_sources()

    assert config.workspace_path == workspace_root.resolve()
    assert config.state_root_override == str(state_root)
    assert config.state_root == state_root.resolve()


def test_from_sources_allows_override_dict_to_replace_state_root(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    env_state_root = tmp_path / "env-state"
    override_state_root = tmp_path / "override-state"

    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("STATE_ROOT_OVERRIDE", str(env_state_root))

    config = AppConfig.from_sources(
        overrides={"state_root_override": str(override_state_root)},
    )

    assert config.state_root_override == str(override_state_root)
    assert config.state_root == override_state_root.resolve()


def test_from_sources_reads_auto_install_models_flag_from_environment(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("AUTO_INSTALL_RECOMMENDED_MODELS", "0")

    config = AppConfig.from_sources()

    assert config.auto_install_recommended_models is False


def test_from_sources_reads_security_allowed_hosts_from_environment(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("SECURITY_ALLOWED_HOSTS", "*,192.168.20.155")

    config = AppConfig.from_sources()

    assert config.security_allowed_hosts == ("*", "192.168.20.155")
