from __future__ import annotations

import json

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


def test_from_sources_prefers_security_allowed_hosts_and_auto_install_flag(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace_root))
    monkeypatch.setenv("ALLOWED_HOSTS", "legacy.example")
    monkeypatch.setenv("SECURITY_ALLOWED_HOSTS", "agent.pondsec.com,192.168.30.33")
    monkeypatch.setenv("AUTO_INSTALL_RECOMMENDED_MODELS", "0")

    config = AppConfig.from_sources()

    assert config.security_allowed_hosts == ("agent.pondsec.com", "192.168.30.33")
    assert config.auto_install_recommended_models is False


def test_from_sources_falls_back_to_app_root_dotenv_when_cwd_has_no_config(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    app_root = tmp_path / "app-root"
    (app_root / "config").mkdir(parents=True)
    (app_root / ".env").write_text(
        "WORKSPACE_ROOT=/srv/workspace\nSECURITY_ALLOWED_HOSTS=agent.pondsec.com,127.0.0.1\nPUBLIC_BASE_URL=https://agent.pondsec.com\n",
        encoding="utf-8",
    )
    (app_root / "config" / "agent.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(workspace_root)
    monkeypatch.setattr("config.settings._app_source_root", lambda: app_root)
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("SECURITY_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)

    config = AppConfig.from_sources()

    assert config.workspace_root == "/srv/workspace"
    assert config.security_allowed_hosts == ("agent.pondsec.com", "127.0.0.1")
    assert config.public_base_url == "https://agent.pondsec.com"


def test_from_sources_uses_config_path_parent_for_dotenv_lookup(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    config_dir = bundle_root / "config"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "agent.json"
    config_file.write_text(json.dumps({"workspace_root": "/srv/from-json"}), encoding="utf-8")
    (bundle_root / ".env").write_text(
        "SECURITY_ALLOWED_HOSTS=agent.pondsec.com,localhost\n",
        encoding="utf-8",
    )

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("SECURITY_ALLOWED_HOSTS", raising=False)

    config = AppConfig.from_sources(config_path=str(config_file))

    assert config.workspace_root == "/srv/from-json"
    assert config.security_allowed_hosts == ("agent.pondsec.com", "localhost")


def test_normalized_adds_public_base_url_host_to_allowed_hosts():
    config = AppConfig(
        workspace_root=".",
        security_allowed_hosts=("127.0.0.1", "localhost"),
        public_base_url="https://agent.pondsec.com",
    ).normalized()

    assert config.security_allowed_hosts == ("127.0.0.1", "localhost", "agent.pondsec.com")


def test_from_sources_prefers_workspace_env_for_runtime_settings(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / ".env").write_text(
        "PUBLIC_BASE_URL=https://agent.pondsec.com\nSECURITY_ALLOWED_HOSTS=agent.pondsec.com,localhost\nAUTH_SECRET_KEY=workspace-secret\n",
        encoding="utf-8",
    )
    app_root = tmp_path / "app-root"
    (app_root / "config").mkdir(parents=True)
    (app_root / ".env").write_text(
        "PUBLIC_BASE_URL=https://wrong.example.com\nSECURITY_ALLOWED_HOSTS=wrong.example.com\nAUTH_SECRET_KEY=app-secret\n",
        encoding="utf-8",
    )
    (app_root / "config" / "agent.json").write_text("{}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("config.settings._app_source_root", lambda: app_root)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("SECURITY_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("AUTH_SECRET_KEY", raising=False)

    config = AppConfig.from_sources(workspace_override=str(workspace_root))

    assert config.workspace_root == str(workspace_root.resolve())
    assert config.public_base_url == "https://agent.pondsec.com"
    assert config.auth_secret_key == "workspace-secret"
    assert config.security_allowed_hosts == ("agent.pondsec.com", "localhost")
