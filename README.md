# Chutes LiteLLM Proxy

A LiteLLM proxy that exposes a unified OpenAI-compatible API for all major providers.

## Supported providers

| Provider    | Env var              | Model prefix   |
| ----------- | -------------------- | -------------- |
| Anthropic   | `ANTHROPIC_API_KEY`  | `anthropic/`   |
| OpenAI      | `OPENAI_API_KEY`     | `openai/`      |
| Google      | `GEMINI_API_KEY`     | `gemini/`      |
| Perplexity  | `PERPLEXITY_API_KEY` | `perplexity/`  |
| xAI         | `XAI_API_KEY`        | `xai/`         |
| DeepSeek    | `DEEPSEEK_API_KEY`   | `deepseek/`    |
| Chutes E2EE | `CHUTES_API_KEY`     | `chutes-e2ee/` |

> **Native `_PATH` support:** Instead of setting `ANTHROPIC_API_KEY=sk-…` directly,
> you may set `ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic` (or any path to a file
> containing the raw key). The proxy will read the file and treat its content as the
> key. This works for every provider above and is the cleanest way to inject secrets
> from systemd service files.

## Deployment

> [!WARNING]
> The docker image must first be built before the services can start.
> Both `podman` and `docker` are supported, the image can be built using `docker build -t litellm-proxy .`.
>
> Before building the container, you also need to edit the `config.template.yml` so that it contains the name of the providers you will pass to the [run.sh](./run.sh) script.

### Manual

You can find a manual launchd file and systemd service file in the [./services](./services) folder.

Otherwise you can launch the [run.sh](./run.sh) script directly.

```
$ head run.sh
#!/usr/bin/env bash
# Runs the litellm-proxy container. Configuration is supplied entirely through
# environment variables (no CLI arguments are accepted).
#
# Env vars:
#   LITELLM_BIND      Host:port to publish (default: 127.0.0.1:4000; comma-separated)
#   RUNTIME           Container engine: podman | docker | auto (default: auto)
#   *_API_KEY          Raw API keys forwarded as-is.
#   *_API_KEY_PATH     File path containing API key; forwarded as *_API_KEY.
# Example:
#   LITELLM_BIND=127.0.0.1:4000 ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic ./run.sh
```

#### macOS

To install manually :

```sh
cp services/local.litellm-proxy.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local.litellm-proxy.plist
```

To start/stop :

```sh
launchctl load ~/Library/LaunchAgents/local.litellm-proxy.plist
launchctl bootout gui/$UID/local.litellm-proxy
```

#### linux

Install :

```sh
cp services/litellm.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now litellm
```

### Nix

#### home-manager (systemd --user, linux)

```nix
{
  config,
  lib,
  pkgs,
  inputs,
  ...
}: let
  cfg = config.custom.services.litellm;
  # Build environment variable assignments for the service.
  apiKeyEnv = lib.mapAttrsToList (name: value: "${name}=${value}") cfg.envSecrets;
  runScript = pkgs.writeShellScript "litellm-run" (
    builtins.readFile "${inputs.self}/containers/litellm/run.sh"
  );
in {
  options.custom.services.litellm = {
    enable = lib.mkEnableOption "LiteLLM proxy";

    envSecrets = lib.mkOption {
      type = lib.types.attrsOf (lib.types.either lib.types.str lib.types.path);
      default = {};
      description = "Mapping of env var name to a raw key string or a secret file path.";
      example = {
        envSecrets = {
          ANTHROPIC_API_KEY_PATH = config.age.secrets."litellm/anthropic".path;
          OPENAI_API_KEY = config.age.secrets."litellm/openai".path;
          GEMINI_API_KEY = config.age.secrets."litellm/gemini".path;
          CHUTES_API_KEY_PATH = config.age.secrets."litellm/chutes".path;
          PERPLEXITY_API_KEY = config.age.secrets."litellm/perplexity".path;
        };
      };
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 4000;
      description = "Local port to bind the LiteLLM proxy on (bound to 127.0.0.1).";
      example = {
        port = 4000;
      }
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.litellm = {
      Unit = {
        Description = "LiteLLM proxy";
      };
      Service = {
        Type = "simple";
        ExecStart = "${runScript}";
        Restart = "on-failure";
        RestartSec = 5;
        Environment = [
          "LITELLM_BIND=127.0.0.1:${toString cfg.port}"
        ] ++ apiKeyEnv;
      };
      Install = {
        WantedBy = ["default.target"];
      };
    };
  };
}
```

#### nix-darwin (launchd, macOS)

```nix
{
  config,
  lib,
  pkgs,
  username,
  inputs,
  ...
}: let
  cfg = config.custom.macos.litellm;
  # Map secrets into systemd-style KEY=value strings for the launchd EnvVars dict.
  apiKeyEnv = lib.mapAttrsToList (name: value: "${name}=${value}") cfg.envSecrets;
  runScript = pkgs.writeShellScript "litellm-run" (
    builtins.readFile "${inputs.self}/containers/litellm/run.sh"
  );
in {
  options.custom.macos.litellm = {
    enable = lib.mkEnableOption "LiteLLM proxy";

    envSecrets = lib.mkOption {
      type = lib.types.attrsOf (lib.types.either lib.types.str lib.types.path);
      default = {};
      description = "Mapping of env var name to a raw key string or a secret file path.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 4000;
      description = "Local port to bind the LiteLLM proxy on (bound to 127.0.0.1).";
    };
  };

  config = lib.mkIf cfg.enable {
    home-manager.users.${username} = {
      home.activation.bootout-litellm = inputs.home-manager.lib.hm.dag.entryBefore ["setupLaunchAgents"] ''
        /bin/launchctl bootout "gui/$UID/local.litellm-proxy" 2>/dev/null || true
      '';

      home.activation.load-litellm = inputs.home-manager.lib.hm.dag.entryAfter ["setupLaunchAgents"] ''
        /bin/launchctl load "$HOME/Library/LaunchAgents/local.litellm-proxy.plist" 2>/dev/null || true
      '';

      launchd.agents.litellm = {
        enable = true;
        config = {
          Label = "local.litellm-proxy";
          ProgramArguments = [ "${runScript}" ];
          RunAtLoad = true;
          KeepAlive = true;
          EnvironmentVariables = {
            LITELLM_BIND = "127.0.0.1:${toString cfg.port}";
            # Cover Docker Desktop (Intel + Apple Silicon) and Homebrew paths.
            PATH = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin";
          } // cfg.envSecrets;
          StandardOutPath = "/Users/${username}/Library/Logs/litellm.log";
          StandardErrorPath = "/Users/${username}/Library/Logs/litellm-error.log";
        };
      };
    };
  };
}
```

#### Nix flake (native build, no container)

The repository is also a Nix flake that builds the proxy as a native Python
environment (no podman/docker). It builds `litellm` from nixpkgs, the
`chutes-e2ee` transport from source, and fetches the `pqcrypto==0.4.0` wheel
that `chutes-e2ee` requires.

```sh
# Development shell (litellm + chutes provider + tools on $PATH)
nix develop

# Build the proxy package
nix build .#

# Run the proxy (generate_config.py then litellm on 127.0.0.1:4000)
nix run .# -- LITELLM_PORT=4000
# or set env vars when launching:
#   LITELLM_HOST=127.0.0.1 LITELLM_PORT=4000 CHUTES_API_KEY=<key> nix run .#
```

To install it on NixOS, import `nixosModules.default` from the flake and use the
options declared in [`./options.nix`](./options.nix):

```nix
{ inputs, ... }: {
  imports = [ inputs.chutes-litellm.nixosModules.default ];

  services.litellm = {
    enable = true;
    apiKeys = {
      # Read from a file at runtime (proxy resolves _PATH natively)
      CHUTES_API_KEY_PATH    = config.age.secrets."litellm/chutes".path;
      ANTHROPIC_API_KEY_PATH = config.age.secrets."litellm/anthropic".path;
      # Or pass a raw key directly
      OPENAI_API_KEY = "sk-...";
    };
  };
}
```

> The `apiKeys` option maps env var names directly to values. The proxy natively
> resolves `_PATH` suffixes at runtime (e.g. `CHUTES_API_KEY_PATH` is read as a
> file), so you can mix raw keys and secret file paths in the same config.

On first build, fill in the `chutes-e2ee` source hash in `flake.nix` (the
placeholder is intentional):

```sh
nix build .#default 2>&1 | grep -A2 'got:'
```

## Usage

The proxy is OpenAI-compatible. Point any tool at `http://127.0.0.1:<port>` with any API key (auth is disabled by default but can be enabled through the [config.template.yml](./config.template.yml) file).

**List available models:**

```sh
curl http://127.0.0.1:4000/v1/models | jq '.data[].id'
```

**Chat (curl):**

```sh
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**opencode config (`~/.config/opencode/opencode.jsonc`):**

```jsonc
"provider": {
  "litellm": {
    "options": {
      "baseURL": "http://127.0.0.1:4000/v1",
      "type": "openai",
      "apiKey": "dummy" // only if didn't configure any in the litellm config and disabled auth
    }
  }
}
```

## Adding models

The models will be automatically queried from each provider API and made available through the LiteLLM proxy. This is done through the [generate_config.py](./generate_config.py) script.

### Generating opencode models

You need to append models manually to opencode because it needs to know the parameters of each model (context, etc.), the [gen_opencode_models.py](./gen_opencode_models.py) can be used to scrape the parameters of the models and automatically add them to your opencode configuration.

## Testing

Requires the proxy to be running. Tests both non-streaming and streaming:

```sh
# Test Chutes E2EE (default)
./proxy_test.py

# Test any other model
./proxy_test.py anthropic/claude-sonnet-4-6
./proxy_test.py openai/gpt-4o
```

## Chutes E2EE provider

Chutes uses end-to-end encrypted transport. `chutes_provider.py` wraps the standard OpenAI client with a custom HTTPX transport that handles encryption transparently. The
`chutes-e2ee-transport` package is installed in the image at build time.

## Open WebUI

You can also start the litellm proxy along with an open-webui container using docker compose. In the open-webui settings, it should then possible to add `http://litellm:4000/v1` to your OpenAI API Connections.

## Structure

```
services/               # Services to launch the proxy on boot
Dockerfile              # Image definition
config.template.yml     # Source of truth for model list — edit this to add/remove models
generate_config.py      # Runs at container startup, expands wildcards into config.generated.yml
chutes_provider.py      # Custom LiteLLM provider for Chutes E2EE transport
proxy_test.py           # Smoke test against the running proxy (streaming + non-streaming)
gen_opencode_models.py  # Generates the opencode.jsonc models block from /model/info
run.sh                  # Runs the container via rootless Podman (used by systemd on NixOS)
```
