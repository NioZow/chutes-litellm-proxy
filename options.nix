# NixOS module for the LiteLLM proxy (with the Chutes E2EE provider).
#
# Import this from a NixOS configuration and enable it with:
#
#   services.litellm = {
#     enable = true;
#     apiKeys = {
#       CHUTES_API_KEY_PATH     = config.age.secrets."litellm/chutes".path;
#       ANTHROPIC_API_KEY_PATH  = config.age.secrets."litellm/anthropic".path;
#       OPENAI_API_KEY          = "sk-abc123";
#     };
#   };
#
# Values may be either a raw string (forwarded as-is) or a path (e.g. a secret
# file). The key is the exact environment variable name that the proxy will see.
# Use *_PATH suffixes when you want the proxy to read the key from a file at
# runtime; use the plain *_API_KEY name when you want to pass the key directly.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.litellm;

  # Forward apiKeys directly into the service environment.
  apiKeyEnv = lib.mapAttrsToList (name: value: "${name}=${toString value}") cfg.apiKeys;
in {
  options.services.litellm = {
    enable = lib.mkEnableOption "LiteLLM proxy (Chutes E2EE)";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The litellm-proxy package (defaults to this flake's build).";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Interface the proxy binds to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 4000;
      description = "Port the proxy listens on.";
    };

    apiKeys = lib.mkOption {
      type = lib.types.attrsOf (lib.types.either lib.types.str lib.types.path);
      default = {};
      example = {
        CHUTES_API_KEY_PATH = "/run/secrets/chutes";
        OPENAI_API_KEY = "sk-raw-key-string";
      };
      description = ''
        Mapping of environment variable names to their values. Values may be raw
        strings (passed through as-is) or paths (e.g. secret files). The proxy
        natively supports variables ending in `_PATH`: it reads the file and
        treats the content as the underlying key.
      '';
    };

    configTemplate = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Optional path to a custom config.template.yml. When null, the template
        bundled in the package is used.
      '';
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the configured port in the firewall (only useful when host is not 127.0.0.1).";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.litellm = {
      description = "LiteLLM proxy (Chutes E2EE)";
      wantedBy = ["multi-user.target"];
      after = ["network-online.target"];
      wants = ["network-online.target"];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/litellm-proxy";
        Restart = "on-failure";
        RestartSec = 5;
        DynamicUser = true;
        StateDirectory = "litellm";
        StateDirectoryMode = "0700";
        AmbientCapabilities = [];
        CapabilityBoundingSet = [];
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = ["/var/lib/litellm"];
        Environment =
          [
            "PYTHONUNBUFFERED=1"
            "LITELLM_HOST=${cfg.host}"
            "LITELLM_PORT=${toString cfg.port}"
            "LITELLM_OUTPUT=/var/lib/litellm/config.generated.yml"
          ]
          ++ lib.optional (cfg.configTemplate != null) "LITELLM_TEMPLATE=${toString cfg.configTemplate}"
          ++ apiKeyEnv;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];
  };
}
