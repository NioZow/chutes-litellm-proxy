# NixOS module for the LiteLLM proxy (with the Chutes E2EE provider).
#
# Import this from a NixOS configuration and enable it with:
#
#   services.litellm = {
#     enable = true;
#     apiKeys = {
#       CHUTES_API_KEY  = config.age.secrets."litellm/chutes".path;
#       ANTHROPIC_API_KEY = config.age.secrets."litellm/anthropic".path;
#       OPENAI_API_KEY  = config.age.secrets."litellm/openai".path;
#     };
#   };
#
# Each value of `apiKeys` is a path to a file containing the raw key value
# (no trailing newline, no `KEY=` prefix). The service reads them at runtime.

{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.litellm;

  # Build a launch script that (a) sources each secret file into the matching
  # environment variable, (b) points the wrapper at the configured host/port and
  # any custom config template, and (c) execs the proxy.  The secret paths are
  # baked into the script (paths are not secret), but the values are read only
  # at runtime, so plaintext keys never land in the Nix store.
  envExports = lib.concatMapStringsSep "\n" (
    name: ''
      export ${name}="$(cat '${cfg.apiKeys.${name}}')"
    ''
  ) (lib.attrNames cfg.apiKeys);

  startScript = pkgs.writeShellScript "litellm-start" ''
    set -eu
    ${envExports}
    export LITELLM_HOST='${cfg.host}'
    export LITELLM_PORT='${toString cfg.port}'
    export LITELLM_OUTPUT='/var/lib/litellm/config.generated.yml'
    ${lib.optionalString (cfg.configTemplate != null) "export LITELLM_TEMPLATE='${cfg.configTemplate}'"}
    exec '${cfg.package}/bin/litellm-proxy'
  '';
in
{
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
      type = lib.types.attrsOf lib.types.path;
      default = {};
      example = {
        CHUTES_API_KEY = "/run/secrets/chutes";
        ANTHROPIC_API_KEY = "/run/secrets/anthropic";
      };
      description = ''
        Mapping of environment variable name (e.g. `CHUTES_API_KEY`) to the path
        of a file containing that key's raw value.
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
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = startScript;
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
        ReadWritePaths = [ "/var/lib/litellm" ];
        Environment = [
          "PYTHONUNBUFFERED=1"
        ];
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
  };
}