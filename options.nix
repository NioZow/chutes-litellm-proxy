# NixOS / home-manager module for the LiteLLM proxy (with the Chutes E2EE
# provider).
#
# The module is a function of a small record supplied by `flake.nix`, so the
# exact same option set (and CLI wiring) is reused for three deployments:
#
#   * scope = "system", isHomeManager = false, isDarwin = false -> NixOS root service
#   * scope = "user",   isHomeManager = false, isDarwin = false -> NixOS user  service
#   * scope = "user",   isHomeManager = true                   -> home-manager user service
#
# `mkService` is already bound to the matching `nix-services` schema. `scope`
# must be a *literal* (it decides the shape of the emitted fragment:
# `systemd.services` vs `systemd.user.services` vs `launchd.agents`), which is
# why each deployment gets its own module instance instead of a `scope` option.
#
# Enable it from the host/home config with:
#
#   services.litellm = {
#     enable = true;
#     apiKeys = {
#       CHUTES_API_KEY_PATH    = config.age.secrets."litellm/chutes".path;
#       ANTHROPIC_API_KEY_PATH = config.age.secrets."litellm/anthropic".path;
#       OPENAI_API_KEY         = "sk-abc123";
#     };
#   };
#
# Values may be either a raw string (forwarded as-is) or a path (e.g. a secret
# file). The key is the exact environment variable name that the proxy will see.
# Use *_PATH suffixes when you want the proxy to read the key from a file at
# runtime; use the plain *_API_KEY name when you want to pass the key directly.
{
  mkService,
  scope ? "system",
  isHomeManager ? false,
  isDarwin ? false,
  # NixOS-only: enables `networking.firewall` handling. Never set under HM.
  withFirewall ? (scope == "system"),
}: {
  config,
  lib,
  ...
}: let
  cfg = config.services.litellm;
  isSystem = scope == "system";

  # Forward apiKeys directly into the service environment. Values may be raw
  # strings (forwarded as-is) or paths (e.g. secret files).
  apiKeyEnv = lib.mapAttrs (name: value: toString value) cfg.apiKeys;

  # Where the generated config is written.
  #   systemd (system) -> `%S` expands to /var/lib; StateDirectory=litellm
  #                       creates /var/lib/litellm.
  #   systemd (user)   -> `%S` expands to ~/.local/state; StateDirectory=litellm
  #                       creates ~/.local/state/litellm.
  #   launchd          -> no `%` specifier expansion, so use an absolute path.
  outputPath =
    if isHomeManager && isDarwin
    then "${config.home.homeDirectory}/.local/state/litellm/config.generated.yml"
    else "%S/litellm/config.generated.yml";

  environment =
    {
      PYTHONUNBUFFERED = "1";
      # Use the model cost map bundled in the litellm wheel instead of fetching
      # it from raw.githubusercontent.com at import time. Keeps startup offline
      # (and behind an allowlisting forward proxy) and matches the test suite.
      LITELLM_LOCAL_MODEL_COST_MAP = "true";
      LITELLM_HOST = cfg.host;
      LITELLM_PORT = toString cfg.port;
      LITELLM_OUTPUT = outputPath;
      CHUTES_TEE_MEASUREMENTS_TTL = toString cfg.teeMeasurementsTtl;
      CHUTES_ATTESTATION_TTL = toString cfg.attestationTtl;
      CHUTES_ATTESTATION_FAILURE_TTL = toString cfg.attestationFailureTtl;
      CHUTES_ATTESTATION_MODEL_MAP_TTL = toString cfg.attestationModelMapTtl;
      CHUTES_CUSTOM_PROVIDER = lib.boolToString cfg.customProvider;
      # The Chutes API and model-listing endpoints are fixed
      # (https://api.chutes.ai and https://llm.chutes.ai); they are not
      # configurable.
      CHUTES_LOG_LEVEL = cfg.logLevel;
    }
    // lib.optionalAttrs (cfg.configTemplate != null) {LITELLM_TEMPLATE = toString cfg.configTemplate;}
    // lib.optionalAttrs (cfg.teeMeasurementsFile != null) {CHUTES_TEE_MEASUREMENTS_FILE = toString cfg.teeMeasurementsFile;}
    // lib.optionalAttrs (cfg.dcapPccsUrl != null) {CHUTES_DCAP_PCCS_URL = cfg.dcapPccsUrl;}
    // lib.optionalAttrs (cfg.nvidiaNrasUrl != null) {CHUTES_NVIDIA_NRAS_URL = cfg.nvidiaNrasUrl;}
    // lib.optionalAttrs (cfg.logFile != null) {CHUTES_LOG_FILE = cfg.logFile;}
    // lib.optionalAttrs (cfg.httpProxy != null) {
      HTTP_PROXY = cfg.httpProxy;
      http_proxy = cfg.httpProxy;
    }
    // lib.optionalAttrs (cfg.httpsProxy != null) {
      HTTPS_PROXY = cfg.httpsProxy;
      https_proxy = cfg.httpsProxy;
    }
    // lib.optionalAttrs (cfg.allProxy != null) {
      ALL_PROXY = cfg.allProxy;
      all_proxy = cfg.allProxy;
    }
    // lib.optionalAttrs (cfg.noProxy != null) {
      NO_PROXY = cfg.noProxy;
      no_proxy = cfg.noProxy;
    }
    // apiKeyEnv;

  # systemd hardening, tuned per scope. `DynamicUser` and the capability
  # directives are only valid for system services. User services already run
  # unprivileged, so they get a conservative subset (no `ProtectSystem`/
  # `ProtectHome`, which could block writes under the user's state dir).
  hardening =
    if isSystem
    then {
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
    }
    else {
      StateDirectory = "litellm";
      StateDirectoryMode = "0700";
      NoNewPrivileges = true;
      PrivateTmp = true;
    };
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

    httpProxy = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "http://10.200.0.9:3128";
      description = ''
        HTTP proxy URL exported to the service as both `HTTP_PROXY` and
        `http_proxy`. When null, neither variable is set. Every outbound
        provider request made by the proxy (model listing, attestation, chat
        completions) honours this.
      '';
    };

    httpsProxy = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "http://10.200.0.9:3128";
      description = ''
        HTTPS proxy URL exported to the service as both `HTTPS_PROXY` and
        `https_proxy`. When null, neither variable is set. Usually the same as
        `httpProxy` for a plain HTTP CONNECT proxy.
      '';
    };

    allProxy = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "http://10.200.0.9:3128";
      description = ''
        Catch-all proxy URL exported to the service as both `ALL_PROXY` and
        `all_proxy`. When null, neither variable is set.
      '';
    };

    noProxy = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "localhost,127.0.0.1,.trs,10.200.0.0/24";
      description = ''
        Comma-separated bypass list exported to the service as both `NO_PROXY`
        and `no_proxy`. When null, neither variable is set. Use it for internal
        hosts and the proxy endpoint itself so they are reached directly.
      '';
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

    teeMeasurementsFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Optional local TEE measurement policy (YAML, Chutes schema). When set it
        is the sole measurement source and the public
        /servers/tee/measurements endpoint is not used
        (CHUTES_TEE_MEASUREMENTS_FILE). Hardware-root attestation and
        measurement matching are always enforced.
      '';
    };

    teeMeasurementsTtl = lib.mkOption {
      type = lib.types.int;
      default = 3600;
      description = ''
        How long the published TEE measurement list is cached (seconds;
        CHUTES_TEE_MEASUREMENTS_TTL). A last-known-good list is kept for
        transient endpoint failures.
      '';
    };

    attestationTtl = lib.mkOption {
      type = lib.types.int;
      default = 300;
      description = ''
        How long verified instance lists are cached (seconds). Higher values
        reduce scanning overhead but delay reaction to new or revoked instances
        (CHUTES_ATTESTATION_TTL).
      '';
    };

    attestationFailureTtl = lib.mkOption {
      type = lib.types.int;
      default = 30;
      description = ''
        How long a failed attestation verdict is cached (seconds). Prevents
        retry storms against broken chutes (CHUTES_ATTESTATION_FAILURE_TTL).
      '';
    };

    attestationModelMapTtl = lib.mkOption {
      type = lib.types.int;
      default = 300;
      description = ''
        How long the model-to-chute id lookup is cached (seconds)
        (CHUTES_ATTESTATION_MODEL_MAP_TTL).
      '';
    };

    dcapPccsUrl = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Intel/Phala collateral (PCCS) base URL for DCAP quote verification.
        Leave null to use the default (CHUTES_DCAP_PCCS_URL).
      '';
    };

    nvidiaNrasUrl = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        NVIDIA Remote Attestation Service (NRAS) endpoint for GPU verification.
        Leave null to use the default (CHUTES_NVIDIA_NRAS_URL).
      '';
    };

    customProvider = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Serve Chutes models through the self-contained custom provider
        (chutes_e2ee/<model>) instead of the built-in chutes provider
        (CHUTES_CUSTOM_PROVIDER).
      '';
    };

    logLevel = lib.mkOption {
      type = lib.types.str;
      default = "info";
      description = ''
        Logging level for the chutes_litellm logger: debug, info, warning,
        error or critical (CHUTES_LOG_LEVEL).
      '';
    };

    logFile = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = ''
        Path to a log file for the proxy. When null, the default
        <literal>~/.local/state/chutes_litellm/proxy.log</literal> is used (or
        nothing if the directory is not writable). Set to <literal>"none"</literal>
        to disable file logging entirely (CHUTES_LOG_FILE).
      '';
    };
  };

  config = lib.mkIf cfg.enable (
    (mkService (
      {
        name = "litellm";
        description = "LiteLLM proxy (Chutes E2EE)";
        command = "${cfg.package}/bin/litellm-proxy";
        inherit scope;
        environment = environment;
        extraSystemdServiceConfig = hardening;
        # launchd (home-manager) needs an absolute log dir; systemd ignores it.
        logDir =
          if isHomeManager
          then "${config.home.homeDirectory}/Library/Logs"
          else null;
      }
      // lib.optionalAttrs isSystem {
        after = ["network-online.target"];
        wants = ["network-online.target"];
      }
    ))
    // lib.optionalAttrs withFirewall {
      networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];
    }
  );
}
