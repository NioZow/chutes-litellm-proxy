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

    verifyAttestation = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Enable the fail-closed attestation gate. When true, the proxy verifies
        TEE/GPU evidence for every instance of a chute before encrypting traffic
        (CHUTES_VERIFY_ATTESTATION).
      '';
    };

    verifyQuote = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Require Intel DCAP TDX-quote hardware-root verification
        (CHUTES_VERIFY_QUOTE). Needs the attestation SDKs in the package.
      '';
    };

    verifyGpu = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Require NVIDIA NRAS GPU hardware-root verification
        (CHUTES_VERIFY_GPU). Needs the attestation SDKs in the package.
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

    e2eeApiBase = lib.mkOption {
      type = lib.types.str;
      default = "https://api.chutes.ai";
      description = "E2EE/attestation API base URL (CHUTES_E2EE_API_BASE).";
    };

    e2eeModelsBase = lib.mkOption {
      type = lib.types.str;
      default = "https://llm.chutes.ai";
      description = "Model-listing base URL (CHUTES_E2EE_MODELS_BASE).";
    };

    e2eeHosts = lib.mkOption {
      type = lib.types.str;
      default = "llm.chutes.ai,api.chutes.ai";
      description = "Comma-separated hosts whose traffic is end-to-end encrypted (CHUTES_E2EE_HOSTS).";
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
            "CHUTES_VERIFY_ATTESTATION=${lib.boolToString cfg.verifyAttestation}"
            "CHUTES_VERIFY_QUOTE=${lib.boolToString cfg.verifyQuote}"
            "CHUTES_VERIFY_GPU=${lib.boolToString cfg.verifyGpu}"
            "CHUTES_ATTESTATION_TTL=${toString cfg.attestationTtl}"
            "CHUTES_ATTESTATION_FAILURE_TTL=${toString cfg.attestationFailureTtl}"
            "CHUTES_ATTESTATION_MODEL_MAP_TTL=${toString cfg.attestationModelMapTtl}"
            "CHUTES_CUSTOM_PROVIDER=${lib.boolToString cfg.customProvider}"
            "CHUTES_E2EE_API_BASE=${cfg.e2eeApiBase}"
            "CHUTES_E2EE_MODELS_BASE=${cfg.e2eeModelsBase}"
            "CHUTES_E2EE_HOSTS=${cfg.e2eeHosts}"
          ]
          ++ lib.optional (cfg.configTemplate != null) "LITELLM_TEMPLATE=${toString cfg.configTemplate}"
          ++ lib.optional (cfg.dcapPccsUrl != null) "CHUTES_DCAP_PCCS_URL=${cfg.dcapPccsUrl}"
          ++ lib.optional (cfg.nvidiaNrasUrl != null) "CHUTES_NVIDIA_NRAS_URL=${cfg.nvidiaNrasUrl}"
          ++ apiKeyEnv;
      };
    };

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];
  };
}
