{
  description = "LiteLLM proxy with the Chutes E2EE provider, built natively with Nix (no podman/docker).";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      lib = nixpkgs.lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = lib.genAttrs systems;

      # -----------------------------------------------------------------------
      # The Chutes E2EE provider relies on `chutes-e2ee`, which pins
      # `pqcrypto==0.4.0`.  `pqcrypto` ships *only* precompiled wheels on PyPI
      # (no sdist) and is not packaged in nixpkgs, so we fetch the wheel that
      # matches the interpreter ABI and host architecture.  See
      # https://pypi.org/project/pqcrypto/0.4.0/ for all available wheels.
      # -----------------------------------------------------------------------
      pqcryptoWheel =
        pkgs: python:
        let
          wheels = {
            # pythonVersion -> { platform -> { url, hash } }
            "3.13" = {
              x86_64 = {
                url = "https://files.pythonhosted.org/packages/bb/01/9c57f061b6798bc478cb552d3653bff4f447b34f989e0f49326fa558719e/pqcrypto-0.4.0-cp313-cp313-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl";
                hash = "sha256-HuKVTsqEFJ4pBkufgvEwvRrF4NljsKNeSKfJc3wuOU4=";
              };
              aarch64 = {
                url = "https://files.pythonhosted.org/packages/0d/d6/98abb4e44df8c361a1a0f67e0306e6f0257c3ca8aef6bedefc0956047949/pqcrypto-0.4.0-cp313-cp313-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
                hash = "sha256-I5zBtgzfVPRefxPEXKq3hFAM6OTn+f3oZXufresSUAg=";
              };
            };
            "3.14" = {
              x86_64 = {
                url = "https://files.pythonhosted.org/packages/57/60/98ed5d9d959c3b5c9d604a832d12e30d249c2cffce3496330fe3855a1599/pqcrypto-0.4.0-cp314-cp314-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl";
                hash = "sha256-EDsFhlgjI/RD78BxiTx3W92cdIUbiFI8CG7zuuK1NvE=";
              };
              aarch64 = {
                url = "https://files.pythonhosted.org/packages/ad/6d/fa97f2003a8b2124970bd238aedaebe36364401042ac49e929e70abb24bd/pqcrypto-0.4.0-cp314-cp314-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
                hash = "sha256-Vz7BHhA8cYX0spivWSSn7iTsduHG+18RrzMrc2CVCtg=";
              };
            };
          };
          platform =
            if pkgs.stdenv.hostPlatform.isx86_64
            then "x86_64"
            else if pkgs.stdenv.hostPlatform.isAarch64
            then "aarch64"
            else throw "pqcrypto wheel not available for ${pkgs.stdenv.hostPlatform.system}";
          spec = (wheels.${python.pythonVersion} or (throw "pqcrypto wheel not configured for Python ${python.pythonVersion}; add it in flake.nix (pqcryptoWheel)")).${platform};
        in
        pkgs.fetchurl {
          inherit (spec) url hash;
        };

      # -----------------------------------------------------------------------
      # Python environment shared by the dev shell and the packaged proxy.
      # -----------------------------------------------------------------------
      mkChutesPython =
        pkgs:
        let
          python = pkgs.python3;

          pqcrypto = python.pkgs.buildPythonPackage {
            pname = "pqcrypto";
            version = "0.4.0";
            format = "wheel";
            src = pqcryptoWheel pkgs python;
            # The wheel is prebuilt but still imports `cffi` at runtime.
            propagatedBuildInputs = [ python.pkgs.cffi ];
          };

          chutes-e2ee = python.pkgs.buildPythonPackage rec {
            pname = "chutes-e2ee";
            version = "0.1.1";
            format = "pyproject";
            src = pkgs.fetchFromGitHub {
              owner = "niozow";
              repo = "chutes-e2ee-transport";
              rev = "5630286de88d0797bf605279f19b974db25bded7";
              hash = "sha256-ow43baqoahITGaeSxMoSyN1hXX4qGVSqlfrb8nFnFFk=";
            };
            nativeBuildInputs = [ python.pkgs.hatchling ];
            propagatedBuildInputs = [
              python.pkgs.cryptography
              python.pkgs.httpx
              pqcrypto
            ];
          };
        in
        python.withPackages (
          ps: [
            # litellm with the `proxy` (+ runtime) extras so the HTTP proxy works.
            (ps.litellm.overridePythonAttrs (old: {
              dependencies =
                old.dependencies
                ++ (old.optional-dependencies.proxy or [])
                ++ (old.optional-dependencies.proxy-runtime or []);
              optional-dependencies = {};
            }))
            ps.openai
            chutes-e2ee
          ]
        );

      # -----------------------------------------------------------------------
      # The runnable proxy: bundles the repo's provider/config scripts plus the
      # Python environment, and launches `generate_config.py` before `litellm`.
      # -----------------------------------------------------------------------
      mkProxy =
        pkgs:
        let
          env = mkChutesPython pkgs;
        in
        pkgs.stdenv.mkDerivation {
          pname = "litellm-proxy";
          version = "0.1.0";
          src = lib.cleanSource ./.;

          installPhase = ''
            runHook preInstall

            mkdir -p $out/bin $out/share/litellm
            cp chutes_provider.py generate_config.py config.template.yml $out/share/litellm/

            cat > $out/bin/litellm-proxy <<'EOF'
            #!${pkgs.stdenv.shell}
            set -eu
            export PATH="${env}/bin:${pkgs.coreutils}/bin"

            SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
            SHARE="$SELF_DIR/../share/litellm"
            export PYTHONPATH="$SHARE:''${PYTHONPATH:-}"

            TEMPLATE="''${LITELLM_TEMPLATE:-$SHARE/config.template.yml}"
            OUTPUT="''${LITELLM_OUTPUT:-''${TMPDIR:-/tmp}/litellm-config.generated.yml}"
            HOST="''${LITELLM_HOST:-127.0.0.1}"
            PORT="''${LITELLM_PORT:-4000}"

            LITELLM_TEMPLATE="$TEMPLATE" LITELLM_OUTPUT="$OUTPUT" \
              python "$SHARE/generate_config.py"

            exec litellm --config "$OUTPUT" --host "$HOST" --port "$PORT"
            EOF
            chmod +x $out/bin/litellm-proxy

            runHook postInstall
          '';

          meta = {
            description = "LiteLLM proxy with the Chutes E2EE provider";
            mainProgram = "litellm-proxy";
            license = lib.licenses.mit;
          };
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = mkProxy pkgs;
          litellm = mkProxy pkgs;
          python = mkChutesPython pkgs;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/litellm-proxy";
        };
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          proxy = mkProxy pkgs;
          python = mkChutesPython pkgs;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
              pkgs.ruff
              pkgs.git
            ];

            shellHook = ''
              export LITELLM_HOST="''${LITELLM_HOST:-127.0.0.1}"
              export LITELLM_PORT="''${LITELLM_PORT:-4000}"
              echo "LiteLLM proxy dev shell. Run: ./proxy_test.py  (or)  ${proxy}/bin/litellm-proxy"
            '';
          };
        }
      );

      # NixOS module. `pkgs` here is the host's pkgs; we wrap options.nix so
      # that `services.litellm.package` defaults to this flake's proxy build.
      nixosModules.default =
        {
          lib,
          pkgs,
          ...
        }:
        {
          imports = [ ./options.nix ];
          services.litellm.package = lib.mkDefault (mkProxy pkgs);
        };
    };
}