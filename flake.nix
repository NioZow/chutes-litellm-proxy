{
  description = "LiteLLM proxy with the Chutes E2EE provider, built natively with Nix (no podman/docker).";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nix-services.url = "github:niozow/nix-service";
  };

  outputs = {
    self,
    nixpkgs,
    nix-services,
  }: let
    lib = nixpkgs.lib;
    # aarch64-darwin is supported because pqcrypto 0.4.0 ships a macOS arm64
    # wheel; x86_64-darwin is intentionally excluded (no macOS Intel wheel
    # exists on PyPI).
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "aarch64-darwin"
    ];
    forAllSystems = lib.genAttrs systems;

    # -----------------------------------------------------------------------
    # The Chutes E2EE provider relies on `chutes-e2ee`, which pins
    # `pqcrypto==0.4.0`.  `pqcrypto` ships *only* precompiled wheels on PyPI
    # (no sdist) and is not packaged in nixpkgs, so we fetch the wheel that
    # matches the interpreter ABI, OS and host architecture.  See
    # https://pypi.org/project/pqcrypto/0.4.0/ for all available wheels.
    #
    # macOS support: pqcrypto 0.4.0 only publishes macOS ARM64 wheels
    # (`macosx_11_0_arm64`), no Intel ones — so Darwin builds are limited to
    # `aarch64-darwin`.
    # -----------------------------------------------------------------------
    pqcryptoWheel = pkgs: python: let
      wheels = {
        # pythonVersion -> { platform -> { url, hash } }
        # platform is one of: linux-x86_64, linux-aarch64, darwin-aarch64
        "3.13" = {
          "linux-x86_64" = {
            url = "https://files.pythonhosted.org/packages/bb/01/9c57f061b6798bc478cb552d3653bff4f447b34f989e0f49326fa558719e/pqcrypto-0.4.0-cp313-cp313-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl";
            hash = "sha256-HuKVTsqEFJ4pBkufgvEwvRrF4NljsKNeSKfJc3wuOU4=";
          };
          "linux-aarch64" = {
            url = "https://files.pythonhosted.org/packages/0d/d6/98abb4e44df8c361a1a0f67e0306e6f0257c3ca8aef6bedefc0956047949/pqcrypto-0.4.0-cp313-cp313-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
            hash = "sha256-I5zBtgzfVPRefxPEXKq3hFAM6OTn+f3oZXufresSUAg=";
          };
          "darwin-aarch64" = {
            url = "https://files.pythonhosted.org/packages/dd/5d/dd3752d741f8772eb7f0a49226bb02966c1fd7b85c7aa83027213a2e9973/pqcrypto-0.4.0-cp313-cp313-macosx_11_0_arm64.whl";
            hash = "sha256-7ejwrfsw3zXw/t+rMnd5J5L1+uXrZsS0s9envYmmR3g=";
          };
        };
        "3.14" = {
          "linux-x86_64" = {
            url = "https://files.pythonhosted.org/packages/57/60/98ed5d9d959c3b5c9d604a832d12e30d249c2cffce3496330fe3855a1599/pqcrypto-0.4.0-cp314-cp314-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl";
            hash = "sha256-EDsFhlgjI/RD78BxiTx3W92cdIUbiFI8CG7zuuK1NvE=";
          };
          "linux-aarch64" = {
            url = "https://files.pythonhosted.org/packages/ad/6d/fa97f2003a8b2124970bd238aedaebe36364401042ac49e929e70abb24bd/pqcrypto-0.4.0-cp314-cp314-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl";
            hash = "sha256-Vz7BHhA8cYX0spivWSSn7iTsduHG+18RrzMrc2CVCtg=";
          };
          "darwin-aarch64" = {
            url = "https://files.pythonhosted.org/packages/cf/f2/335ca98cf32d0723b86e672cd0b06da92bd091e3599b881b70e93f4faef4/pqcrypto-0.4.0-cp314-cp314-macosx_11_0_arm64.whl";
            hash = "sha256-YIfG5G+2Afp2rkC8PrVhiefOPam1B0BmkxW+P7V1cBA=";
          };
        };
      };
      platform =
        if pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isx86_64
        then "linux-x86_64"
        else if pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isAarch64
        then "linux-aarch64"
        else if pkgs.stdenv.hostPlatform.isDarwin && pkgs.stdenv.hostPlatform.isAarch64
        then "darwin-aarch64"
        else throw "pqcrypto wheel not available for ${pkgs.stdenv.hostPlatform.system}";
      spec = (wheels.${python.pythonVersion} or (throw "pqcrypto wheel not configured for Python ${python.pythonVersion}; add it in flake.nix (pqcryptoWheel)")).${platform};
    in
      pkgs.fetchurl {
        inherit (spec) url hash;
      };

    # -----------------------------------------------------------------------
    # Optional hardware-attestation SDKs (Intel DCAP + NVIDIA).  None are in
    # nixpkgs, so vendor them from PyPI like pqcrypto.  dcap-qvl is a Rust
    # extension but ships an `abi3` wheel (cp38+, so Python-version agnostic);
    # nv-attestation-sdk / nv-local-gpu-verifier are pure-Python wheels.
    # Everything here is only referenced when `withAttestation = true`, so the
    # base environment still builds with neither SDK present.
    # -----------------------------------------------------------------------
    wheelPlatform = pkgs:
      if pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isx86_64
      then "linux-x86_64"
      else if pkgs.stdenv.hostPlatform.isLinux && pkgs.stdenv.hostPlatform.isAarch64
      then "linux-aarch64"
      else if pkgs.stdenv.hostPlatform.isDarwin && pkgs.stdenv.hostPlatform.isAarch64
      then "darwin-aarch64"
      else throw "no prebuilt wheel for ${pkgs.stdenv.hostPlatform.system}";

    dcapQvlWheel = pkgs: let
      wheels = {
        "linux-x86_64" = {
          url = "https://files.pythonhosted.org/packages/df/e9/ba237d9da48fa794bbf257af17b1ec60bdc652813f16103ca27cde88eda4/dcap_qvl-0.6.3-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
          hash = "sha256-OscqDvLrflkaxmZGs4maDgCKRr90PQdxbDTXVAIUwpU=";
        };
        "linux-aarch64" = {
          url = "https://files.pythonhosted.org/packages/9d/ed/fa6bf937ca1c9d9c12c970c4c642730e764026b762014790f3086f2cefc9/dcap_qvl-0.6.3-cp38-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl";
          hash = "sha256-e7RBF33RP3HqatasqCeVaoU1HQ5hZzztRqrBkhrFkmE=";
        };
        "darwin-aarch64" = {
          url = "https://files.pythonhosted.org/packages/ae/5c/5812299b6a8ee7ee7eb40a9f7cfa8f59e5b9e1bc06e1a4911aa3978f8087/dcap_qvl-0.6.3-cp38-abi3-macosx_11_0_arm64.whl";
          hash = "sha256-Gha7i56yHyfhDxiGc1Nn2oLKgOM0446o30wNgMEhE3M=";
        };
      };
    in
      pkgs.fetchurl (wheels.${wheelPlatform pkgs});

    # Pure-Python NVIDIA wheels (same bytes on every platform).
    nvAttestationSdkWheel = pkgs:
      pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/91/36/3be7582492af9282143fd145bbd7b6af27038f0ffcaec626121c74fecca7/nv_attestation_sdk-2.7.3-py3-none-any.whl";
        hash = "sha256-fDTTR1OjZ/OBdHKYoE03p6msfHaGymEgyLOoBdXUj2g=";
      };

    nvLocalGpuVerifierWheel = pkgs:
      pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/34/73/6ef820c55fd563e43d2f92b1944ed9c2b98a830c6d951c8dca268a754340/nv_local_gpu_verifier-2.7.3-py3-none-any.whl";
        hash = "sha256-JyWc49goyhAtsX9LcMe4XBiCpV+ZFT63Bvyxhm2w/no=";
      };

    # -----------------------------------------------------------------------
    # Python environment shared by the dev shell and the packaged proxy.
    # -----------------------------------------------------------------------
    mkChutesPython = pkgs: {withAttestation ? false}: let
      python = pkgs.python3;

      pqcrypto = python.pkgs.buildPythonPackage {
        pname = "pqcrypto";
        version = "0.4.0";
        format = "wheel";
        src = pqcryptoWheel pkgs python;
        # The wheel is prebuilt but still imports `cffi` at runtime.
        propagatedBuildInputs = [python.pkgs.cffi];
      };

      # TODO(local-dev): the `instance_filter` hook and fork tests live in the
      # local clone at ./chutes-e2ee-transport (see pyproject [tool.uv.sources]),
      # which is *ahead* of this pinned rev.  Leave this rev as the committed
      # default; once the fork changes are pushed, update `rev`/`hash` here (and
      # delete the [tool.uv.sources] override in pyproject.toml + re-lock).
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
        nativeBuildInputs = [python.pkgs.hatchling];
        propagatedBuildInputs = [
          python.pkgs.cryptography
          python.pkgs.httpx
          pqcrypto
        ];
      };

      # litellm 1.97.0+ imports `expression` (Expression.tagged_union) in its
      # MCP outbound-credentials module at process start, but nixpkgs ships no
      # `expression` for Python 3.14, so it was silently dropped from the `proxy`
      # extra. Package it here (pure-python wheel) so the proxy starts at all.
      expression = python.pkgs.buildPythonPackage rec {
        pname = "expression";
        version = "5.7.0";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/07/9d/790e25dcba0b299f9a756ae2dcc52a705be07bd1c3fd54267ade0521bea6/expression-5.7.0-py3-none-any.whl";
          hash = "sha256-2NkDy53cslLb1kYS4ym9hvCddwx4Eur4+cwLn45kgL0=";
        };
        propagatedBuildInputs = [python.pkgs.typing-extensions];
        doCheck = false;
      };

      # --- optional hardware-root attestation (only with withAttestation) ---
      # Intel DCAP quote verification (Rust abi3 wheel, no Python deps).
      dcap-qvl = python.pkgs.buildPythonPackage rec {
        pname = "dcap-qvl";
        version = "0.6.3";
        format = "wheel";
        src = dcapQvlWheel pkgs;
        doCheck = false;
      };

      # CVE-2024-23342 (the "Minerva" attack) is a timing side-channel in
      # python-ecdsa's *private-key* operations (ECDSA signing / nonce
      # derivation); it requires the secret key to be present.  NVIDIA's GPU
      # and switch attestation verifiers only ever verify signatures: both
      # import just `VerifyingKey` + `BadSignatureError` and call
      # `VerifyingKey.from_pem(cert_pubkey).verify(...)` — see
      # nv_local_gpu_verifier/verifier/attestation/__init__.py.  Verification
      # uses public key material only and is not exposed to the side channel.
      # nixpkgs flags the whole package insecure regardless, so we strip the
      # advisory for this verification-only dependency instead of globally
      # permitting insecure packages.
      ecdsa = python.pkgs.ecdsa.overridePythonAttrs (old: {
        meta = (old.meta or {}) // {knownVulnerabilities = [];};
      });

      # NVIDIA local GPU verifier (`import verifier`), a runtime dep of the
      # nv-attestation-sdk import chain.  Its PyPI metadata pins ancient
      # versions (cryptography==43, signxml==3.2); we use nixpkgs' current
      # versions, which is what the code actually runs against.
      nv-local-gpu-verifier = python.pkgs.buildPythonPackage rec {
        pname = "nv-local-gpu-verifier";
        version = "2.7.3";
        format = "wheel";
        src = nvLocalGpuVerifierWheel pkgs;
        propagatedBuildInputs =
          [ecdsa]
          ++ (with python.pkgs; [
            asn1
            cryptography
            lxml
            nvidia-ml-py
            pyjwt
            pyopenssl
            requests
            signxml
            urllib3
            xmlschema
          ]);
        # The wheel's METADATA pins exact versions (ecdsa==0.18.0,
        # cryptography==43.0.1, ...) which we intentionally do not honour: we
        # run against nixpkgs' current versions, so the runtime-dependency
        # check can never pass.  `doCheck` does not suppress this pre-install
        # hook; `dontCheckRuntimeDeps` does.
        dontCheckRuntimeDeps = true;
        doCheck = false;
      };

      # NVIDIA NRAS GPU-evidence verification.  Only the runtime import chain is
      # provided (the PyPI metadata also lists pylint/twine/pytest as runtime
      # deps, which are not needed and are deliberately omitted).
      nv-attestation-sdk = python.pkgs.buildPythonPackage rec {
        pname = "nv-attestation-sdk";
        version = "2.7.3";
        format = "wheel";
        src = nvAttestationSdkWheel pkgs;
        propagatedBuildInputs =
          [ecdsa]
          ++ (with python.pkgs; [
            cryptography
            nv-local-gpu-verifier
            nvidia-ml-py
            pyjwt
            pyopenssl
            requests
            signxml
            urllib3
            xmlschema
          ]);
        dontCheckRuntimeDeps = true;
        doCheck = false;
      };
    in
      python.withPackages (
        ps:
          [
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
            expression
          ]
          ++ lib.optionals withAttestation [
            dcap-qvl
            nv-attestation-sdk
          ]
      );

    # -----------------------------------------------------------------------
    # The runnable proxy: bundles the chutes_litellm package (config template
    # included) plus the Python environment, and launches it via the CLI entry
    # point (generate_config -> E2EE install -> litellm).
    # -----------------------------------------------------------------------
    mkProxy = pkgs: {withAttestation ? false}: let
      env = mkChutesPython pkgs {inherit withAttestation;};
    in
      pkgs.stdenv.mkDerivation {
        pname = "litellm-proxy";
        version = "0.1.0";
        src = ./.;

        installPhase = ''
          runHook preInstall

          mkdir -p $out/bin $out/share/litellm/src
          cp -r src/chutes_litellm $out/share/litellm/src/

          cat > $out/bin/litellm-proxy <<'EOF'
          #!${pkgs.stdenv.shell}
          set -eu
          export PATH="${env}/bin:${pkgs.coreutils}/bin"

          SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
          SHARE="$SELF_DIR/../share/litellm"
          export PYTHONPATH="$SHARE/src:''${PYTHONPATH:-}"

          # Runs generate_config -> E2EE transport install -> litellm.
          # LITELLM_TEMPLATE / LITELLM_OUTPUT / LITELLM_HOST / LITELLM_PORT are
          # honoured via the environment (see chutes_litellm.cli).
          exec python -m chutes_litellm.cli
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

    # -----------------------------------------------------------------------
    # Shared service-module builder. `scope`, `isDarwin` and `homeManager` are
    # *constants* per output: they decide which unit schema `nix-services` emits,
    # and that shape is forced during module merge, so they must not be read
    # from `pkgs`/`config` (that would recurse). See the nix-services README.
    # -----------------------------------------------------------------------
    mkServiceModule = {
      scope,
      isDarwin ? false,
      homeManager ? false,
    }: {
      lib,
      pkgs,
      ...
    }: {
      imports = [
        ((import ./options.nix) {
          mkService = nix-services.lib.mkService {
            inherit lib isDarwin homeManager;
            username = "root";
          };
          inherit scope isDarwin;
          isHomeManager = homeManager;
          withFirewall = !homeManager;
        })
      ];
      services.litellm.package = lib.mkDefault (mkProxy pkgs {withAttestation = true;});
    };
  in {
    packages = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        default = mkProxy pkgs {};
        litellm = mkProxy pkgs {};
        # Same proxy with the optional Intel DCAP / NVIDIA SDKs baked in.
        attestation = mkProxy pkgs {withAttestation = true;};
        python = mkChutesPython pkgs {};
        python-attestation = mkChutesPython pkgs {withAttestation = true;};
      }
    );

    apps = forAllSystems (system: {
      default = {
        type = "app";
        program = "${self.packages.${system}.default}/bin/litellm-proxy";
      };
    });

    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
        python = mkChutesPython pkgs {};
      in {
        default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.ruff
            pkgs.git
          ];

          shellHook = ''
            export name="chutes-litellm-proxy"

            if [ ! -f .venv/bin/activate ]; then
              uv venv
              uv pip install -e .
              uv pip install -e ".[dev]"
            fi

            source .venv/bin/activate

            export LITELLM_HOST="''${LITELLM_HOST:-127.0.0.1}"
            export LITELLM_PORT="''${LITELLM_PORT:-4000}"
            export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      }
    );

    # -----------------------------------------------------------------------
    # Service modules, all sharing options.nix:
    #
    #   nixosModules.default  NixOS system service (root)
    #   nixosModules.user     NixOS user service (systemd --user, declared in NixOS config)
    #   homeModules.default   home-manager user service (Linux, systemd --user)
    #   homeModules.darwin    home-manager user service (macOS, launchd)
    # -----------------------------------------------------------------------
    nixosModules = {
      default = mkServiceModule {scope = "system";};
      user = mkServiceModule {scope = "user";};
    };

    homeModules = {
      default = mkServiceModule {
        scope = "user";
        homeManager = true;
      };
      darwin = mkServiceModule {
        scope = "user";
        homeManager = true;
        isDarwin = true;
      };
    };
  };
}
