#!/usr/bin/env bash
# Runs the litellm-proxy container via rootless Podman (preferred) or Docker.
#
# All configuration is supplied through environment variables. No CLI arguments
# are accepted.
#
# Environment variables:
#   LITELLM_BIND      Host:port to publish on (default: 127.0.0.1:4000:4000).
#                     Multiple ports can be passed as a comma-separated list.
#   RUNTIME           Container engine: podman | docker | auto (default: auto).
#   *_API_KEY         Raw API keys. Any variable ending in _API_KEY is copied
#                     into the container as-is.
#   *_API_KEY_PATH    File path containing a raw API key. The file is read and
#                     forwarded into the container under the base name (e.g.
#                     ANTHROPIC_API_KEY_PATH becomes ANTHROPIC_API_KEY).
#
# Examples:
#   LITELLM_BIND=127.0.0.1:4000 ./run.sh
#   LITELLM_BIND=0.0.0.0:4000,192.168.1.10:4000 RUNTIME=docker ./run.sh
#   ANTHROPIC_API_KEY_PATH=/run/secrets/anthropic ./run.sh
#
set -eu

RUNTIME=${RUNTIME:-auto}
IMAGE_NAME="litellm-proxy"
CONTAINER_NAME="$IMAGE_NAME"

TMPENV=$(mktemp)
trap 'rm -f "$TMPENV"' EXIT

if [[ "$RUNTIME" == "auto" ]] || [[ -z "$RUNTIME" ]]; then
  if command -v podman >/dev/null 2>&1; then
    RUNTIME=podman
  else
    RUNTIME=docker
  fi
fi

# Resolve LITELLM_BIND into -p flags
PUBLISH_ARGS=()
IFS=',' read -ra BINDINGS <<< "${LITELLM_BIND:-127.0.0.1:4000}"
for binding in "${BINDINGS[@]}"; do
  PUBLISH_ARGS+=(-p "${binding}:4000")
done

while IFS='=' read -r env_key env_val; do
  if [[ "$env_key" == *_API_KEY_PATH ]]; then
    base_key="${env_key%_PATH}"
    if [[ -f "$env_val" ]]; then
      printf '%s=%s\n' "$base_key" "$(cat "$env_val")" >>"$TMPENV"
    fi
  elif [[ "$env_key" == *_API_KEY ]]; then
    printf '%s=%s\n' "$env_key" "$env_val" >>"$TMPENV"
  fi
done < <(env)

exec $RUNTIME run --rm \
  --name "$CONTAINER_NAME" \
  "${PUBLISH_ARGS[@]}" \
  --env-file "$TMPENV" \
  "$IMAGE_NAME"
