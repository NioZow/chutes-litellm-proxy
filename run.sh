#!/usr/bin/env bash
# usage: run.sh [BIND=interface:port ...] [KEY=filepath ...]
# example:
#   - run.sh BIND=127.0.0.1:4000 BIND=192.168.90.1:8888 ANTHROPIC_API_KEY=/tmp/anthropic_api_key
#   - run.sh BIND=0.0.0.0:4000 OPENAI_API_KEY=/tmp/openai_api_key
# note: also reads the runtime from the "RUNTIME" env variable -> podman | docker | auto
# note: if no interface is specified `127.0.0.1:4000:4000`
# note: also add all env variables like "*_API_KEY" to the container.
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

PUBLISH_ARGS=()
for pair in "$@"; do
  KEY="${pair%%=*}"
  VALUE="${pair#*=}"
  if [[ "$KEY" == "BIND" ]]; then
    PUBLISH_ARGS+=(-p "${VALUE}:4000")
  else
    printf '%s=%s\n' "${KEY}" "$(cat "${VALUE}")" >>"$TMPENV"
  fi
done

while IFS='=' read -r env_key env_val; do
  if [[ "$env_key" == *_API_KEY ]]; then
    printf '%s=%s\n' "$env_key" "$env_val" >>"$TMPENV"
  fi
done < <(env)

if [[ ${#PUBLISH_ARGS[@]} -eq 0 ]]; then
  # default interface
  PUBLISH_ARGS+=(-p "127.0.0.1:4000:4000")
fi

exec $RUNTIME run --rm \
  --name "$CONTAINER_NAME" \
  "${PUBLISH_ARGS[@]}" \
  --env-file "$TMPENV" \
  "$IMAGE_NAME"
