#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"${PYTHON_BIN}" "${ROOT_DIR}/baseline/utils/vast_setup.py" "$@"

ENV_FILE="${ROOT_DIR}/.env.vast"
if [[ -f "${ENV_FILE}" ]]; then
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # shellcheck disable=SC1090
    . "${ENV_FILE}"
    printf 'Loaded %s into the current shell.\n' "${ENV_FILE}"
  else
    printf 'Source %s to load the exported environment into your shell.\n' "${ENV_FILE}"
  fi
fi
