#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Unified interactive downloader for:
#   - Hugging Face models / datasets
#   - ModelScope models
#
# Demo:
#   chmod +x downhub.sh
#   bash downhub.sh
#
# Non-interactive example:
#   SOURCE=hf RESOURCE=model REPO_ID=StarVLA/Qwen3-VL-OFT-Robotwin2 \
#   REAL_BASE=/mnt/data/liwenbo_datas/models \
#   LINK_BASE=$HOME/projects/VLA/starVLA/playground/Pretrained_models \
#   bash downhub.sh
#
# nohup example:
#   nohup bash downhub.sh > downhub.log 2>&1 &
# =========================================================


# =========================
# Defaults (can be overridden by env)
# =========================
SOURCE="${SOURCE:-}"          # hf | ms | auto
RESOURCE="${RESOURCE:-}"      # model | dataset
REPO_ID="${REPO_ID:-}"        # e.g. StarVLA/Qwen3-VL-OFT-Robotwin2
LOCAL_NAME="${LOCAL_NAME:-}"  # local dir name; default: basename(REPO_ID)

REAL_BASE="${REAL_BASE:-}"    # e.g. /mnt/data/liwenbo_datas/models or .../datasets
LINK_BASE="${LINK_BASE:-}"    # e.g. $HOME/projects/VLA/starVLA/playground/Pretrained_models
HF_CACHE_ROOT="${HF_CACHE_ROOT:-}"  # e.g. /mnt/nas_ssd/.../models  -> auto build hf_home/hf_home/hub
HF_HOME="${HF_HOME:-}"
HF_HUB_CACHE="${HF_HUB_CACHE:-}"

MAX_RETRIES="${MAX_RETRIES:-10}"
SLEEP_SECS="${SLEEP_SECS:-20}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
ASSUME_YES="${ASSUME_YES:-0}"  # 1 means auto-confirm in non-interactive runs
BACKGROUND_DOWNLOAD="${BACKGROUND_DOWNLOAD:-0}"  # 1 means spawn a background worker

# You can add your own mirror / proxy / internal gateway probes here.
# These are only connectivity probes, not direct download commands.
PROBE_URLS_DEFAULT=(
  "https://huggingface.co"
  "https://hf-mirror.com"
  "https://modelscope.cn"
  "https://www.modelscope.cn"
)

# =========================
# Helpers
# =========================

color() {
  local c="$1"; shift
  case "$c" in
    red)    printf "\033[31m%s\033[0m\n" "$*" ;;
    green)  printf "\033[32m%s\033[0m\n" "$*" ;;
    yellow) printf "\033[33m%s\033[0m\n" "$*" ;;
    blue)   printf "\033[34m%s\033[0m\n" "$*" ;;
    cyan)   printf "\033[36m%s\033[0m\n" "$*" ;;
    magenta) printf "\033[35m%s\033[0m\n" "$*" ;;
    bold)   printf "\033[1m%s\033[0m\n" "$*" ;;
    *)      printf "%s\n" "$*" ;;
  esac
}

section() {
  color magenta "------------------------------------------------------------"
  color bold "$1"
  color magenta "------------------------------------------------------------"
}

info() {
  color cyan "[INFO] $*"
}

success() {
  color green "[ OK ] $*"
}

warn() {
  color yellow "[WARN] $*"
}

error() {
  color red "[ERR ] $*"
}

print_kv() {
  local key="$1"
  local val="$2"
  printf "\033[1m%-12s\033[0m = %s\n" "$key" "$val"
}

launch_background_download() {
  local ts log_file
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="/tmp/downhub_${LOCAL_NAME}_${ts}.log"

  info "Starting background worker..."
  nohup env \
    ASSUME_YES=1 \
    BACKGROUND_DOWNLOAD=0 \
    SOURCE="$SOURCE" \
    RESOURCE="$RESOURCE" \
    REPO_ID="$REPO_ID" \
    LOCAL_NAME="$LOCAL_NAME" \
    REAL_BASE="$REAL_BASE" \
    LINK_BASE="$LINK_BASE" \
    HF_CACHE_ROOT="$HF_CACHE_ROOT" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    MAX_RETRIES="$MAX_RETRIES" \
    SLEEP_SECS="$SLEEP_SECS" \
    CONNECT_TIMEOUT="$CONNECT_TIMEOUT" \
    bash "$0" >"$log_file" 2>&1 &

  local pid=$!
  success "Background download started. PID=$pid"
  info "Log file: $log_file"
  info "Check progress: tail -f $log_file"
}

ask() {
  local prompt="$1"
  local default="${2:-}"
  local ans=""
  if [[ -n "$default" ]]; then
    # -e enables readline editing (arrow keys, backspace) in interactive TTY.
    read -e -r -p "$prompt [$default]: " ans || true
    echo "${ans:-$default}"
  else
    # -e enables readline editing (arrow keys, backspace) in interactive TTY.
    read -e -r -p "$prompt: " ans || true
    echo "$ans"
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

python_exists() {
  command_exists python || command_exists python3
}

get_python() {
  if command_exists python; then
    echo "python"
  elif command_exists python3; then
    echo "python3"
  else
    return 1
  fi
}

basename_repo() {
  local repo="$1"
  echo "${repo##*/}"
}

default_real_base_for_resource() {
  local resource="$1"
  local fixed_model_base="/mnt/nas_ssd/workspace/wenboli/projects/Wall-X/wallx/models/qwen"
  local cwd
  cwd="$(pwd)"
  if [[ "$resource" == "dataset" ]]; then
    echo "$cwd/playground/_hub_store/datasets"
  else
    echo "$fixed_model_base"
  fi
}

default_link_base_for_resource() {
  local resource="$1"
  local cwd
  cwd="$(pwd)"
  if [[ "$resource" == "dataset" ]]; then
    echo "$cwd/playground/Datasets"
  else
    echo "$cwd/playground/Pretrained_models"
  fi
}

validate_positive_int() {
  local v="$1"
  [[ "$v" =~ ^[0-9]+$ ]] && [[ "$v" -gt 0 ]]
}

derive_hf_cache_root_from_real_base() {
  local base="$1"
  if [[ "$base" == */models ]]; then
    echo "$base"
  elif [[ "$base" == */models/* ]]; then
    echo "${base%%/models/*}/models"
  elif [[ "$base" == */datasets ]]; then
    echo "$base"
  elif [[ "$base" == */datasets/* ]]; then
    echo "${base%%/datasets/*}/datasets"
  else
    echo "$base"
  fi
}

ensure_hf_cache_paths() {
  if [[ -z "$HF_CACHE_ROOT" ]]; then
    HF_CACHE_ROOT="$(derive_hf_cache_root_from_real_base "$REAL_BASE")"
  fi

  if [[ -z "$HF_HOME" ]]; then
    HF_HOME="$HF_CACHE_ROOT/hf_home"
  fi

  if [[ -z "$HF_HUB_CACHE" ]]; then
    HF_HUB_CACHE="$HF_HOME/hub"
  fi
}

normalize_source_resource() {
  SOURCE="$(echo "$SOURCE" | tr '[:upper:]' '[:lower:]')"
  RESOURCE="$(echo "$RESOURCE" | tr '[:upper:]' '[:lower:]')"
}

recompute_paths() {
  REAL_DIR="$REAL_BASE/$LOCAL_NAME"
  LINK_DIR="$LINK_BASE/$LOCAL_NAME"
}

edit_params_before_download() {
  section "Edit Parameters"
  info "Choose one field to modify:"
  echo "  1) SOURCE"
  echo "  2) RESOURCE"
  echo "  3) REPO_ID"
  echo "  4) LOCAL_NAME"
  echo "  5) REAL_BASE"
  echo "  6) LINK_BASE"
  echo "  7) HF_CACHE_ROOT"
  echo "  8) Done"

  local choice
  choice="$(ask "Select number" "8")"

  case "$choice" in
    1)
      SOURCE="$(ask "Choose source (hf/ms/auto)" "$SOURCE")"
      ;;
    2)
      RESOURCE="$(ask "Choose resource type (model/dataset)" "$RESOURCE")"
      ;;
    3)
      REPO_ID="$(ask "Enter repo id" "$REPO_ID")"
      ;;
    4)
      LOCAL_NAME="$(ask "Local directory name" "$LOCAL_NAME")"
      ;;
    5)
      REAL_BASE="$(ask "Real storage base directory" "$REAL_BASE")"
      ;;
    6)
      LINK_BASE="$(ask "Symlink base directory" "$LINK_BASE")"
      ;;
    7)
      HF_CACHE_ROOT="$(ask "HF cache root directory" "$HF_CACHE_ROOT")"
      ;;
    8)
      return 0
      ;;
    *)
        warn "Unknown choice: $choice"
      ;;
  esac

  normalize_source_resource

  if [[ "$LOCAL_NAME" == *"/"* ]]; then
    color yellow "[Normalize] LOCAL_NAME contains '/'. Using basename instead."
    LOCAL_NAME="$(basename_repo "$LOCAL_NAME")"
  fi

  if [[ -z "$LOCAL_NAME" ]]; then
    LOCAL_NAME="$(basename_repo "$REPO_ID")"
  fi

  recompute_paths
  ensure_hf_cache_paths
}

run_initial_wizard_with_back() {
  local step=0
  local ans=""

  section "Interactive Setup Wizard"
  info "Input 'b' to go back to previous step."

  while true; do
    case "$step" in
      0)
        ans="$(ask "Run connectivity probe now? (y/n)" "y")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=0
          continue
        fi
        if [[ "$ans" =~ ^[Yy]$ ]]; then
          probe_urls "$CONNECT_TIMEOUT"
        fi
        step=1
        ;;

      1)
        ans="$(ask "Choose source (hf/ms/auto)" "${SOURCE:-auto}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=0
          continue
        fi
        SOURCE="$ans"
        step=2
        ;;

      2)
        ans="$(ask "Choose resource type (model/dataset)" "${RESOURCE:-model}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=1
          continue
        fi
        RESOURCE="$ans"
        normalize_source_resource
        step=3
        ;;

      3)
        ans="$(ask "Enter repo id (e.g. StarVLA/Qwen3-VL-OFT-Robotwin2 or IPEC-COMMUNITY/fractal20220817_data_lerobot)" "${REPO_ID:-}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=2
          continue
        fi
        REPO_ID="$ans"
        step=4
        ;;

      4)
        ans="$(ask "Local directory name" "${LOCAL_NAME:-$(basename_repo "$REPO_ID")}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=3
          continue
        fi
        LOCAL_NAME="$ans"
        if [[ "$LOCAL_NAME" == *"/"* ]]; then
          color yellow "[Normalize] LOCAL_NAME contains '/'. Using basename instead."
          LOCAL_NAME="$(basename_repo "$LOCAL_NAME")"
        fi
        step=5
        ;;

      5)
        ans="$(ask "Real storage base directory" "${REAL_BASE:-$(default_real_base_for_resource "$RESOURCE")}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=4
          continue
        fi
        REAL_BASE="$ans"
        step=6
        ;;

      6)
        ans="$(ask "Symlink base directory" "${LINK_BASE:-$(default_link_base_for_resource "$RESOURCE")}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=5
          continue
        fi
        LINK_BASE="$ans"
        step=7
        ;;

      7)
        ans="$(ask "HF cache root directory" "${HF_CACHE_ROOT:-$(derive_hf_cache_root_from_real_base "$REAL_BASE")}")"
        if [[ "$ans" =~ ^[Bb]$ ]]; then
          step=6
          continue
        fi
        HF_CACHE_ROOT="$ans"
        break
        ;;
    esac
  done
}

probe_one_url() {
  local url="$1"
  local timeout="$2"

  if command_exists curl; then
    if curl -L -I --connect-timeout "$timeout" --max-time $((timeout + 8)) \
      -sS "$url" >/tmp/downhub_probe.out 2>/tmp/downhub_probe.err; then
      success "Reachable: $url"
      return 0
    else
      warn "Unreachable: $url"
      return 1
    fi
  else
    local py
    py="$(get_python)"
    "$py" - "$url" "$timeout" <<PY >/dev/null 2>&1
import sys, urllib.request
url = sys.argv[1]
timeout = int(sys.argv[2])
try:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as _:
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
    if [[ $? -eq 0 ]]; then
      success "Reachable: $url"
      return 0
    else
      warn "Unreachable: $url"
      return 1
    fi
  fi
}

probe_one_url_speed() {
  local url="$1"
  local timeout="$2"
  local speed_url="$url/robots.txt"

  if ! command_exists curl; then
    warn "curl not found, skip speed test for $speed_url"
    return 1
  fi

  local out
  out="$(curl -L -o /dev/null -sS \
    --connect-timeout "$timeout" \
    --max-time $((timeout + 20)) \
    -w "%{http_code} %{time_total} %{speed_download}" \
    "$speed_url" || true)"

  local code total speed
  code="$(echo "$out" | awk '{print $1}')"
  total="$(echo "$out" | awk '{print $2}')"
  speed="$(echo "$out" | awk '{print $3}')"

  if [[ -n "$code" && "$code" != "000" && -n "$speed" ]]; then
    local kbps mbps
    kbps="$(awk -v s="$speed" 'BEGIN{printf "%.1f", s/1024}')"
    mbps="$(awk -v s="$speed" 'BEGIN{printf "%.2f", s/1024/1024}')"
    info "Speed sample $speed_url -> HTTP $code, ${kbps} KB/s (${mbps} MB/s), t=${total}s"
    return 0
  fi

  warn "Speed sample failed for $speed_url"
  return 1
}

probe_urls() {
  local timeout="$1"
  section "[Step 1] Connectivity + Speed Probe"
  info "Testing reachable domains and sampling transfer speed"
  local ok_count=0
  local speed_count=0
  for url in "${PROBE_URLS_DEFAULT[@]}"; do
    if probe_one_url "$url" "$timeout"; then
      ok_count=$((ok_count + 1))
      if probe_one_url_speed "$url" "$timeout"; then
        speed_count=$((speed_count + 1))
      fi
    fi
  done
  echo
  info "Reachable count: $ok_count / ${#PROBE_URLS_DEFAULT[@]}"
  info "Speed sampled : $speed_count / $ok_count reachable hosts"
  echo
}

require_hf_tool() {
  if command_exists hf; then
    echo "hf"
    return 0
  elif command_exists huggingface-cli; then
    echo "huggingface-cli"
    return 0
  else
    color red "ERROR: Neither 'hf' nor 'huggingface-cli' found in PATH."
    color yellow "Tip: pip install -U huggingface_hub"
    return 1
  fi
}

require_modelscope_python() {
  local py
  py="$(get_python)" || {
    color red "ERROR: python/python3 not found."
    return 1
  }

  if "$py" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("modelscope") else 1)
PY
  then
    echo "$py"
    return 0
  else
    color red "ERROR: modelscope python package not found."
    color yellow "Tip: pip install -U modelscope"
    return 1
  fi
}

ensure_parent_dir() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
}

safe_ln_sfn() {
  local src="$1"
  local dst="$2"

  ensure_parent_dir "$dst"

  if [[ -e "$dst" && ! -L "$dst" ]]; then
    color red "ERROR: $dst exists and is NOT a symlink."
    color yellow "Please move/remove it manually, or choose another LINK_BASE."
    return 1
  fi

  ln -sfn "$src" "$dst"
}

show_summary() {
  section "Configuration Summary"
  print_kv "SOURCE" "$SOURCE"
  print_kv "RESOURCE" "$RESOURCE"
  print_kv "REPO_ID" "$REPO_ID"
  print_kv "LOCAL_NAME" "$LOCAL_NAME"
  print_kv "REAL_BASE" "$REAL_BASE"
  print_kv "LINK_BASE" "$LINK_BASE"
  print_kv "HF_HOME" "$HF_HOME"
  print_kv "HF_HUB_CACHE" "$HF_HUB_CACHE"
  print_kv "MAX_RETRIES" "$MAX_RETRIES"
  print_kv "SLEEP_SECS" "$SLEEP_SECS"
  print_kv "BG_DOWNLOAD" "$BACKGROUND_DOWNLOAD"
  print_kv "REAL_DIR" "$REAL_DIR"
  print_kv "LINK_DIR" "$LINK_DIR"
}

# =========================
# Download implementations
# =========================

hf_download_once() {
  local tool="$1"
  local resource="$2"
  local repo_id="$3"
  local real_dir="$4"

  mkdir -p "$real_dir"

  export HF_HOME HF_HUB_CACHE

  if [[ "$tool" == "hf" ]]; then
    if [[ "$resource" == "dataset" ]]; then
      hf download "$repo_id" --repo-type dataset --local-dir "$real_dir"
    else
      hf download "$repo_id" --local-dir "$real_dir"
    fi
  else
    if [[ "$resource" == "dataset" ]]; then
      huggingface-cli download "$repo_id" --repo-type dataset --local-dir "$real_dir"
    else
      huggingface-cli download "$repo_id" --local-dir "$real_dir"
    fi
  fi
}

hf_download_with_retry() {
  local resource="$1"
  local repo_id="$2"
  local real_dir="$3"

  local tool
  tool="$(require_hf_tool)" || return 1

  local rc=0
  local sleep_secs="$SLEEP_SECS"

  for i in $(seq 1 "$MAX_RETRIES"); do
    local log="/tmp/downhub_hf_attempt_${i}.log"
    color blue "[HF] Attempt $i/$MAX_RETRIES ... log=$log"

    set +e
    hf_download_once "$tool" "$resource" "$repo_id" "$real_dir" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -eq 0 ]]; then
      color green "[HF] Download finished."
      return 0
    fi

    if grep -q "429" "$log"; then
      color yellow "[HF] Hit rate limit / auth-related issue."
      color yellow "Tip: hf auth login  或者 export HF_TOKEN=..."
    else
      color yellow "[HF] Download failed with rc=$rc"
    fi

    color yellow "[HF] Sleep ${sleep_secs}s then retry..."
    sleep "$sleep_secs"
    if [[ "$sleep_secs" -lt 600 ]]; then
      sleep_secs=$((sleep_secs * 2))
    fi
  done

  return 1
}

ms_model_download_once() {
  local py="$1"
  local repo_id="$2"
  local real_dir="$3"

  mkdir -p "$real_dir"

  "$py" - "$repo_id" "$real_dir" <<'PY'
import sys
repo_id = sys.argv[1]
real_dir = sys.argv[2]

from modelscope import snapshot_download
snapshot_download(repo_id, local_dir=real_dir)
print(f"[MS] snapshot_download done: {real_dir}")
PY
}

ms_model_download_with_retry() {
  local repo_id="$1"
  local real_dir="$2"

  local py
  py="$(require_modelscope_python)" || return 1

  local rc=0
  local sleep_secs="$SLEEP_SECS"

  for i in $(seq 1 "$MAX_RETRIES"); do
    local log="/tmp/downhub_ms_attempt_${i}.log"
    color blue "[MS] Attempt $i/$MAX_RETRIES ... log=$log"

    set +e
    ms_model_download_once "$py" "$repo_id" "$real_dir" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -eq 0 ]]; then
      color green "[MS] Download finished."
      return 0
    fi

    color yellow "[MS] Download failed with rc=$rc"
    color yellow "[MS] Sleep ${sleep_secs}s then retry..."
    sleep "$sleep_secs"
    if [[ "$sleep_secs" -lt 600 ]]; then
      sleep_secs=$((sleep_secs * 2))
    fi
  done

  return 1
}


# =========================
# Interactive input
# =========================

if [[ "$ASSUME_YES" == "1" ]]; then
  info "ASSUME_YES=1 => skip interactive setup wizard."
else
  run_initial_wizard_with_back
fi

# Normalize after wizard / env injection.
normalize_source_resource

if [[ -z "$LOCAL_NAME" && -n "$REPO_ID" ]]; then
  LOCAL_NAME="$(basename_repo "$REPO_ID")"
fi

# Guardrail: LOCAL_NAME should be a leaf directory name, not a full path.
if [[ "$LOCAL_NAME" == *"/"* ]]; then
  color yellow "[Normalize] LOCAL_NAME contains '/'. Using basename instead."
  LOCAL_NAME="$(basename_repo "$LOCAL_NAME")"
fi

if [[ -z "$REAL_BASE" ]]; then
  REAL_BASE="$(default_real_base_for_resource "$RESOURCE")"
fi

if [[ -z "$LINK_BASE" ]]; then
  LINK_BASE="$(default_link_base_for_resource "$RESOURCE")"
fi

ensure_hf_cache_paths

if ! validate_positive_int "$MAX_RETRIES"; then
  error "MAX_RETRIES must be a positive integer. Current: $MAX_RETRIES"
  exit 2
fi

if ! validate_positive_int "$SLEEP_SECS"; then
  error "SLEEP_SECS must be a positive integer. Current: $SLEEP_SECS"
  exit 2
fi

if ! validate_positive_int "$CONNECT_TIMEOUT"; then
  error "CONNECT_TIMEOUT must be a positive integer. Current: $CONNECT_TIMEOUT"
  exit 2
fi

if [[ -z "$REPO_ID" ]]; then
  error "REPO_ID cannot be empty."
  exit 2
fi

recompute_paths

show_summary

if [[ "$ASSUME_YES" == "1" ]]; then
  info "ASSUME_YES=1 => skip confirmation prompt."
else
  while true; do
    ans="$(ask "Proceed with download? (y/n/b=back)" "y")"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      break
    fi
    if [[ "$ans" =~ ^[Nn]$ ]]; then
      warn "Aborted."
      exit 0
    fi
    if [[ "$ans" =~ ^[Bb]$ ]]; then
      edit_params_before_download
      show_summary
      continue
    fi
    warn "Please input y / n / b"
  done

  bg_ans="$(ask "Download in background? (y/n)" "n")"
  if [[ "$bg_ans" =~ ^[Yy]$ ]]; then
    BACKGROUND_DOWNLOAD=1
  fi
fi

if [[ "$BACKGROUND_DOWNLOAD" == "1" ]]; then
  launch_background_download
  exit 0
fi


# =========================
# Main dispatch
# =========================

case "$RESOURCE" in
  model|dataset) ;;
  *)
    error "RESOURCE must be model or dataset."
    exit 2
    ;;
esac

case "$SOURCE" in
  hf)
    section "Download"
    info "Using Hugging Face"
    hf_download_with_retry "$RESOURCE" "$REPO_ID" "$REAL_DIR"
    ;;
  ms)
    if [[ "$RESOURCE" != "model" ]]; then
      error "This script currently supports ModelScope for models only."
      warn "For datasets, use SOURCE=hf or extend the script with a dedicated ModelScope dataset flow."
      exit 3
    fi
    section "Download"
    info "Using ModelScope"
    ms_model_download_with_retry "$REPO_ID" "$REAL_DIR"
    ;;
  auto)
    if [[ "$RESOURCE" == "dataset" ]]; then
      section "Download"
      info "AUTO mode: dataset defaults to Hugging Face"
      hf_download_with_retry "$RESOURCE" "$REPO_ID" "$REAL_DIR"
    else
      section "Download"
      info "AUTO mode: try Hugging Face first, then ModelScope"
      if hf_download_with_retry "$RESOURCE" "$REPO_ID" "$REAL_DIR"; then
        :
      else
        warn "HF failed, fallback to ModelScope..."
        ms_model_download_with_retry "$REPO_ID" "$REAL_DIR"
      fi
    fi
    ;;
  *)
    error "SOURCE must be hf, ms, or auto."
    exit 2
    ;;
esac


# =========================
# Symlink + verify
# =========================

section "Symlink + Verify"
info "Create/update symlink:"
echo "  $LINK_DIR -> $REAL_DIR"
safe_ln_sfn "$REAL_DIR" "$LINK_DIR"

info "Symlink detail:"
ls -ld "$LINK_DIR" || true

if [[ -d "$LINK_DIR" ]]; then
  success "Directory exists via symlink: $LINK_DIR"
else
  error "Missing link target: $LINK_DIR"
  exit 4
fi

info "Sample files:"
find "$LINK_DIR" -maxdepth 2 -type f | head -n 20 || true

success "Done."