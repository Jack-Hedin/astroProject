#!/bin/bash
set -e

# Always run from the script's directory so relative paths and inputs resolve
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="MegaRun"
mkdir -p "$BASE_DIR"

# Defaults
DEFAULT_NFRAMES=150
DEFAULT_SEED=41

NFRAMES=$DEFAULT_NFRAMES
RUN_NAME=""
SEED=$DEFAULT_SEED

FRAMES_SPECIFIED=0

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --frames|-f|--nframe)
      NFRAMES="$2"
      FRAMES_SPECIFIED=1
      shift 2
      ;;
    --output|--name|-o|--run-dir)
      RUN_NAME="$2"
      shift 2
      ;;
    --seed|-s)
      SEED="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--frames <int>] [--output <dir>] [--seed <int>]"
      echo ""
      echo "Options:"
      echo "  --frames, -f <int>   Number of frames (default: $DEFAULT_NFRAMES)"
      echo "  --output, -o <dir>   Output folder inside MegaRun/ (default: auto)"
      echo "  --seed,   -s <int>   Random seed (default: $DEFAULT_SEED)"
      echo "  --help,   -h         Show this help message"
      exit 0
      ;;
    *)
      echo "Error: Unknown argument '$1'"
      echo "Usage: $0 [--frames <int>] [--output <dir>] [--seed <int>]"
      exit 1
      ;;
  esac
done

# If no run name specified:
if [[ -z "$RUN_NAME" ]]; then
  if [[ $FRAMES_SPECIFIED -eq 1 ]]; then
    RUN_NAME="frames_${NFRAMES}"
    # If frames_X already exists, append a run suffix so multiple runs don't overwrite
    if [[ -d "$BASE_DIR/$RUN_NAME" ]]; then
      i=1
      while [[ -d "$BASE_DIR/${RUN_NAME}_run${i}" ]]; do
        ((i++))
      done
      RUN_NAME="${RUN_NAME}_run${i}"
    fi
  else
    # Find next available run_N
    i=1
    while [[ -d "$BASE_DIR/run_$i" ]]; do
      ((i++))
    done
    RUN_NAME="run_$i"
  fi
fi

# Determine full path to RUN_DIR
if [[ "$RUN_NAME" = /* ]]; then
  RUN_DIR="$RUN_NAME"
elif [[ "$RUN_NAME" = "$BASE_DIR"/* ]]; then
  RUN_DIR="$RUN_NAME"
elif [[ "$RUN_NAME" = "$BASE_DIR" ]]; then
  RUN_DIR="$BASE_DIR"
else
  RUN_DIR="$BASE_DIR/$RUN_NAME"
fi

mkdir -p "$RUN_DIR"

echo "=========================================================="
echo "Starting run: $RUN_DIR"
echo "Frames      : $NFRAMES"
echo "Seed        : $SEED"
echo "=========================================================="

DIR_LS50="$RUN_DIR/ls50"
DIR_LS150="$RUN_DIR/ls150"
DIR_LS150_MF="$RUN_DIR/ls150_MF"
DIR_LS_C="$RUN_DIR/ls_c"

echo "[1/4] Running Broadband_solve stage 1 (ls50)..."
python3 Broadband_solve.py --nframe "$NFRAMES" --seed "$SEED" --niter 50 --whiten 1 --restart 0 --output "$DIR_LS50"

echo "[2/4] Running Broadband_solve stage 2 (ls150)..."
python3 Broadband_solve.py --nframe "$NFRAMES" --seed "$SEED" --niter 150 --whiten 2 --restart 1 --output "$DIR_LS150" --estimate_path "$DIR_LS50/estimate.fits"

echo "[3/4] Running Broadband_solve stage 3 MF (ls150_MF)..."
python3 Broadband_solve.py --nframe "$NFRAMES" --seed "$SEED" --niter 0 --whiten 2 --restart 1 --output "$DIR_LS150_MF" --estimate_path "$DIR_LS150/estimate.fits" --filter 1 --support 1

echo "Detecting sources..."
python3 detect_sources_new.py "$DIR_LS150_MF" --center 255.5 251

echo "Extracting pixel spectrum with IRAF CL..."
#(cd "$RUN_DIR" && cl < "$SCRIPT_DIR/pix.cl")

echo "[4/4] Running Broadband_solve stage 4 companion (ls_c)..."
python3 Broadband_solve.py --nframe "$NFRAMES" --seed "$SEED" --niter 0 --whiten 2 --restart 1 --output "$DIR_LS_C" --estimate_path "$DIR_LS150/estimate.fits" --companion 1 --support 1

# Copy outputs into RUN_DIR
#cp "$DIR_LS_C"/companion_spectrum* "$RUN_DIR"/
#cp "$DIR_LS150"/delta_* "$RUN_DIR"/

# Generate comparison plots inside RUN_DIR
#(cd "$RUN_DIR" && python3 "$SCRIPT_DIR/plot_companion_spectrum.py")

echo "=========================================================="
echo "Run complete! All outputs saved in ${RUN_DIR}"
echo "=========================================================="
