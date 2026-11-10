# Draws logs/figs/monitor<step>.png for every HellaSwag eval in main.log (every 500 steps), in the
# background for the life of a training job. It keys on the eval line, not the checkpoint: llm.c
# writes checkpoint N just before it evaluates step N. Figures missed while no job ran are drawn
# by the next job, each cut off at its own step. Sourced by train_flags.sh, after the guard, so
# queued segments get it too and a chunk the guard turns away starts none. logs/ is git-ignored.
case "${SLURM_JOB_NAME:-}" in
  gpt2fwe_train|gpt2fwe_chunk)
    (
      log=$RUNS_DIR/gpt2_124m_3ep/main.log
      figs=$ROOT/logs/figs
      mkdir -p "$figs"
      while true; do
        for s in $(grep -o '^s:[0-9]* eval' "$log" 2>/dev/null | tr -dc '0-9\n' | sort -un || true); do
          [ -e "$figs/monitor$s.png" ] && continue
          "$PY" -I "$ROOT/scripts/plot_progress.py" "$log" "$figs/monitor$s.png" --upto "$s" \
            --baseline "$RUNS_DIR/hellaswag_gpt2_baseline.txt" >> "$figs/plot_errors.log" 2>&1 || true
        done
        sleep 30
      done
    ) &
    ;;
esac
