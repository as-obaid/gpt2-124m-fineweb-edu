# Keeps two training jobs from writing runs/gpt2_124m_3ep at the same time. Sourced by
# train_flags.sh, so it also reaches 8-hour segments queued before this file existed: Slurm keeps
# the sbatch script from submit time, but reads the files it sources when the job starts.
#
# gpt2fwe_train  8-hour segments on gpu (submit_train.sh). They never wait in the queue for a
#                chunk. One that starts while a chunk runs cancels the chunk and trains. The chunk
#                loses only the steps since its last checkpoint: llm.c writes DONE_<step> after a
#                checkpoint is whole, and it has no SIGTERM handler, so it dies when cancelled.
# gpt2fwe_chunk  1-hour segments on sharing (submit_chunks.sh). One that starts while a segment
#                runs cancels the queued chunks and exits, which ends the chain. Otherwise, once
#                it has run for 5 minutes, it queues one successor behind itself (if none is
#                queued), so the chain keeps going until a segment starts or the run is finished.
#                One that starts while another chunk runs exits.
#
# If squeue keeps failing (the controller is overloaded), a chunk exits rather than guess, and a
# segment cancels every chunk by name and trains. A cancelled or timed-out chunk can sit in
# COMPLETING while Slurm cleans up, so both kinds wait for that, but at most GUARD_WAIT_SECONDS:
# a node stuck in COMPLETING must not hold up training.
GUARD_TRAIN_JOB=gpt2fwe_train
GUARD_CHUNK_JOB=gpt2fwe_chunk
GUARD_WAIT_SECONDS=${GUARD_WAIT_SECONDS:-120}
GUARD_SQUEUE_TRIES=${GUARD_SQUEUE_TRIES:-6}
GUARD_RETRY_SLEEP=${GUARD_RETRY_SLEEP:-10}
GUARD_SUCCESSOR_DELAY=${GUARD_SUCCESSOR_DELAY:-300}

guard_jobs() {  # my jobs named $1 in states $2, other than this one; fails if squeue keeps failing
  local out
  for _ in $(seq "$GUARD_SQUEUE_TRIES"); do
    if out=$(timeout 30 squeue --me --name="$1" -t "$2" -h -o %i 2>/dev/null); then
      printf '%s\n' "$out" | grep -vx -e "${SLURM_JOB_ID:-}" -e '' || true
      return 0
    fi
    sleep "$GUARD_RETRY_SLEEP"
  done
  echo "guard: squeue failed $GUARD_SQUEUE_TRIES times" >&2
  return 1
}

guard_wait_for_chunks() {
  local waited=0 left
  while left=$(guard_jobs "$GUARD_CHUNK_JOB" RUNNING,COMPLETING) && [ -n "$left" ]; do
    if [ "$waited" -ge "$GUARD_WAIT_SECONDS" ]; then
      echo "guard: chunk(s) $(echo $left) still listed after ${GUARD_WAIT_SECONDS}s;" \
           "they were signalled, so training anyway" >&2
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

case "${SLURM_JOB_NAME:-}" in
  "$GUARD_TRAIN_JOB")
    if ! guard_chunks=$(guard_jobs "$GUARD_CHUNK_JOB" RUNNING); then
      echo "guard: cannot list chunks; cancelling all of them by name before training" >&2
      scancel --me --name="$GUARD_CHUNK_JOB" || true
    elif [ -n "$guard_chunks" ]; then
      echo "guard: cancelling 1-hour chunk(s) $(echo $guard_chunks) before this segment trains"
      scancel $guard_chunks
    fi
    guard_wait_for_chunks
    ;;
  "$GUARD_CHUNK_JOB")
    # train.sbatch only checks for a finished run after this file, so stop here, or every chunk
    # would queue another one forever after the last step.
    if [ -e "$RUNS_DIR/gpt2_124m_3ep/DONE_$(printf %08d "$MAX_STEPS")" ]; then
      echo "guard: the run is finished; not queuing another chunk"
      exit 0
    fi
    if ! guard_segments=$(guard_jobs "$GUARD_TRAIN_JOB" RUNNING,COMPLETING) \
       || ! guard_others=$(guard_jobs "$GUARD_CHUNK_JOB" RUNNING); then
      echo "guard: cannot ask Slurm what else is running; exiting rather than risk two writers" >&2
      exit 0
    fi
    if [ -n "$guard_segments" ]; then
      echo "guard: an 8-hour segment is running; cancelling queued chunks and exiting"
      scancel --me --name="$GUARD_CHUNK_JOB" --state=PENDING
      exit 0
    fi
    if [ -n "$guard_others" ]; then
      echo "guard: another chunk is running, exiting"
      exit 0
    fi
    guard_wait_for_chunks
    # Queue the successor only after this chunk has run for GUARD_SUCCESSOR_DELAY seconds. Slurm
    # kills a job's leftover processes when it ends, so a chunk that crashes at startup (say, on a
    # checkpoint llm.c can't resume from) queues nothing instead of looping through the queue.
    (
      sleep "$GUARD_SUCCESSOR_DELAY"
      if guard_pending=$(guard_jobs "$GUARD_CHUNK_JOB" PENDING) && [ -z "$guard_pending" ]; then
        bash "$ROOT/scripts/submit_chunks.sh" 1 "${CHUNK_GPU:-h100}" \
          || echo "guard: could not queue the next chunk; this one trains anyway" >&2
      fi
    ) &
    ;;
esac
