#!/bin/bash
# Tests scripts/train_guard.sh against fake squeue and scancel, so it runs without Slurm.
# Usage: bash tests/test_train_guard.sh [GUARD]   (GUARD defaults to scripts/train_guard.sh)
set -uo pipefail
GUARD=${1:-$(cd "$(dirname "$0")/.." && pwd)/scripts/train_guard.sh}
T=$(mktemp -d)
mkdir -p "$T/bin"

cat > "$T/bin/squeue" <<'EOF'
#!/bin/bash
# Prints job IDs from running_<name> and completing_<name>, for the states asked for with -t.
# A completing job is gone after it has been listed once, as if its epilog finished.
[ -n "${FAKE_SQUEUE_FAIL:-}" ] && exit 1
states=""
while [ $# -gt 0 ]; do
  case $1 in --name=*) name=${1#--name=};; -t) states=$2; shift;; esac
  shift
done
case ",$states," in *,RUNNING,*) [ -f "$FAKE_DIR/running_$name" ] && cat "$FAKE_DIR/running_$name";; esac
case ",$states," in *,PENDING,*) [ -f "$FAKE_DIR/pending_$name" ] && cat "$FAKE_DIR/pending_$name";; esac
case ",$states," in *,COMPLETING,*) [ -f "$FAKE_DIR/stuck_$name" ] && cat "$FAKE_DIR/stuck_$name";; esac
case ",$states," in *,COMPLETING,*) [ -f "$FAKE_DIR/completing_$name" ] && { cat "$FAKE_DIR/completing_$name"; rm "$FAKE_DIR/completing_$name"; };; esac
exit 0
EOF

cat > "$T/bin/scancel" <<'EOF'
#!/bin/bash
echo "$*" >> "$FAKE_DIR/scancel.log"
for a in "$@"; do
  case $a in
    [0-9]*) for f in "$FAKE_DIR"/running_*; do [ -f "$f" ] && { grep -vx "$a" "$f" > "$f.tmp" || true; mv "$f.tmp" "$f"; }; done;;
  esac
done
EOF
printf '#!/bin/bash\nshift\nexec "$@"\n' > "$T/bin/timeout"
chmod +x "$T/bin/squeue" "$T/bin/scancel" "$T/bin/timeout"

fails=0
export GUARD_SUCCESSOR_DELAY=0
# run_case NAME JOB_NAME JOB_ID "running train" "running chunks" EXPECT_CONTINUE EXPECT_SCANCEL
#          [completing train] [completing chunk]
# Env: PENDING_CHUNKS = queued chunk IDs; STUCK_CHUNKS = chunks that never leave COMPLETING; WANT_SUBMIT = expected submit_chunks.sh calls ("args;")
run_case() {
  local name=$1 job_name=$2 job_id=$3 train=$4 chunk=$5 want_cont=$6 want_cancel=$7
  local train_cg=${8:-} chunk_cg=${9:-}
  local d; d=$(mktemp -d)
  printf '%s' "$train" | tr ' ' '\n' | sed '/^$/d' > "$d/running_gpt2fwe_train"
  printf '%s' "$chunk" | tr ' ' '\n' | sed '/^$/d' > "$d/running_gpt2fwe_chunk"
  [ -n "$train_cg" ] && printf '%s\n' "$train_cg" > "$d/completing_gpt2fwe_train"
  [ -n "$chunk_cg" ] && printf '%s\n' "$chunk_cg" > "$d/completing_gpt2fwe_chunk"
  : > "$d/scancel.log"; : > "$d/submit.log"
  [ -n "${STUCK_CHUNKS:-}" ] && printf '%s\n' "$STUCK_CHUNKS" > "$d/stuck_gpt2fwe_chunk"
  [ -n "${PENDING_CHUNKS:-}" ] && printf '%s\n' "$PENDING_CHUNKS" > "$d/pending_gpt2fwe_chunk"
  mkdir -p "$d/root/scripts" "$d/runs/gpt2_124m_3ep"
  [ -n "${DONE_STEP:-}" ] && : > "$d/runs/gpt2_124m_3ep/DONE_$DONE_STEP"
  printf '#!/bin/bash\necho "$*" >> "%s/submit.log"\n' "$d" > "$d/root/scripts/submit_chunks.sh"
  local out status
  out=$(FAKE_DIR=$d ROOT=$d/root RUNS_DIR=$d/runs MAX_STEPS=56352 PATH="$T/bin:$PATH" \
        SLURM_JOB_NAME=$job_name SLURM_JOB_ID=$job_id \
        bash -c 'set -euo pipefail; source "$0"; echo CONTINUED
                 # KILL_EARLY: the job dies before the successor delay, as Slurm kills its leftovers
                 if [ -n "${KILL_EARLY:-}" ]; then kill $(jobs -p) 2>/dev/null || true; fi
                 wait' "$GUARD" 2>&1)
  status=$?
  local got_cont=no; [[ $out == *CONTINUED* ]] && got_cont=yes
  local got_cancel; got_cancel=$(tr '\n' ';' < "$d/scancel.log")
  local got_submit; got_submit=$(tr '\n' ';' < "$d/submit.log")
  if [ "$status" = 0 ] && [ "$got_cont" = "$want_cont" ] && [ "$got_cancel" = "$want_cancel" ] \
     && [ "$got_submit" = "${WANT_SUBMIT:-}" ]; then
    echo "PASS  $name"
  else
    echo "FAIL  $name: status=$status continued=$got_cont (want $want_cont) scancel='$got_cancel' (want '$want_cancel') submit='$got_submit' (want '${WANT_SUBMIT:-}')"
    echo "      output: $out"
    fails=$((fails + 1))
  fi
  rm -rf "$d"
}

run_case "segment, nothing else running: trains"        gpt2fwe_train 999 "999" ""    yes ""
run_case "segment, chunk running: cancels it, trains"   gpt2fwe_train 999 "999" "111" yes "111;"
run_case "chunk, segment running: cancels queue, exits" gpt2fwe_chunk 999 "222" "999" no  "--me --name=gpt2fwe_chunk --state=PENDING;"
run_case "chunk, other chunk running: exits"            gpt2fwe_chunk 999 ""    "333 999" no ""
WANT_SUBMIT="1 h100;" run_case "chunk, only itself running: trains, queues next" gpt2fwe_chunk 999 ""    "999" yes ""
run_case "smoke test, chunk running: untouched"         gpt2fwe_smoke 999 ""    "111" yes ""
run_case "no Slurm job: untouched"                      ""            ""  ""    "111" yes ""
WANT_SUBMIT="1 h100;" run_case "chunk, previous chunk completing: waits, trains" gpt2fwe_chunk 999 ""    "999" yes "" "" 444
run_case "segment, chunk completing: waits, trains"         gpt2fwe_train 999 "999" ""    yes "" "" 444
run_case "chunk, segment completing: cancels queue, exits"  gpt2fwe_chunk 999 ""    "999" no  "--me --name=gpt2fwe_chunk --state=PENDING;" 222 ""
PENDING_CHUNKS=555 run_case "chunk, next already queued: trains, queues none" gpt2fwe_chunk 999 "" "999" yes ""
PENDING_CHUNKS=555 run_case "chunk, segment running, chunk queued: no successor" gpt2fwe_chunk 999 "222" "999" no "--me --name=gpt2fwe_chunk --state=PENDING;"
CHUNK_GPU=h200 WANT_SUBMIT="1 h200;" run_case "chunk on h200: successor on h200" gpt2fwe_chunk 999 "" "999" yes ""
DONE_STEP=00056352 run_case "chunk, run finished: exits, queues none" gpt2fwe_chunk 999 "" "999" no ""
DONE_STEP=00051000 WANT_SUBMIT="1 h100;" run_case "chunk, mid-run checkpoint only: trains, queues next" gpt2fwe_chunk 999 "" "999" yes ""
DONE_STEP=00056352 run_case "segment, run finished: continues to train.sbatch's own check" gpt2fwe_train 999 "999" "" yes ""
GUARD_SUCCESSOR_DELAY=5 KILL_EARLY=1 run_case "chunk dies before the delay: queues none" gpt2fwe_chunk 999 "" "999" yes ""
GUARD_SUCCESSOR_DELAY=1 WANT_SUBMIT="1 h100;" run_case "chunk outlives the delay: queues next" gpt2fwe_chunk 999 "" "999" yes ""

GUARD_WAIT_SECONDS=0 STUCK_CHUNKS=777 run_case "segment, chunk stuck completing: trains anyway" gpt2fwe_train 999 "999" "" yes ""
GUARD_WAIT_SECONDS=0 STUCK_CHUNKS=777 WANT_SUBMIT="1 h100;" run_case "chunk, old chunk stuck completing: trains anyway" gpt2fwe_chunk 999 "" "999" yes ""

export GUARD_SQUEUE_TRIES=1 GUARD_RETRY_SLEEP=0
FAKE_SQUEUE_FAIL=1 run_case "chunk, Slurm unreachable: exits, no training" gpt2fwe_chunk 999 "" "999" no ""
FAKE_SQUEUE_FAIL=1 run_case "segment, Slurm unreachable: cancels chunks by name, trains" gpt2fwe_train 999 "999" "" yes "--me --name=gpt2fwe_chunk;"

rm -rf "$T"
echo "$fails failed"
exit $((fails > 0))
