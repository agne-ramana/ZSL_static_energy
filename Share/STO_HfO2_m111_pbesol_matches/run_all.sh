#!/bin/bash
set -o pipefail
cd "$(dirname "$0")"

max_new=${1:-999999}
LEDGER=.submitted.tags
touch "$LEDGER"

LOCAL=()
while IFS= read -r line; do
    [ -n "$line" ] && LOCAL+=("$line")
done < "$LEDGER"

ACTIVE=()
while IFS= read -r line; do
    [ -n "$line" ] && ACTIVE+=("$line")
done < <(squeue -u "$USER" -h -o %j 2>/dev/null)

since=$(date -d '-14 days' +%Y-%m-%d 2>/dev/null || date -v-14d +%Y-%m-%d 2>/dev/null)
HIST=()
while IFS= read -r line; do
    [ -n "$line" ] && HIST+=("$line")
done < <(sacct -u "$USER" -n -X --format=JobName%64 --starttime="$since" 2>/dev/null)

is_known() {
    local tag="$1" x
    for x in "${LOCAL[@]}"; do [ "$x" = "$tag" ] && return 0; done
    for x in "${ACTIVE[@]}"; do [ "$x" = "v4_$tag" ] || [ "$x" = "$tag" ] && return 0; done
    for x in "${HIST[@]}"; do [ "$x" = "v4_$tag" ] || [ "$x" = "$tag" ] && return 0; done
    return 1
}

submitted=0
skipped=0
for d in configs/*/ ; do
    tag=$(basename "$d")
    if is_known "$tag"; then
        skipped=$((skipped + 1))
        continue
    fi
    if (( submitted >= max_new )); then
        echo "reached --max of $max_new new submissions this run, stopping (re-run to continue)"
        break
    fi
    printf 'submitting %-40s ... ' "$tag"
    if out=$(cd "$d" && sbatch submit.sh 2>&1); then
        echo "$out"
        echo "$tag" >> "$LEDGER"
        submitted=$((submitted + 1))
    else
        echo "FAILED: $out"
        echo "stopping here (this is what a QOS/submit-limit error looks like) -- re-run ./run_all.sh later to submit the rest"
        break
    fi
done
echo
echo "this run: submitted $submitted, skipped $skipped (ledger + scheduler)"
