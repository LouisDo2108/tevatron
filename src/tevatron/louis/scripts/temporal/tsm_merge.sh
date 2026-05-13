# Average 3 checkpoints
python /home/thuy0050/code/TMRL/src/tevatron/louis/scripts/temporal/tsm_merge.py \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/after/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/before/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/early/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/late/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/range_from_to/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/single_year_in/model.safetensors \
    /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/range_between_and/model.safetensors \
    -o /home/thuy0050/mg61_scratch2/thuy0050/exp/tevatron/temporal_nobel_prize/tsm/facebook/contriever/merged/model.safetensors