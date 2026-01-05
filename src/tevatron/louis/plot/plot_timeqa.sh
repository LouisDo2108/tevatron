source ~/.bashrc
mamba activate tevatron

ROOT_DIR=/home/thuy0050/code/tevatron/src/tevatron/louis

cd $ROOT_DIR

BACKBONE="contriever"
METRIC="ndcg"
python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
    --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
    --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
    --backbone $BACKBONE

BACKBONE="contriever"
METRIC="recall100"
python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
    --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
    --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
    --metric Recall@100 \
    --backbone $BACKBONE

# BACKBONE="bge"
# METRIC="ndcg"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --backbone $BACKBONE

# BACKBONE="bge"
# METRIC="recall100"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --metric Recall@100 \
#     --backbone $BACKBONE

# BACKBONE="gte"
# METRIC="ndcg"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --backbone $BACKBONE

# BACKBONE="gte"
# METRIC="recall100"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --metric Recall@100 \
#     --backbone $BACKBONE

# BACKBONE="gte1.5"
# METRIC="ndcg"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --backbone $BACKBONE

# BACKBONE="gte1.5"
# METRIC="recall100"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --metric Recall@100 \
#     --backbone $BACKBONE

# BACKBONE="nomic"
# METRIC="ndcg"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --backbone $BACKBONE

# BACKBONE="nomic"
# METRIC="recall100"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --metric Recall@100 \
#     --backbone $BACKBONE

# BACKBONE="bgem3"
# METRIC="ndcg"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --backbone $BACKBONE

# BACKBONE="bgem3"
# METRIC="recall100"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_main.py \
#     --input_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.csv \
#     --output_path $ROOT_DIR/plot/timeqa/$BACKBONE/$METRIC.pdf \
#     --metric Recall@100 \
#     --backbone $BACKBONE
