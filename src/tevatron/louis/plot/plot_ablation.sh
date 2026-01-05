source ~/.bashrc
mamba activate tevatron

ROOT_DIR=/home/thuy0050/code/tevatron/src/tevatron/louis

cd $ROOT_DIR

BACKBONE="contriever"
METRIC="recall100"

python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_ablation.py \
    --input_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.csv \
    --output_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.pdf \
    --backbone $BACKBONE \
    --metric $METRIC

BACKBONE="contriever"
METRIC="ndcg"
python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_ablation.py \
    --input_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.csv \
    --output_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.pdf \
    --backbone $BACKBONE \
    --metric $METRIC

BACKBONE="bge"
METRIC="recall100"

python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_ablation.py \
    --input_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.csv \
    --output_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.pdf \
    --backbone $BACKBONE \
    --metric $METRIC

BACKBONE="bge"
METRIC="ndcg"
python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_ablation.py \
    --input_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.csv \
    --output_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.pdf \
    --backbone $BACKBONE \
    --metric $METRIC

# BACKBONE="bge"
# METRIC="nq"
# python /home/thuy0050/code/tevatron/src/tevatron/louis/plot/plot_ablation.py \
#     --input_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.csv \
#     --output_path $ROOT_DIR/plot/ablation/$BACKBONE\_ablation\_$METRIC.pdf \
#     --backbone $BACKBONE \
#     --metric $METRIC