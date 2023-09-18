# python entrypoint.py infer \
# --model_filepath=./rl-july-example/model.pt \
# --result_filepath=./output.txt \
# --scratch_dirpath=./scratch \
# --examples_dirpath=./rl-july-example/clean-example-data/ \
# --round_training_dataset_dirpath=/workspace/manoj/trojai-datasets/nlp-question-answering-aug2023 \
# --metaparameters_filepath=./metaparameters.json \
# --schema_filepath=./metaparameters_schema.json \
# --learned_parameters_dirpath=./learned_parameters/

DIR="/workspace/manoj/trojai-datasets/rl-lavaworld-jul2023/models"

for model_filepath in `ls $DIR`; do
    echo $model_filepath
    python entrypoint.py infer \
    --model_filepath=$DIR/$model_filepath/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch \
    --examples_dirpath=$DIR/$model_filepath/clean-example-data/ \
    --round_training_dataset_dirpath=$DIR \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./learned_parameters/
done