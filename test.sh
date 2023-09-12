python entrypoint.py infer \
--model_filepath=./rl-july-example/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch \
--examples_dirpath=./rl-july-example/clean-example-data/ \
--round_training_dataset_dirpath=/workspace/manoj/trojai-datasets/nlp-question-answering-aug2023 \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/

