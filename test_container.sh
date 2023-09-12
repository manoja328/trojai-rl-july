python metafile_generator.py

export APPTAINER_TMPDIR=/workspace/manoj/tmp/
sudo -E apptainer build detector.simg detector.def

DIR=`pwd`

singularity run \
--bind $DIR \
--nv ./detector.simg infer \
--model_filepath=./rl-july-example/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch \
--examples_dirpath=./rl-july-example/clean-example-data/ \
--round_training_dataset_dirpath=/workspace/manoj/trojai-datasets/nlp-question-answering-aug2023 \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/


~/gdrive files upload detector.simg
echo "=== === === === ==="
echo "upload to google drive ...."
echo "rename as:  nlp-question-answering-aug2023_sts_SRIweightv0.simg"
echo "=== don't forget to share it with trojai@gmail ==="