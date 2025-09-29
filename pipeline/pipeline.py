import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
import yaml, os

CONFIG_PATH = os.path.join("config", "pipeline_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

role = config["role_arn"]
sess = sagemaker.Session()

processor = ScriptProcessor(
    image_uri=config["image_uri"],
    command=["python3"],
    instance_type=config["instance_type"],
    instance_count=1,
    role=role,
)

step_preprocess = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    code="scripts/preprocess.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{config['bucket_name']}/{config['train_path']}",
            destination="/opt/ml/processing/input/train"
        ),
        ProcessingInput(
            source=f"s3://{config['bucket_name']}/{config['test_path']}",
            destination="/opt/ml/processing/input/test"
        ),
        ProcessingInput(
            source=f"s3://{config['bucket_name']}/{config['valid_path']}",
            destination="/opt/ml/processing/input/valid"
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{config['bucket_name']}/{config['train_save_path']}"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
            destination=f"s3://{config['bucket_name']}/{config['test_save_path']}"
        ),
        ProcessingOutput(
            output_name="valid",
            source="/opt/ml/processing/output/valid",
            destination=f"s3://{config['bucket_name']}/{config['valid_save_path']}"
        ),
    ],
)

pipeline = Pipeline(
    name="NERPreprocessPipeline",
    steps=[step_preprocess],
    sagemaker_session=sess,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
