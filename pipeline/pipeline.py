import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
import yaml, os

CONFIG_PATH = os.path.join("config", "pipeline_config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

role = config["role_arn"]
sess = sagemaker.Session()

# Preprocessing Processor 
processor = ScriptProcessor(
    image_uri=config["image_uri"],
    command=["python3"],
    instance_type=config["instance_type"],
    instance_count=1,
    role=role,
)

# Preprocessing Step
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

# Training Estimator
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="scripts",
    instance_type=config["instance_type_train"],
    instance_count=1,
    role=role,
    transformers_version="4.49.0",
    pytorch_version="2.5.1",
    py_version="py311",
)

# Training Step
step_train = TrainingStep(
    name="TrainModel",
    estimator=huggingface_estimator,  
    inputs={
        "train": TrainingInput(  
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        ),
        "valid": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["valid"].S3Output.S3Uri,
        ),
        "test" : TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
        )
    },
)

# Pipeline
pipeline = Pipeline(
    name="NERPreprocessAndTrainPipeline",
    steps=[step_preprocess, step_train],
    sagemaker_session=sess,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f"Pipeline started: {execution.arn}")

#  MODEL REGISTRATION STEP 