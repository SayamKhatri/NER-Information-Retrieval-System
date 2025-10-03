import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.huggingface import HuggingFace
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
import yaml, os
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join



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

# Retrive metrics
# Retrieve metrics
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_train.properties.OutputDataConfig.S3OutputPath,
                step_train.properties.TrainingJobName,
                "output/data/evaluation_results.json"
            ]
        ),
        content_type="application/json"
    )
)


# Model registeration step
step_register = RegisterModel(
    name='NerModel',
    estimator=huggingface_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    image_uri=config["image_uri"],
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
    transform_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
    model_package_group_name="biobert-ner-models",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics,
)


# Pipeline
pipeline = Pipeline(
    name="NERPreprocessAndTrainPipeline",
    steps=[step_preprocess, step_train, step_register],
    sagemaker_session=sess,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f"Pipeline started: {execution.arn}")

