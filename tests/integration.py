import boto3
import sys
import os

if 1 == 1:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from pipelines.InstaDeepMHCIPresentation.pipeline import get_pipeline

region = boto3.Session().region_name
role = "arn:aws:iam::975628797022:role/service-role/AmazonSageMaker-ExecutionRole-20250826T094769"

model_package_group_name = "InstaDeepMHCIPresentationModelPackageGroup"
pipeline_name = "InstaDeepMHCIPresentationPipeline"

pipeline = get_pipeline(
    region=region,
    role=role,
    model_package_group_name=model_package_group_name,
    pipeline_name=pipeline_name,
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.wait(delay=60, max_attempts=200)
status = execution.describe()
print(status)
if status["PipelineExecutionStatus"] == "Succeeded":
    print("Success")
    sys.exit(0)
else:
    print("Failure!")
    sys.exit(5)
