"""Validate the configured OSWorld task and AWS launch without starting an instance."""

import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from PIL import Image

# This script is launched with the selected OSWorld checkout as its working directory.
sys.path.insert(0, str(Path.cwd()))
load_dotenv(Path.cwd()/".env")


def main():
    from desktop_env.controllers.python import PythonController
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.aws import manager
    from scripts.python import run_multienv_qwen as runner

    domain, task_id, width, height = sys.argv[1:]
    region = os.environ["AWS_REGION"]
    subnet = os.environ["AWS_SUBNET_ID"]
    security_group = os.environ["AWS_SECURITY_GROUP_ID"]
    assets = Path(os.environ["OSWORLD_FILE_BASE_URL"]).expanduser()
    if not assets.is_dir() or not any(entry.is_dir() for entry in assets.iterdir()):
        raise ValueError("OSWORLD_FILE_BASE_URL must point at the local task assets")

    flags = runner.config([
        "--provider_name", "aws", "--region", region, "--headless",
        "--screen_width", width, "--screen_height", height, "--num_envs", "1",
        "--max_steps", "1", "--no-enable_recording", "--no-enable_vnc",
    ])
    task = runner._load_example(domain, task_id, flags)
    if not callable(getattr(task, "setup", None)) or not callable(getattr(task, "evaluate", None)):
        raise ValueError(f"Task {task_id} has no callable setup or evaluator")
    spec = runner._env_spec(task, flags)
    ami = spec["snapshot_name"]
    configured_ami = os.environ.get("OSWORLD_AMI_ID")
    if configured_ami and ami != configured_ami:
        raise ValueError("The runner selected an AMI different from OSWORLD_AMI_ID")

    ec2 = boto3.client("ec2", region_name=region, config=Config(
        connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1},
    ))
    image = ec2.describe_images(ImageIds=[ami])["Images"][0]
    if image["State"] != "available":
        raise ValueError(f"The configured AMI is {image['State']}")
    subnet_vpc = ec2.describe_subnets(SubnetIds=[subnet])["Subnets"][0]["VpcId"]
    group_vpc = ec2.describe_security_groups(GroupIds=[security_group])["SecurityGroups"][0]["VpcId"]
    if subnet_vpc != group_vpc:
        raise ValueError("The subnet and security group are in different VPCs")

    # Expose only read methods and RunInstances forced to DryRun. The provider
    # builds the real request but receives no instance ID to wait on or terminate.
    dry_client = SimpleNamespace(
        describe_images=ec2.describe_images,
        describe_snapshots=ec2.describe_snapshots,
        run_instances=lambda **kwargs: ec2.run_instances(DryRun=True, **kwargs),
    )
    with patch.object(manager, "boto3", SimpleNamespace(client=lambda *a, **k: dry_client)), \
         patch.object(manager.logger, "disabled", True):
        try:
            manager._allocate_vm(
                region=region, ami_id=ami, screen_size=(int(width), int(height)),
                instance_type=spec["instance_type"], volume_size=spec["volume_size"],
                os_type=spec["os_type"],
            )
        except ClientError as error:
            if error.response["Error"]["Code"] != "DryRunOperation":
                raise
        else:
            raise RuntimeError("Expected AWS to return DryRunOperation")

    instances = ec2.describe_instances(Filters=[
        {"Name": "image-id", "Values": [ami]},
        {"Name": "subnet-id", "Values": [subnet]},
        {"Name": "instance.group-id", "Values": [security_group]},
        {"Name": "instance-state-name", "Values": ["running"]},
    ])
    running = [instance for r in instances["Reservations"] for instance in r["Instances"]
               if instance.get("PrivateIpAddress")]
    report = {
        "task": f"{domain}/{task_id}", "task_setup_and_evaluator_loaded": True,
        "region": region, "ami_state": image["State"], "subnet_and_security_group_match": True,
        "launch_permission": "DryRunOperation", "running_matching_desktops": len(running),
        "screenshot": "skipped",
    }
    # A screenshot GET exercises the same observation path without constructing
    # DesktopEnv, whose AWS lifecycle methods can start or terminate an instance.
    if len(running) == 1:
        controller = PythonController(running[0]["PrivateIpAddress"], server_port=5000)
        controller.retry_times = 1
        controller.retry_interval = 0
        observation = DesktopEnv._get_obs(SimpleNamespace(
            controller=controller, require_a11y_tree=False, require_terminal=False, instruction=None,
        ))
        screenshot = observation["screenshot"]
        if screenshot:
            with Image.open(io.BytesIO(screenshot)) as data:
                data.verify()
                report["screenshot"] = {"width": data.width, "height": data.height}
        else:
            report["screenshot"] = "desktop did not return an image"
    print(json.dumps(report, indent=2))
    return int(len(running) == 1 and not isinstance(report["screenshot"], dict))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ClientError as error:
        sys.exit(f"AWS check failed: {error.response['Error']['Code']}")
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
        sys.exit(str(error))
