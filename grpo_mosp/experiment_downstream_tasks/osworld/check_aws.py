"""Check the AWS identity, pinned desktop AMI, and selected network without launching EC2."""

import argparse
import os

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region")
    parser.add_argument("ami")
    args = parser.parse_args()
    session = boto3.Session(region_name=args.region)
    config = Config(connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1})
    ec2 = session.client("ec2", config=config)
    failures = []

    def check(label, request):
        try:
            message = request()
        except (BotoCoreError, ClientError, IndexError) as error:
            if isinstance(error, ClientError):
                error = error.response["Error"]["Code"]
            elif isinstance(error, IndexError):
                error = "resource missing or not visible"
            failures.append(label)
            print(f"{label}: {error}", flush=True)
            return
        print(f"{label}: {message}", flush=True)
        return message

    check("AWS identity", lambda: session.client("sts", config=config).get_caller_identity()["Arn"])
    image_state = check(
        f"Benchmark AMI {args.ami} in {args.region}",
        lambda: ec2.describe_images(ImageIds=[args.ami])["Images"][0]["State"],
    )
    if image_state and image_state != "available":
        failures.append("AMI availability")
    network = (
        ("AWS_SUBNET_ID", lambda identifier: ec2.describe_subnets(SubnetIds=[identifier])["Subnets"]),
        ("AWS_SECURITY_GROUP_ID", lambda identifier: ec2.describe_security_groups(
            GroupIds=[identifier])["SecurityGroups"]),
    )
    vpcs = []
    for key, describe in network:
        identifier = os.environ.get(key)
        if not identifier:
            failures.append(key)
            print(f"{key}: unset", flush=True)
            continue
        vpc = check(key, lambda: describe(identifier)[0]["VpcId"])
        if vpc:
            vpcs.append(vpc)
    if len(set(vpcs)) > 1:
        failures.append("VPC mismatch")
        print("The benchmark subnet and security group must share a VPC.", flush=True)
    if not failures:
        print("AWS inventory checks passed. Reachability and launch permissions still need a live test.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
