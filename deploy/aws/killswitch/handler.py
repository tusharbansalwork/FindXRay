"""Budget kill switch: when the monthly AWS budget is exceeded, stop the API from running.

Triggered by the budget's SNS notification. Setting reserved concurrency to 0 makes every request to
the function URL fail fast (throttled) without deleting anything. To turn the API back on:
  aws lambda delete-function-concurrency --function-name findxray-api
"""

import json
import os

import boto3


def handler(event, context):
    target = os.environ["TARGET_FUNCTION"]
    boto3.client("lambda").put_function_concurrency(FunctionName=target, ReservedConcurrentExecutions=0)
    message = [r.get("Sns", {}).get("Subject") for r in event.get("Records", [])]
    print(json.dumps({"action": "throttled", "function": target, "trigger": message}))
    return {"throttled": target}
