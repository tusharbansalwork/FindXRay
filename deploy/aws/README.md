# FindXRay API on AWS

The API (search + report agent) runs on **AWS Lambda** (arm64, Python 3.12), started by the
[AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) so the FastAPI app runs unchanged.
The website stays on Render (static, free) and calls the Lambda function URL.

```
browser ── findxray.onrender.com (static page, encoder runs in the browser)
   │  embedding / X-ray
   ▼
Lambda function URL (public, response streaming) ──► Lambda findxray-api (FastAPI)
                                                        ├─ case library bundled in the package
                                                        ├─ DynamoDB  findxray-report-cap   (daily report cap, shared)
                                                        ├─ SSM Parameter Store /findxray/* (Gemini + Langfuse keys)
                                                        └─ CloudWatch logs + alarms ─► SNS email
AWS Budgets ($1/month) ─► SNS ─► findxray-killswitch Lambda (sets the API's concurrency to 0)
GitHub Actions (deploy-only identity, main branch) ─► test ─► build ─► update code ─► smoke test
```

Everything is Terraform in this folder. Secrets are not in Terraform: they are put into SSM once from `.env`.

## Cost: designed to stay at $0

| Service | Always-free monthly allowance | Expected use |
|---|---|---|
| Lambda | 1M requests, 400,000 GB-seconds | thousands of requests, a few hundred GB-s |
| DynamoDB (provisioned 2 RCU / 2 WCU) | 25 RCU, 25 WCU, 25 GB | one item per day |
| CloudWatch | 10 alarms, 5 GB logs | 4 alarms, MBs of logs (14-day retention) |
| SSM standard parameters, SNS email, Budgets | free | 3 parameters, 1 budget |

The package is uploaded directly to Lambda (no S3, no container registry), so there is no storage charge.
Guards against a surprise bill: the 60-a-day report cap, the account's Lambda concurrency limit, the $1 budget
alert, and the kill switch that turns the API off if the budget is exceeded
(`aws lambda delete-function-concurrency --function-name findxray-api` turns it back on).

## AWS Free plan (new sign-up "projects") — what changes
Accounts created through the new sign-up start on the Free plan inside a project. Checked against AWS's
published project policies:
- **One Region**, assigned by country: India -> Asia Pacific (Sydney), `ap-southeast-2` (the default here).
  Confirm yours in AWS Settings -> Project -> Additional info.
- Lambda, function URLs, DynamoDB, IAM roles, CloudWatch, SNS, Systems Manager and Budgets are all allowed.
- **OIDC identity providers are denied** (`iam:*Provider*`), so GitHub Actions deploys with a deploy-only IAM
  user's access key (`github_auth = "user"`, the default). With advanced features, switch to `"oidc"`.
- No IAM users are needed for you: `aws login` signs the CLI in with your console session (temporary credentials).
- The Free plan ends after 6 months (or when the $100 credit is used up); upgrade then, or point the site back
  at the Render API (`frontend/site/js/config.js`).

## First deployment (Windows, PowerShell)

1. **Tools** (once): `winget install Amazon.AWSCLI` and `winget install Hashicorp.Terraform`, then open a new terminal.
2. **Sign the CLI in** (opens the browser; no long-lived keys). The second profile hands the same session to
   Terraform and boto3, which may not read `aws login` sessions directly yet:
   ```powershell
   aws login --profile findxray-login
   aws configure set region ap-southeast-2 --profile findxray-login
   aws configure set credential_process "aws configure export-credentials --profile findxray-login --format process" --profile findxray
   aws configure set region ap-southeast-2 --profile findxray
   $env:AWS_PROFILE = "findxray"
   aws sts get-caller-identity
   ```
   Sessions last up to 12 hours; run `aws login --profile findxray-login` again when they expire.
3. **Secrets into SSM** (reads `.env`; keys never appear in your terminal history):
   ```powershell
   pip install -e ".[load]"
   python scripts/aws_put_secrets.py
   ```
4. **Build the package** (Linux arm64 wheels are downloaded, so this works on Windows):
   ```powershell
   python scripts/build_lambda.py
   ```
5. **Create everything**:
   ```powershell
   cd deploy/aws
   copy terraform.tfvars.example terraform.tfvars    # then put your email in it
   terraform init
   terraform plan      # read it: about 30 resources to add, 0 to change, 0 to destroy
   terraform apply
   cd ../..
   ```
   Note the outputs `api_url` and `deploy_identity`. Confirm the "AWS Notification - Subscription Confirmation" email.
6. **Smoke test** the live API: `python scripts/smoke_test.py --url <api_url>`
7. **CI/CD**: create the deploy user's key (it can only update this one function):
   `aws iam create-access-key --user-name findxray-github-deploy`. In GitHub → Settings → Secrets and variables →
   Actions: secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`; variables `FINDXRAY_API_URL` = `api_url`,
   `AWS_REGION` = `ap-southeast-2`. From then on every push to `main` that touches the API runs the tests,
   deploys and smoke-tests.
8. **Point the website at AWS**: set `AWS_API` in `frontend/site/js/config.js` to `api_url` (no trailing slash) and push.

## Evaluation (after deploying)

```powershell
# 1. Correctness: all 720 benchmark queries through the live API must reproduce notebook 01 exactly
python scripts/check_api_against_benchmark.py --url <api_url>

# 2. Load test: 50 simulated visitors for 3 minutes (search only; reports are capped by design)
locust -f loadtest/locustfile.py --headless --host <api_url> -u 50 -r 5 -t 3m --csv results/03_cloud/aws

# 3. Server-side view of the same window: cold starts, per-endpoint p50/p95/p99, memory, cost per 1,000
python scripts/lambda_metrics.py --minutes 15 --out results/03_cloud/aws_lambda_metrics.json

# 4. Same load test against the Render API, for comparison
locust -f loadtest/locustfile.py --headless --host https://findxray-api.onrender.com -u 50 -r 5 -t 3m --csv results/03_cloud/render
```

Results (2026-10-31, measured from India; raw files in `results/03_cloud/`):

| Check | Result |
|---|---|
| Correctness | 720/720 benchmark queries reproduce P@5 0.4172 and mAP@10 0.2843 exactly |
| Load, 50 visitors for 3 min | 3,541 requests, **0 errors**, 19.9 req/s, median 430 ms, p95 470 ms |
| Server time per search | p50 2.3 ms, p95 30 ms, p99 44 ms |
| Cold starts | 5 (0.14% of requests), 3.7 s p50: misses the 3 s target (agent stack imported at start-up) |
| Same load on Render free | median 6.2 s, 5.8 req/s (saturated): AWS ~14x faster |
| Cost | $0.00043 per 1,000 requests at list price, $0 in the free tier |

Seen from India, latency is almost all distance to Sydney (~0.41 s per round trip). CloudFront in front of the
function URL, or Mumbai once off the Free plan, would cut it.

Service-level objectives the results are checked against:

| SLO | Target |
|---|---|
| Search correctness | P@5 = 0.4172 and mAP@10 = 0.2843 through the API (exact match with the benchmark) |
| Search latency, server-side | p95 under 100 ms (warm) |
| Search latency, seen from India (Sydney region) | p95 under 500 ms |
| Errors under load | under 0.1% of requests |
| Cold start | under 3 s to first response |
| Full report | p95 under 40 s |
| Cost | $0 a month at expected traffic |
