# FindXRay API on AWS Lambda: the same FastAPI app as on Render, run by the AWS Lambda Web Adapter,
# behind a public Lambda function URL with response streaming (for /report/stream).

data "aws_caller_identity" "me" {}
data "aws_partition" "current" {}

locals {
  function_name = "${var.name}-api"
  ssm_prefix    = "/${var.name}/"
  ssm_arn       = "arn:${data.aws_partition.current.partition}:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter/${var.name}"
  adapter_layer = "arn:${data.aws_partition.current.partition}:lambda:${var.region}:753240598075:layer:LambdaAdapterLayerArm64:${var.web_adapter_layer_version}"
}

# ---------- daily report cap, shared by all instances ----------
# Provisioned 2/2 capacity stays inside DynamoDB's always-free 25 RCU / 25 WCU.
resource "aws_dynamodb_table" "cap" {
  name           = "${var.name}-report-cap"
  billing_mode   = "PROVISIONED"
  read_capacity  = 2
  write_capacity = 2
  hash_key       = "day"

  attribute {
    name = "day"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# ---------- IAM role for the function (least privilege) ----------
data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = "${local.function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "ReportCap"
    actions   = ["dynamodb:UpdateItem", "dynamodb:GetItem"]
    resources = [aws_dynamodb_table.cap.arn]
  }
  statement {
    sid       = "ReadSecrets"
    actions   = ["ssm:GetParametersByPath"]
    resources = [local.ssm_arn, "${local.ssm_arn}/*"]
  }
  statement {
    sid       = "DecryptSecretsViaSsm"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "api" {
  name   = "${local.function_name}-access"
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

# ---------- the function ----------
resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "api" {
  function_name    = local.function_name
  description      = "FindXRay similar-case search and report agent (FastAPI via Lambda Web Adapter)"
  role             = aws_iam_role.api.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "run.sh"
  filename         = var.lambda_zip
  source_code_hash = filebase64sha256(var.lambda_zip)
  memory_size      = var.memory_mb
  timeout          = 120 # a full report takes 15-30 s; Gemini retries can stretch it
  layers           = [local.adapter_layer]

  environment {
    variables = {
      AWS_LAMBDA_EXEC_WRAPPER      = "/opt/bootstrap"
      AWS_LWA_INVOKE_MODE          = "response_stream"
      AWS_LWA_READINESS_CHECK_PATH = "/health"
      PORT                         = "8000"
      CASEFINDER_LIBRARY           = "/var/task/library"
      CASEFINDER_ORIGINS           = join(",", var.allowed_origins)
      CASEFINDER_DAILY_REPORTS     = tostring(var.daily_reports)
      CASEFINDER_CAP_TABLE         = aws_dynamodb_table.cap.name
      FINDXRAY_SSM_PREFIX          = local.ssm_prefix
      GEMINI_MODEL                 = var.gemini_model
      GEMINI_RPM                   = "10"
      LANGFUSE_BASE_URL            = "https://cloud.langfuse.com"
    }
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy_attachment.api_logs]

  lifecycle {
    # After the first deploy, GitHub Actions ships new code; Terraform only manages configuration.
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE" # a public website API; abuse is bounded by the report cap, concurrency and the budget kill switch
  invoke_mode        = "RESPONSE_STREAM"
  # No CORS block here: FastAPI answers CORS itself (CASEFINDER_ORIGINS); setting both duplicates the headers.
}

# Public function URLs need both permissions (AWS, 2025); the second is limited to calls made through the URL.
resource "aws_lambda_permission" "url_public" {
  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "url_invoke" {
  statement_id             = "AllowInvokeViaFunctionUrl"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.api.function_name
  principal                = "*"
  invoked_via_function_url = true
}
