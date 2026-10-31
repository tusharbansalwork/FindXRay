# Alarms, a $1 budget with email, and a kill switch that turns the API off if the budget is exceeded.

resource "aws_sns_topic" "alerts" {
  name = "${var.name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

locals {
  alarms = {
    # name            = [metric,          statistic, threshold, description]
    server-errors = ["Url5xxCount", "Sum", 5, "5 or more 5xx responses in 5 minutes"]
    function-errors = ["Errors", "Sum", 5, "5 or more failed invocations in 5 minutes"]
    throttles       = ["Throttles", "Sum", 1, "Requests are being throttled (concurrency limit or the budget kill switch)"]
  }
}

resource "aws_cloudwatch_metric_alarm" "api" {
  for_each            = local.alarms
  alarm_name          = "${local.function_name}-${each.key}"
  alarm_description   = each.value[3]
  namespace           = "AWS/Lambda"
  metric_name         = each.value[0]
  statistic           = each.value[1]
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  period              = 300
  evaluation_periods  = 1
  threshold           = each.value[2]
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# Requests close to the 120 s timeout mean Gemini calls are hanging.
resource "aws_cloudwatch_metric_alarm" "slow" {
  alarm_name          = "${local.function_name}-near-timeout"
  alarm_description   = "p99 duration above 90 s over 15 minutes"
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  extended_statistic  = "p99"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  period              = 900
  evaluation_periods  = 1
  threshold           = 90000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ---------- budget + kill switch ----------
resource "aws_sns_topic" "budget" {
  name = "${var.name}-budget"
}

data "aws_iam_policy_document" "budget_topic" {
  statement {
    actions   = ["SNS:Publish"]
    resources = [aws_sns_topic.budget.arn]
    principals {
      type        = "Service"
      identifiers = ["budgets.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.me.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "budget" {
  arn    = aws_sns_topic.budget.arn
  policy = data.aws_iam_policy_document.budget_topic.json
}

resource "aws_budgets_budget" "monthly" {
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification { # early warning
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
  }

  notification { # over budget: email + kill switch
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.alert_email]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget.arn]
  }

  notification { # forecast warning
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }

  depends_on = [aws_sns_topic_policy.budget]
}

data "archive_file" "killswitch" {
  type        = "zip"
  source_file = "${path.module}/killswitch/handler.py"
  output_path = "${path.module}/.build/killswitch.zip"
}

resource "aws_iam_role" "killswitch" {
  name               = "${var.name}-killswitch-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
}

resource "aws_iam_role_policy_attachment" "killswitch_logs" {
  role       = aws_iam_role.killswitch.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "killswitch" {
  name = "${var.name}-killswitch-access"
  role = aws_iam_role.killswitch.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:PutFunctionConcurrency"]
      Resource = aws_lambda_function.api.arn
    }]
  })
}

resource "aws_cloudwatch_log_group" "killswitch" {
  name              = "/aws/lambda/${var.name}-killswitch"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "killswitch" {
  function_name    = "${var.name}-killswitch"
  description      = "Turns the FindXRay API off (reserved concurrency 0) when the monthly budget is exceeded"
  role             = aws_iam_role.killswitch.arn
  runtime          = "python3.12"
  architectures    = ["arm64"]
  handler          = "handler.handler"
  filename         = data.archive_file.killswitch.output_path
  source_code_hash = data.archive_file.killswitch.output_base64sha256
  timeout          = 30
  environment {
    variables = { TARGET_FUNCTION = aws_lambda_function.api.function_name }
  }
  depends_on = [aws_cloudwatch_log_group.killswitch]
}

resource "aws_lambda_permission" "killswitch_sns" {
  statement_id  = "AllowBudgetTopic"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.killswitch.function_name
  principal     = "sns.amazonaws.com"
  source_arn    = aws_sns_topic.budget.arn
}

resource "aws_sns_topic_subscription" "killswitch" {
  topic_arn = aws_sns_topic.budget.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.killswitch.arn
}
