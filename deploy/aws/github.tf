# How GitHub Actions gets permission to deploy new code. Either way it can do exactly one thing:
# update the code of the API function (and read its status).

locals {
  deploy_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["lambda:UpdateFunctionCode", "lambda:GetFunction", "lambda:GetFunctionConfiguration"]
      Resource = aws_lambda_function.api.arn
    }]
  })
}

# ---------- "user": IAM user with a scoped access key (works on the AWS Free plan) ----------
# The access key is created with the AWS CLI, not Terraform, so the secret never lands in the Terraform state:
#   aws iam create-access-key --user-name findxray-github-deploy
resource "aws_iam_user" "github_deploy" {
  count = var.github_auth == "user" ? 1 : 0
  name  = "${var.name}-github-deploy"
}

resource "aws_iam_user_policy" "github_deploy" {
  count  = var.github_auth == "user" ? 1 : 0
  name   = "${var.name}-deploy-code"
  user   = aws_iam_user.github_deploy[0].name
  policy = local.deploy_policy
}

# ---------- "oidc": short-lived credentials, no stored keys (needs advanced features) ----------
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.github_auth == "oidc" ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

data "aws_iam_policy_document" "github_trust" {
  count = var.github_auth == "oidc" ? 1 : 0
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github[0].arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  count                = var.github_auth == "oidc" ? 1 : 0
  name                 = "${var.name}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_trust[0].json
  max_session_duration = 3600
}

resource "aws_iam_role_policy" "github_deploy" {
  count  = var.github_auth == "oidc" ? 1 : 0
  name   = "${var.name}-deploy-code"
  role   = aws_iam_role.github_deploy[0].id
  policy = local.deploy_policy
}
