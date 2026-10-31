output "api_url" {
  description = "Public API URL: put it in frontend/site/js/config.js and in the GitHub variable FINDXRAY_API_URL."
  value       = aws_lambda_function_url.api.function_url
}

output "deploy_identity" {
  description = "github_auth = user: create its key with `aws iam create-access-key --user-name <this>`. oidc: put this role ARN in the GitHub variable AWS_DEPLOY_ROLE_ARN."
  value       = var.github_auth == "user" ? aws_iam_user.github_deploy[0].name : aws_iam_role.github_deploy[0].arn
}

output "function_name" {
  value = aws_lambda_function.api.function_name
}

output "cap_table" {
  value = aws_dynamodb_table.cap.name
}
