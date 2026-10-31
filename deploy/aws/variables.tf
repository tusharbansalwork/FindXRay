variable "region" {
  description = "AWS region. Free-plan projects are pinned to one Region by country; India -> Asia Pacific (Sydney)."
  type        = string
  default     = "ap-southeast-2"
}

variable "name" {
  description = "Prefix for every resource."
  type        = string
  default     = "findxray"
}

variable "alert_email" {
  description = "Email for budget and alarm notifications (confirm the AWS subscription email once)."
  type        = string
}

variable "github_repo" {
  description = "owner/repo allowed to deploy new code through GitHub Actions (OIDC, main branch only)."
  type        = string
  default     = "tusharbansalwork/FindXRay"
}

variable "github_auth" {
  description = <<-EOT
    How GitHub Actions authenticates to deploy:
      "user" - an IAM user limited to updating this one function; its access key goes into GitHub secrets.
               Needed on the AWS Free plan, whose service control policy denies creating OIDC providers.
      "oidc" - short-lived credentials via GitHub's OIDC provider (no stored keys). Preferred once the
               account has advanced features activated.
  EOT
  type        = string
  default     = "user"
  validation {
    condition     = contains(["user", "oidc"], var.github_auth)
    error_message = "github_auth must be \"user\" or \"oidc\"."
  }
}

variable "allowed_origins" {
  description = "Browser origins allowed to call the API (CORS)."
  type        = list(string)
  default     = ["https://findxray.onrender.com", "http://localhost:8080", "http://127.0.0.1:8080"]
}

variable "lambda_zip" {
  description = "Package built by scripts/build_lambda.py. Only used when the function is first created; later code updates go through CI."
  type        = string
  default     = "../../build/findxray-lambda.zip"
}

variable "memory_mb" {
  description = "Lambda memory; CPU scales with it. 1024 MB keeps cold starts short and stays far inside the free tier."
  type        = number
  default     = 1024
}

variable "daily_reports" {
  description = "Live AI reports allowed per Pacific-time day (0 = no cap)."
  type        = number
  default     = 60
}

variable "gemini_model" {
  type    = string
  default = "gemini-3.5-flash-lite"
}

variable "budget_usd" {
  description = "Monthly budget. Above it: email, and the kill switch turns the API off."
  type        = number
  default     = 1
}

variable "web_adapter_layer_version" {
  description = "Version of AWS's public Lambda Web Adapter layer (arm64), see github.com/awslabs/aws-lambda-web-adapter."
  type        = number
  default     = 22
}

variable "log_retention_days" {
  type    = number
  default = 14
}
