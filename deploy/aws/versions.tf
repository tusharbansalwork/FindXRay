terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source = "hashicorp/aws"
      # 6.28 added invoked_via_function_url to aws_lambda_permission (needed for public function URLs)
      version = ">= 6.28, < 7.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.4, < 3.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "findxray", ManagedBy = "terraform" }
  }
}
