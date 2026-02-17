# ===============================================
# VPC MODULE VARIABLES
# ===============================================

variable "cloud_provider" {
  description = "Cloud provider to deploy to (aws, gcp, azure)"
  type        = string
  validation {
    condition     = contains(["aws", "gcp", "azure"], var.cloud_provider)
    error_message = "Cloud provider must be one of: aws, gcp, azure."
  }
}

variable "project_name" {
  description = "Name of the trading bot project"
  type        = string
  default     = "trading-bot"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "TradingBot"
    Environment = "production"
    Managed-By  = "Terraform"
  }
}

# ===============================================
# AWS-SPECIFIC VARIABLES
# ===============================================
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.21.0/24", "10.0.31.0/24"]
}

# ===============================================
# GCP-SPECIFIC VARIABLES
# ===============================================
variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "gcp_public_subnet_cidr" {
  description = "CIDR block for GCP public subnet"
  type        = string
  default     = "10.1.1.0/24"
}

variable "gcp_private_subnet_cidr" {
  description = "CIDR block for GCP private subnet"
  type        = string
  default     = "10.1.10.0/24"
}

variable "gcp_pods_cidr" {
  description = "CIDR block for GKE pods"
  type        = string
  default     = "10.2.0.0/16"
}

variable "gcp_services_cidr" {
  description = "CIDR block for GKE services"
  type        = string
  default     = "10.3.0.0/16"
}

# ===============================================
# AZURE-SPECIFIC VARIABLES
# ===============================================
variable "azure_location" {
  description = "Azure region"
  type        = string
  default     = "East US"
}

variable "azure_vnet_cidr" {
  description = "CIDR block for Azure virtual network"
  type        = string
  default     = "10.4.0.0/16"
}

variable "azure_public_subnet_cidr" {
  description = "CIDR block for Azure public subnet"
  type        = string
  default     = "10.4.1.0/24"
}

variable "azure_private_subnet_cidr" {
  description = "CIDR block for Azure private subnet"
  type        = string
  default     = "10.4.10.0/24"
}

variable "azure_database_subnet_cidr" {
  description = "CIDR block for Azure database subnet"
  type        = string
  default     = "10.4.20.0/24"
}