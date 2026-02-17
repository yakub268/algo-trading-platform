# ===============================================
# AWS TERRAFORM VARIABLES
# ===============================================

# Project Configuration
variable "project_name" {
  description = "Name of the trading bot project"
  type        = string
  default     = "trading-bot"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
  default     = ""
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.28"
}

variable "eks_node_instance_types" {
  description = "Instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "eks_node_desired_size" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "eks_node_min_size" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 2
}

variable "eks_node_max_size" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 20
}

variable "eks_node_disk_size" {
  description = "Disk size for EKS nodes (GB)"
  type        = number
  default     = 100
}

variable "eks_api_access_cidrs" {
  description = "CIDR blocks that can access EKS API"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r5.xlarge"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage (GB)"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS max allocated storage (GB)"
  type        = number
  default     = 1000
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "tradingbot"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "tradingbot"
}

variable "db_backup_retention_period" {
  description = "Database backup retention period (days)"
  type        = number
  default     = 30
}

variable "db_multi_az" {
  description = "Enable multi-AZ deployment for RDS"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.xlarge"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention period (days)"
  type        = number
  default     = 30
}

variable "alert_email_addresses" {
  description = "Email addresses for alerts"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all services"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all services"
  type        = bool
  default     = true
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

variable "backup_window" {
  description = "Preferred backup window (UTC)"
  type        = string
  default     = "03:00-04:00"
}

# Auto Scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "cluster_autoscaler_scale_down_delay" {
  description = "Time to wait before scaling down (minutes)"
  type        = number
  default     = 10
}

variable "cluster_autoscaler_scale_down_unneeded" {
  description = "Time a node should be unneeded before eligible for scale down (minutes)"
  type        = number
  default     = 10
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_types" {
  description = "Instance types for spot instances"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge", "m4.large", "m4.xlarge"]
}

variable "spot_max_price" {
  description = "Maximum price for spot instances"
  type        = string
  default     = ""
}

# Compliance and Governance
variable "enable_config_rules" {
  description = "Enable AWS Config rules for compliance"
  type        = bool
  default     = true
}

variable "enable_cloudtrail" {
  description = "Enable CloudTrail for audit logging"
  type        = bool
  default     = true
}

variable "enable_guardduty" {
  description = "Enable GuardDuty for threat detection"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable Security Hub for security posture management"
  type        = bool
  default     = true
}

# Disaster Recovery
variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-west-2"
}

variable "enable_cross_region_replication" {
  description = "Enable cross-region replication for disaster recovery"
  type        = bool
  default     = true
}

variable "rto_minutes" {
  description = "Recovery Time Objective in minutes"
  type        = number
  default     = 60
}

variable "rpo_minutes" {
  description = "Recovery Point Objective in minutes"
  type        = number
  default     = 15
}

# Performance Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for all services"
  type        = bool
  default     = true
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

variable "performance_insights_retention_period" {
  description = "Performance Insights retention period (days)"
  type        = number
  default     = 7
}