# ===============================================
# AWS DEPLOYMENT CONFIGURATION
# ===============================================
# Complete AWS infrastructure for trading bot platform
# Includes VPC, EKS, RDS, ElastiCache, and monitoring

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Backend configuration for state management
  backend "s3" {
    bucket         = "trading-bot-terraform-state"
    key            = "aws/trading-bot/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project          = "TradingBot"
      Environment      = var.environment
      ManagedBy        = "Terraform"
      Owner            = "Trading-Team"
      CostCenter       = "Engineering"
      Compliance       = "SOX"
      DataClassification = "Confidential"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  azs        = slice(data.aws_availability_zones.available.names, 0, 3)

  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ===============================================
# NETWORKING INFRASTRUCTURE
# ===============================================
module "vpc" {
  source = "../modules/vpc"

  cloud_provider = "aws"
  project_name   = var.project_name
  environment    = var.environment
  common_tags    = local.common_tags

  vpc_cidr                = var.vpc_cidr
  availability_zones      = local.azs
  public_subnet_cidrs     = [for i, az in local.azs : cidrsubnet(var.vpc_cidr, 8, i + 1)]
  private_subnet_cidrs    = [for i, az in local.azs : cidrsubnet(var.vpc_cidr, 8, i + 10)]
  database_subnet_cidrs   = [for i, az in local.azs : cidrsubnet(var.vpc_cidr, 8, i + 20)]
}

# ===============================================
# KUBERNETES CLUSTER (EKS)
# ===============================================
module "eks" {
  source = "../modules/eks"

  cloud_provider = "aws"
  cluster_name   = "${var.project_name}-${var.environment}"
  common_tags    = local.common_tags

  vpc_id             = module.vpc.aws_vpc_id
  public_subnet_ids  = module.vpc.aws_public_subnet_ids
  private_subnet_ids = module.vpc.aws_private_subnet_ids

  kubernetes_version   = var.kubernetes_version
  node_instance_types  = var.eks_node_instance_types
  node_desired_size    = var.eks_node_desired_size
  node_min_size        = var.eks_node_min_size
  node_max_size        = var.eks_node_max_size
  node_disk_size       = var.eks_node_disk_size

  api_server_access_cidrs = var.eks_api_access_cidrs
}

# ===============================================
# DATABASE INFRASTRUCTURE
# ===============================================
# PostgreSQL RDS Instance
resource "aws_db_subnet_group" "trading_db_subnet_group" {
  name       = "${var.project_name}-db-subnet-group"
  subnet_ids = module.vpc.aws_database_subnet_ids

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-db-subnet-group"
  })
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.project_name}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.aws_vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "PostgreSQL access from VPC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-rds-sg"
  })
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "db_password" {
  name        = "${var.project_name}-db-password"
  description = "Database password for trading bot"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = jsonencode({
    username = var.db_username
    password = random_password.db_password.result
  })
}

resource "aws_db_instance" "trading_db" {
  identifier = "${var.project_name}-db"

  # Engine configuration
  engine                = "postgres"
  engine_version        = "15.4"
  instance_class        = var.db_instance_class
  allocated_storage     = var.db_allocated_storage
  max_allocated_storage = var.db_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true

  # Database configuration
  db_name  = var.db_name
  username = var.db_username
  password = random_password.db_password.result
  port     = 5432

  # Network configuration
  db_subnet_group_name   = aws_db_subnet_group.trading_db_subnet_group.name
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
  publicly_accessible    = false

  # Backup configuration
  backup_retention_period = var.db_backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  copy_tags_to_snapshot  = true

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn

  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7

  # High availability
  multi_az = var.db_multi_az

  # Deletion protection
  deletion_protection = var.environment == "prod" ? true : false
  skip_final_snapshot = var.environment == "prod" ? false : true

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-db"
  })
}

# RDS Enhanced Monitoring Role
resource "aws_iam_role" "rds_enhanced_monitoring" {
  name = "${var.project_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_enhanced_monitoring" {
  role       = aws_iam_role.rds_enhanced_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ===============================================
# REDIS CACHE (ELASTICACHE)
# ===============================================
resource "aws_elasticache_subnet_group" "trading_cache_subnet_group" {
  name       = "${var.project_name}-cache-subnet-group"
  subnet_ids = module.vpc.aws_private_subnet_ids

  tags = local.common_tags
}

resource "aws_security_group" "elasticache_sg" {
  name        = "${var.project_name}-elasticache-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.aws_vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Redis access from VPC"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-elasticache-sg"
  })
}

resource "aws_elasticache_replication_group" "trading_redis" {
  replication_group_id         = "${var.project_name}-redis"
  description                  = "Redis cluster for trading bot"

  # Engine configuration
  engine               = "redis"
  engine_version       = "7.0"
  parameter_group_name = "default.redis7"
  port                 = 6379
  node_type            = var.redis_node_type

  # Cluster configuration
  num_cache_clusters = var.redis_num_cache_nodes

  # Network configuration
  subnet_group_name  = aws_elasticache_subnet_group.trading_cache_subnet_group.name
  security_group_ids = [aws_security_group.elasticache_sg.id]

  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth_token.result

  # Backup
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  # Maintenance
  maintenance_window = "sun:05:00-sun:07:00"

  # Monitoring
  notification_topic_arn = aws_sns_topic.alerts.arn

  tags = local.common_tags
}

resource "random_password" "redis_auth_token" {
  length  = 32
  special = true
}

resource "aws_secretsmanager_secret" "redis_auth_token" {
  name        = "${var.project_name}-redis-auth-token"
  description = "Redis authentication token"

  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "redis_auth_token" {
  secret_id     = aws_secretsmanager_secret.redis_auth_token.id
  secret_string = random_password.redis_auth_token.result
}

# ===============================================
# APPLICATION LOAD BALANCER
# ===============================================
resource "aws_security_group" "alb_sg" {
  name        = "${var.project_name}-alb-sg"
  description = "Security group for Application Load Balancer"
  vpc_id      = module.vpc.aws_vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP access"
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-alb-sg"
  })
}

resource "aws_lb" "trading_alb" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets           = module.vpc.aws_public_subnet_ids

  enable_deletion_protection = var.environment == "prod" ? true : false

  # Enable access logs
  access_logs {
    bucket  = aws_s3_bucket.alb_logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = local.common_tags
}

# S3 bucket for ALB access logs
resource "aws_s3_bucket" "alb_logs" {
  bucket        = "${var.project_name}-alb-logs-${random_string.bucket_suffix.result}"
  force_destroy = var.environment != "prod"

  tags = local.common_tags
}

resource "aws_s3_bucket_policy" "alb_logs_policy" {
  bucket = aws_s3_bucket.alb_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_elb_service_account.main.id}:root"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/alb-logs/*"
      },
      {
        Effect = "Allow"
        Principal = {
          Service = "delivery.logs.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.alb_logs.arn}/alb-logs/*"
      }
    ]
  })
}

resource "aws_s3_bucket_versioning" "alb_logs_versioning" {
  bucket = aws_s3_bucket.alb_logs.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "alb_logs_encryption" {
  bucket = aws_s3_bucket.alb_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

data "aws_elb_service_account" "main" {}

# ===============================================
# MONITORING AND ALERTING
# ===============================================
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"

  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count = length(var.alert_email_addresses)

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email_addresses[count.index]
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "trading_app_logs" {
  name              = "/aws/trading-bot/application"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "trading_api_logs" {
  name              = "/aws/trading-bot/api"
  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

# ===============================================
# IAM ROLES AND POLICIES
# ===============================================
# IAM role for EKS to access AWS services
resource "aws_iam_role" "trading_bot_role" {
  name = "${var.project_name}-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "trading_bot_policy" {
  name = "${var.project_name}-service-policy"
  role = aws_iam_role.trading_bot_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "rds:DescribeDBInstances",
          "elasticache:DescribeReplicationGroups",
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}