# ===============================================
# KUBERNETES CLUSTER MODULE VARIABLES
# ===============================================

variable "cloud_provider" {
  description = "Cloud provider to deploy to (aws, gcp, azure)"
  type        = string
  validation {
    condition     = contains(["aws", "gcp", "azure"], var.cloud_provider)
    error_message = "Cloud provider must be one of: aws, gcp, azure."
  }
}

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "trading-bot-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
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
# NODE CONFIGURATION
# ===============================================
variable "node_desired_size" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}

variable "node_min_size" {
  description = "Minimum number of nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of nodes"
  type        = number
  default     = 20
}

variable "node_disk_size" {
  description = "Size of node disk in GB"
  type        = number
  default     = 100
}

# ===============================================
# AWS-SPECIFIC VARIABLES
# ===============================================
variable "vpc_id" {
  description = "VPC ID for AWS resources"
  type        = string
  default     = ""
}

variable "public_subnet_ids" {
  description = "List of public subnet IDs"
  type        = list(string)
  default     = []
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs"
  type        = list(string)
  default     = []
}

variable "node_instance_types" {
  description = "List of instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "api_server_access_cidrs" {
  description = "CIDR blocks that can access the API server"
  type        = list(string)
  default     = ["0.0.0.0/0"]
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

variable "gcp_vpc_name" {
  description = "GCP VPC network name"
  type        = string
  default     = ""
}

variable "gcp_subnet_name" {
  description = "GCP subnet name"
  type        = string
  default     = ""
}

variable "gcp_node_machine_type" {
  description = "GCP machine type for nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "gke_node_sa_roles" {
  description = "IAM roles for GKE node service account"
  type        = list(string)
  default = [
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer"
  ]
}

# ===============================================
# AZURE-SPECIFIC VARIABLES
# ===============================================
variable "azure_location" {
  description = "Azure region"
  type        = string
  default     = "East US"
}

variable "azure_resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = ""
}

variable "azure_subnet_id" {
  description = "Azure subnet ID"
  type        = string
  default     = ""
}

variable "azure_node_vm_size" {
  description = "Azure VM size for nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "aks_admin_group_object_ids" {
  description = "Azure AD group object IDs for AKS admin access"
  type        = list(string)
  default     = []
}

# ===============================================
# CLUSTER ADD-ONS AND FEATURES
# ===============================================
variable "enable_pod_security_policy" {
  description = "Enable Pod Security Policy"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable network policy"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable horizontal pod autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable metrics server"
  type        = bool
  default     = true
}

variable "enable_ingress_nginx" {
  description = "Enable NGINX ingress controller"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable cert-manager for SSL certificates"
  type        = bool
  default     = true
}

variable "enable_external_dns" {
  description = "Enable external-dns for automatic DNS management"
  type        = bool
  default     = false
}

variable "enable_prometheus_monitoring" {
  description = "Enable Prometheus monitoring stack"
  type        = bool
  default     = true
}

# ===============================================
# SECURITY CONFIGURATION
# ===============================================
variable "encryption_config_key_id" {
  description = "KMS key ID for cluster encryption"
  type        = string
  default     = ""
}

variable "cluster_endpoint_private_access" {
  description = "Enable private API server endpoint access"
  type        = bool
  default     = true
}

variable "cluster_endpoint_public_access" {
  description = "Enable public API server endpoint access"
  type        = bool
  default     = true
}

variable "cluster_log_types" {
  description = "List of control plane log types to enable"
  type        = list(string)
  default     = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
}

# ===============================================
# BACKUP AND DISASTER RECOVERY
# ===============================================
variable "enable_backup" {
  description = "Enable cluster backup"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_disaster_recovery" {
  description = "Enable disaster recovery setup"
  type        = bool
  default     = true
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = ""
}