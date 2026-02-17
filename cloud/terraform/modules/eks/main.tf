# ===============================================
# KUBERNETES CLUSTER MODULE (EKS/GKE/AKS)
# ===============================================
# Enterprise-grade Kubernetes clusters across cloud providers

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

# ===============================================
# AWS EKS CLUSTER
# ===============================================
resource "aws_iam_role" "eks_cluster_role" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name = "${var.cluster_name}-cluster-role"

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

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  count = var.cloud_provider == "aws" ? 1 : 0

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role[0].name
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  count = var.cloud_provider == "aws" ? 1 : 0

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster_role[0].name
}

resource "aws_security_group" "eks_cluster_sg" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name        = "${var.cluster_name}-cluster-sg"
  description = "Security group for EKS cluster"
  vpc_id      = var.vpc_id

  # HTTPS access for API server
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS access to EKS API"
  }

  # Node group communication
  ingress {
    from_port = 1025
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Node group communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-cluster-sg"
  })
}

resource "aws_eks_cluster" "trading_cluster" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role[0].arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(var.public_subnet_ids, var.private_subnet_ids)
    security_group_ids      = [aws_security_group.eks_cluster_sg[0].id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.api_server_access_cidrs
  }

  # Enable logging
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # Enable encryption
  encryption_config {
    resources = ["secrets"]
    provider {
      key_id = aws_kms_key.eks_encryption_key[0].arn
    }
  }

  tags = merge(var.common_tags, {
    Name = var.cluster_name
    Type = "kubernetes-cluster"
  })

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
  ]
}

# KMS key for EKS encryption
resource "aws_kms_key" "eks_encryption_key" {
  count = var.cloud_provider == "aws" ? 1 : 0

  description         = "KMS key for EKS cluster encryption"
  key_usage           = "ENCRYPT_DECRYPT"
  deletion_window_in_days = 7

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-encryption-key"
  })
}

resource "aws_kms_alias" "eks_encryption_key_alias" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name          = "alias/${var.cluster_name}-eks-encryption"
  target_key_id = aws_kms_key.eks_encryption_key[0].key_id
}

# ===============================================
# EKS NODE GROUP
# ===============================================
resource "aws_iam_role" "eks_node_role" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name = "${var.cluster_name}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  count = var.cloud_provider == "aws" ? 1 : 0

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_node_role[0].name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  count = var.cloud_provider == "aws" ? 1 : 0

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_node_role[0].name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  count = var.cloud_provider == "aws" ? 1 : 0

  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_node_role[0].name
}

resource "aws_eks_node_group" "trading_nodes" {
  count = var.cloud_provider == "aws" ? 1 : 0

  cluster_name    = aws_eks_cluster.trading_cluster[0].name
  node_group_name = "${var.cluster_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node_role[0].arn
  subnet_ids      = var.private_subnet_ids

  instance_types = var.node_instance_types
  capacity_type  = "ON_DEMAND"
  ami_type       = "AL2_x86_64"

  scaling_config {
    desired_size = var.node_desired_size
    max_size     = var.node_max_size
    min_size     = var.node_min_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  # Enable launch template for advanced configuration
  launch_template {
    name    = aws_launch_template.eks_node_template[0].name
    version = aws_launch_template.eks_node_template[0].latest_version
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-nodes"
  })

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]
}

resource "aws_launch_template" "eks_node_template" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name_prefix   = "${var.cluster_name}-node-template-"
  image_id      = data.aws_ssm_parameter.eks_ami_release_version[0].value
  instance_type = var.node_instance_types[0]

  vpc_security_group_ids = [aws_security_group.eks_node_sg[0].id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = var.node_disk_size
      volume_type = "gp3"
      encrypted   = true
      kms_key_id  = aws_kms_key.eks_encryption_key[0].arn
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = aws_eks_cluster.trading_cluster[0].name
    cluster_endpoint    = aws_eks_cluster.trading_cluster[0].endpoint
    cluster_ca_data     = aws_eks_cluster.trading_cluster[0].certificate_authority[0].data
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.common_tags, {
      Name = "${var.cluster_name}-node"
    })
  }

  tags = var.common_tags
}

resource "aws_security_group" "eks_node_sg" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name        = "${var.cluster_name}-node-sg"
  description = "Security group for EKS nodes"
  vpc_id      = var.vpc_id

  # Node to node communication
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
    description = "Node to node communication"
  }

  # Cluster to node communication
  ingress {
    from_port       = 1025
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster_sg[0].id]
    description     = "Cluster to node communication"
  }

  # HTTPS from cluster
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster_sg[0].id]
    description     = "HTTPS from cluster"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound traffic"
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-node-sg"
  })
}

data "aws_ssm_parameter" "eks_ami_release_version" {
  count = var.cloud_provider == "aws" ? 1 : 0

  name = "/aws/service/eks/optimized-ami/${var.kubernetes_version}/amazon-linux-2/recommended/image_id"
}

# ===============================================
# GOOGLE GKE CLUSTER
# ===============================================
resource "google_container_cluster" "trading_cluster" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name     = var.cluster_name
  location = var.gcp_region
  network  = var.gcp_vpc_name
  subnetwork = var.gcp_subnet_name

  # Enable network policy
  network_policy {
    enabled = true
  }

  # IP allocation policy for secondary ranges
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Enable private nodes
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "10.100.0.0/28"
  }

  # Master authorized networks
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.gcp_project_id}.svc.id.goog"
  }

  # Enable database encryption
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.gke_encryption_key[0].id
  }

  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1

  # Addons
  addons_config {
    horizontal_pod_autoscaling {
      disabled = false
    }
    http_load_balancing {
      disabled = false
    }
    network_policy_config {
      disabled = false
    }
  }
}

resource "google_container_node_pool" "trading_nodes" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name       = "${var.cluster_name}-nodes"
  location   = var.gcp_region
  cluster    = google_container_cluster.trading_cluster[0].name
  node_count = var.node_desired_size

  node_config {
    preemptible  = false
    machine_type = var.gcp_node_machine_type

    # Google recommends custom service accounts with minimal permissions
    service_account = google_service_account.gke_node_sa[0].email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    disk_size_gb = var.node_disk_size
    disk_type    = "pd-ssd"

    # Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Security
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  autoscaling {
    min_node_count = var.node_min_size
    max_node_count = var.node_max_size
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_service_account" "gke_node_sa" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  account_id   = "${var.cluster_name}-node-sa"
  display_name = "GKE Node Service Account"
}

resource "google_project_iam_member" "gke_node_sa_roles" {
  count = var.cloud_provider == "gcp" ? length(var.gke_node_sa_roles) : 0

  project = var.gcp_project_id
  role    = var.gke_node_sa_roles[count.index]
  member  = "serviceAccount:${google_service_account.gke_node_sa[0].email}"
}

resource "google_kms_key_ring" "gke_keyring" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name     = "${var.cluster_name}-keyring"
  location = var.gcp_region
}

resource "google_kms_crypto_key" "gke_encryption_key" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name     = "${var.cluster_name}-encryption-key"
  key_ring = google_kms_key_ring.gke_keyring[0].id

  lifecycle {
    prevent_destroy = true
  }
}

# ===============================================
# AZURE AKS CLUSTER
# ===============================================
resource "azurerm_kubernetes_cluster" "trading_cluster" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                = var.cluster_name
  location            = var.azure_location
  resource_group_name = var.azure_resource_group_name
  dns_prefix          = "${var.cluster_name}-dns"
  kubernetes_version  = var.kubernetes_version

  # Default node pool
  default_node_pool {
    name                = "system"
    node_count          = var.node_desired_size
    vm_size             = var.azure_node_vm_size
    vnet_subnet_id      = var.azure_subnet_id
    enable_auto_scaling = true
    min_count          = var.node_min_size
    max_count          = var.node_max_size
    os_disk_size_gb    = var.node_disk_size
    os_disk_type       = "Managed"

    # Enable cluster autoscaler
    upgrade_settings {
      max_surge = "10%"
    }
  }

  # Identity
  identity {
    type = "SystemAssigned"
  }

  # Network profile
  network_profile {
    network_plugin     = "azure"
    network_policy     = "calico"
    dns_service_ip     = "10.5.0.10"
    docker_bridge_cidr = "172.17.0.1/16"
    service_cidr       = "10.5.0.0/16"
  }

  # Enable private cluster
  private_cluster_enabled = true

  # Role-based access control
  role_based_access_control_enabled = true

  azure_active_directory_role_based_access_control {
    managed = true
    admin_group_object_ids = var.aks_admin_group_object_ids
  }

  # Monitoring
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.aks_workspace[0].id
  }

  tags = var.common_tags
}

resource "azurerm_log_analytics_workspace" "aks_workspace" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                = "${var.cluster_name}-workspace"
  location            = var.azure_location
  resource_group_name = var.azure_resource_group_name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.common_tags
}