# ===============================================
# VPC MODULE OUTPUTS
# ===============================================

# ===============================================
# AWS OUTPUTS
# ===============================================
output "aws_vpc_id" {
  description = "ID of the AWS VPC"
  value       = var.cloud_provider == "aws" ? aws_vpc.trading_vpc[0].id : null
}

output "aws_vpc_cidr" {
  description = "CIDR block of the AWS VPC"
  value       = var.cloud_provider == "aws" ? aws_vpc.trading_vpc[0].cidr_block : null
}

output "aws_public_subnet_ids" {
  description = "IDs of the AWS public subnets"
  value       = var.cloud_provider == "aws" ? aws_subnet.public_subnets[*].id : []
}

output "aws_private_subnet_ids" {
  description = "IDs of the AWS private subnets"
  value       = var.cloud_provider == "aws" ? aws_subnet.private_subnets[*].id : []
}

output "aws_database_subnet_ids" {
  description = "IDs of the AWS database subnets"
  value       = var.cloud_provider == "aws" ? aws_subnet.database_subnets[*].id : []
}

output "aws_internet_gateway_id" {
  description = "ID of the AWS Internet Gateway"
  value       = var.cloud_provider == "aws" ? aws_internet_gateway.trading_igw[0].id : null
}

output "aws_nat_gateway_ids" {
  description = "IDs of the AWS NAT Gateways"
  value       = var.cloud_provider == "aws" ? aws_nat_gateway.nat_gateways[*].id : []
}

# ===============================================
# GCP OUTPUTS
# ===============================================
output "gcp_vpc_name" {
  description = "Name of the GCP VPC"
  value       = var.cloud_provider == "gcp" ? google_compute_network.trading_vpc[0].name : null
}

output "gcp_vpc_self_link" {
  description = "Self link of the GCP VPC"
  value       = var.cloud_provider == "gcp" ? google_compute_network.trading_vpc[0].self_link : null
}

output "gcp_public_subnet_name" {
  description = "Name of the GCP public subnet"
  value       = var.cloud_provider == "gcp" ? google_compute_subnetwork.public_subnet[0].name : null
}

output "gcp_private_subnet_name" {
  description = "Name of the GCP private subnet"
  value       = var.cloud_provider == "gcp" ? google_compute_subnetwork.private_subnet[0].name : null
}

output "gcp_public_subnet_self_link" {
  description = "Self link of the GCP public subnet"
  value       = var.cloud_provider == "gcp" ? google_compute_subnetwork.public_subnet[0].self_link : null
}

output "gcp_private_subnet_self_link" {
  description = "Self link of the GCP private subnet"
  value       = var.cloud_provider == "gcp" ? google_compute_subnetwork.private_subnet[0].self_link : null
}

output "gcp_pods_cidr" {
  description = "CIDR range for GKE pods"
  value       = var.cloud_provider == "gcp" ? var.gcp_pods_cidr : null
}

output "gcp_services_cidr" {
  description = "CIDR range for GKE services"
  value       = var.cloud_provider == "gcp" ? var.gcp_services_cidr : null
}

# ===============================================
# AZURE OUTPUTS
# ===============================================
output "azure_resource_group_name" {
  description = "Name of the Azure resource group"
  value       = var.cloud_provider == "azure" ? azurerm_resource_group.trading_rg[0].name : null
}

output "azure_resource_group_location" {
  description = "Location of the Azure resource group"
  value       = var.cloud_provider == "azure" ? azurerm_resource_group.trading_rg[0].location : null
}

output "azure_vnet_name" {
  description = "Name of the Azure virtual network"
  value       = var.cloud_provider == "azure" ? azurerm_virtual_network.trading_vnet[0].name : null
}

output "azure_vnet_id" {
  description = "ID of the Azure virtual network"
  value       = var.cloud_provider == "azure" ? azurerm_virtual_network.trading_vnet[0].id : null
}

output "azure_public_subnet_id" {
  description = "ID of the Azure public subnet"
  value       = var.cloud_provider == "azure" ? azurerm_subnet.public_subnet[0].id : null
}

output "azure_private_subnet_id" {
  description = "ID of the Azure private subnet"
  value       = var.cloud_provider == "azure" ? azurerm_subnet.private_subnet[0].id : null
}

output "azure_database_subnet_id" {
  description = "ID of the Azure database subnet"
  value       = var.cloud_provider == "azure" ? azurerm_subnet.database_subnet[0].id : null
}

# ===============================================
# GENERAL OUTPUTS
# ===============================================
output "vpc_info" {
  description = "VPC information for the selected cloud provider"
  value = {
    cloud_provider = var.cloud_provider
    aws = var.cloud_provider == "aws" ? {
      vpc_id               = aws_vpc.trading_vpc[0].id
      public_subnet_ids    = aws_subnet.public_subnets[*].id
      private_subnet_ids   = aws_subnet.private_subnets[*].id
      database_subnet_ids  = aws_subnet.database_subnets[*].id
    } : null
    gcp = var.cloud_provider == "gcp" ? {
      vpc_name            = google_compute_network.trading_vpc[0].name
      public_subnet_name  = google_compute_subnetwork.public_subnet[0].name
      private_subnet_name = google_compute_subnetwork.private_subnet[0].name
    } : null
    azure = var.cloud_provider == "azure" ? {
      resource_group_name   = azurerm_resource_group.trading_rg[0].name
      vnet_name            = azurerm_virtual_network.trading_vnet[0].name
      public_subnet_id     = azurerm_subnet.public_subnet[0].id
      private_subnet_id    = azurerm_subnet.private_subnet[0].id
      database_subnet_id   = azurerm_subnet.database_subnet[0].id
    } : null
  }
}