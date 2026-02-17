# ===============================================
# VPC MODULE FOR MULTI-CLOUD DEPLOYMENT
# ===============================================
# Creates secure network infrastructure across AWS/GCP/Azure

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
  }
}

# ===============================================
# AWS VPC CONFIGURATION
# ===============================================
resource "aws_vpc" "trading_vpc" {
  count = var.cloud_provider == "aws" ? 1 : 0

  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-vpc"
    Type = "trading-infrastructure"
  })
}

resource "aws_internet_gateway" "trading_igw" {
  count = var.cloud_provider == "aws" ? 1 : 0

  vpc_id = aws_vpc.trading_vpc[0].id

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-igw"
  })
}

resource "aws_subnet" "public_subnets" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  vpc_id                  = aws_vpc.trading_vpc[0].id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-public-${var.availability_zones[count.index]}"
    Type = "public"
    Tier = "frontend"
  })
}

resource "aws_subnet" "private_subnets" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  vpc_id            = aws_vpc.trading_vpc[0].id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-private-${var.availability_zones[count.index]}"
    Type = "private"
    Tier = "backend"
  })
}

resource "aws_subnet" "database_subnets" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  vpc_id            = aws_vpc.trading_vpc[0].id
  cidr_block        = var.database_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-database-${var.availability_zones[count.index]}"
    Type = "database"
    Tier = "data"
  })
}

# NAT Gateways for private subnet internet access
resource "aws_eip" "nat_eips" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  domain = "vpc"

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-nat-eip-${count.index + 1}"
  })
}

resource "aws_nat_gateway" "nat_gateways" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.public_subnets[count.index].id

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-nat-${count.index + 1}"
  })

  depends_on = [aws_internet_gateway.trading_igw]
}

# Route Tables
resource "aws_route_table" "public_rt" {
  count = var.cloud_provider == "aws" ? 1 : 0

  vpc_id = aws_vpc.trading_vpc[0].id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.trading_igw[0].id
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-public-rt"
  })
}

resource "aws_route_table" "private_rt" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  vpc_id = aws_vpc.trading_vpc[0].id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }

  tags = merge(var.common_tags, {
    Name = "${var.project_name}-private-rt-${count.index + 1}"
  })
}

# Route Table Associations
resource "aws_route_table_association" "public_rta" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt[0].id
}

resource "aws_route_table_association" "private_rta" {
  count = var.cloud_provider == "aws" ? length(var.availability_zones) : 0

  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt[count.index].id
}

# ===============================================
# GOOGLE CLOUD VPC CONFIGURATION
# ===============================================
resource "google_compute_network" "trading_vpc" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
  routing_mode           = "REGIONAL"

  description = "Trading Bot Platform VPC"
}

resource "google_compute_subnetwork" "public_subnet" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name          = "${var.project_name}-public-subnet"
  ip_cidr_range = var.gcp_public_subnet_cidr
  network       = google_compute_network.trading_vpc[0].id
  region        = var.gcp_region

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.gcp_pods_cidr
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.gcp_services_cidr
  }
}

resource "google_compute_subnetwork" "private_subnet" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name                     = "${var.project_name}-private-subnet"
  ip_cidr_range            = var.gcp_private_subnet_cidr
  network                  = google_compute_network.trading_vpc[0].id
  region                   = var.gcp_region
  private_ip_google_access = true
}

resource "google_compute_router" "nat_router" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name    = "${var.project_name}-nat-router"
  network = google_compute_network.trading_vpc[0].id
  region  = var.gcp_region
}

resource "google_compute_router_nat" "nat_gateway" {
  count = var.cloud_provider == "gcp" ? 1 : 0

  name                               = "${var.project_name}-nat-gateway"
  router                             = google_compute_router.nat_router[0].name
  region                             = var.gcp_region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.private_subnet[0].id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
}

# ===============================================
# AZURE VIRTUAL NETWORK CONFIGURATION
# ===============================================
resource "azurerm_resource_group" "trading_rg" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name     = "${var.project_name}-rg"
  location = var.azure_location

  tags = var.common_tags
}

resource "azurerm_virtual_network" "trading_vnet" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                = "${var.project_name}-vnet"
  address_space       = [var.azure_vnet_cidr]
  location            = azurerm_resource_group.trading_rg[0].location
  resource_group_name = azurerm_resource_group.trading_rg[0].name

  tags = var.common_tags
}

resource "azurerm_subnet" "public_subnet" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                 = "${var.project_name}-public-subnet"
  resource_group_name  = azurerm_resource_group.trading_rg[0].name
  virtual_network_name = azurerm_virtual_network.trading_vnet[0].name
  address_prefixes     = [var.azure_public_subnet_cidr]
}

resource "azurerm_subnet" "private_subnet" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                 = "${var.project_name}-private-subnet"
  resource_group_name  = azurerm_resource_group.trading_rg[0].name
  virtual_network_name = azurerm_virtual_network.trading_vnet[0].name
  address_prefixes     = [var.azure_private_subnet_cidr]
}

resource "azurerm_subnet" "database_subnet" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                 = "${var.project_name}-database-subnet"
  resource_group_name  = azurerm_resource_group.trading_rg[0].name
  virtual_network_name = azurerm_virtual_network.trading_vnet[0].name
  address_prefixes     = [var.azure_database_subnet_cidr]

  delegation {
    name = "postgres_delegation"

    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_public_ip" "nat_gateway_ip" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                = "${var.project_name}-nat-gateway-ip"
  location            = azurerm_resource_group.trading_rg[0].location
  resource_group_name = azurerm_resource_group.trading_rg[0].name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = var.common_tags
}

resource "azurerm_nat_gateway" "nat_gateway" {
  count = var.cloud_provider == "azure" ? 1 : 0

  name                    = "${var.project_name}-nat-gateway"
  location                = azurerm_resource_group.trading_rg[0].location
  resource_group_name     = azurerm_resource_group.trading_rg[0].name
  sku_name                = "Standard"
  idle_timeout_in_minutes = 10

  tags = var.common_tags
}

resource "azurerm_nat_gateway_public_ip_association" "nat_gateway_ip_association" {
  count = var.cloud_provider == "azure" ? 1 : 0

  nat_gateway_id       = azurerm_nat_gateway.nat_gateway[0].id
  public_ip_address_id = azurerm_public_ip.nat_gateway_ip[0].id
}

resource "azurerm_subnet_nat_gateway_association" "private_subnet_nat" {
  count = var.cloud_provider == "azure" ? 1 : 0

  subnet_id      = azurerm_subnet.private_subnet[0].id
  nat_gateway_id = azurerm_nat_gateway.nat_gateway[0].id
}