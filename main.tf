# retrieves Azure client configuration, containing metadata like tenant id (needed for key vault)
data "azurerm_client_config" "current" {}

# Creates the azure resource group where all resources are contained in
# The syntax is the resource resource_name resource_alias, where the alias is specific for terraform
resource "azurerm_resource_group" "ml_rg" {
  location = var.location
  name = "ml-${var.prefix}-${var.environment}-rg"
  tags = {
    environment = var.environment
  }
}

# Application Insights resource, which provides monitoring and diagnostics
resource "azurerm_application_insights" "app_ins_ml" {
  name                = "ml-${var.prefix}-${var.environment}-ai"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  application_type    = "web"
}

# Azure Key Vault, which is used to securely store keys and secrets,
resource "azurerm_key_vault" "akv_ml" {
  name                = "ml-${var.prefix}-${var.environment}-kv"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  purge_protection_enabled = false
}

# Azure Storage Account, which is used to store data like files, blobs, and other resources. 
# Each storage account needs to be unique across all workspaces so adding random number at the end 
resource "azurerm_storage_account" "asa_ml" {
  # Need to be unique each time
  name                     = "ml${var.prefix}${var.environment}${random_integer.suffix.result}"
  location                 = azurerm_resource_group.ml_rg.location
  resource_group_name      = azurerm_resource_group.ml_rg.name
  account_tier             = "Standard"
  account_replication_type = "GRS"
  allow_nested_items_to_be_public = false
}

resource "random_integer" "suffix" {
  min = 100000
  max = 999999
}

# Datastore Container, where the preprocessed data will be contained 
resource "azurerm_storage_container" "preprocessed_container" {
  name                  = "preprocessed-data"  
  storage_account_name  = azurerm_storage_account.asa_ml.name
  container_access_type = "private"  
}

# Azure Machine Learning Workspace, the main resource for managing machine learning experiments linked to the resources above
resource "azurerm_machine_learning_workspace" "aml_ws" {
  name = "ml-${lower(var.prefix)}-${lower(var.environment)}-${random_integer.suffix.result}-ws" 
  friendly_name           = var.workspace_display_name
  location                = azurerm_resource_group.ml_rg.location
  resource_group_name     = azurerm_resource_group.ml_rg.name
  application_insights_id = azurerm_application_insights.app_ins_ml.id
  key_vault_id            = azurerm_key_vault.akv_ml.id
  storage_account_id      = azurerm_storage_account.asa_ml.id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }
}

