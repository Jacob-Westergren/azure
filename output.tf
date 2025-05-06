output "storage_account_name" {
  value = azurerm_storage_account.asa_ml.name
  description = "The name of the Storage Account"
}

output "storage_account_blob_endpoint" {
  value = azurerm_storage_account.asa_ml.primary_blob_endpoint
  description = "The blob service endpoint of the Storage Account"
}

output "storage_container_name" {
  value = azurerm_storage_container.preprocessed_container.name
  description = "The name of the container where preprocessed data is stored"
}

output "aml_workspace_name" {
  value = azurerm_machine_learning_workspace.aml_ws.name
  description = "The name of the Azure ML Workspace"
}
