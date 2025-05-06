# All variables that the user might want to change are located here, allowing for easy customization.

variable "location" {
  type        = string
  default     = "Sweden Central"
  description = "The Azure region where the resources will be deployed (default is Sweden Central)"
}

variable "environment" {
  type        = string
  description = "The environment (e.g., 'dev', 'prod') that the resources are being deployed into."
  default     = "dev"
}

variable "prefix" {
  type        = string
  description = "Prefix of the resource name"
  default     = "sl"
}

variable "workspace_display_name" {
  default = "ml-exjobb-sl"
}
