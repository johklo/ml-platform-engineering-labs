@description('Short project name used in Azure resource names.')
@minLength(5)
@maxLength(11)
param namePrefix string = 'mlplat'

@description('Azure region for the workspace and its dependencies.')
param location string = resourceGroup().location

@description('Opt in to creating Azure AI Search and Azure OpenAI for the RAG extension.')
param deployRagServices bool = false

@description('Azure OpenAI account name; must be globally unique.')
param openAiName string = '${namePrefix}-openai-${uniqueString(resourceGroup().id)}'

@description('Azure AI Search service name; must be globally unique.')
param searchName string = '${namePrefix}-search-${uniqueString(resourceGroup().id)}'

@description('Optional developer object ID for user-assigned Azure RAG data-plane roles.')
param developerPrincipalId string = ''

var suffix = uniqueString(resourceGroup().id)
var storageName = toLower(replace('${namePrefix}${suffix}', '-', ''))
var registryName = toLower(replace('${namePrefix}${suffix}', '-', ''))
var vaultName = take('${namePrefix}-kv-${suffix}', 24)
var workspaceName = '${namePrefix}-aml-${suffix}'
var insightsName = '${namePrefix}-appi-${suffix}'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    enablePurgeProtection: true
    publicNetworkAccess: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: insightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Request_Source: 'rest'
  }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'ML Platform Engineering Labs'
    description: 'Learning workspace for synthetic local-first platform engineering labs.'
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: registry.id
    publicNetworkAccess: 'Enabled'
  }
}

resource search 'Microsoft.Search/searchServices@2023-11-01' = if (deployRagServices) {
  name: searchName
  location: location
  sku: {
    name: 'basic'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    disableLocalAuth: true
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http403'
      }
    }
  }
}

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployRagServices) {
  name: openAiName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAiName
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployRagServices) {
  parent: openAi
  name: 'gpt-4o-mini'
  sku: {
    name: 'Standard'
    capacity: 1
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
}

resource searchIndexRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployRagServices && !empty(developerPrincipalId)) {
  name: guid(search.id, developerPrincipalId, 'search-index-data-contributor')
  scope: search
  properties: {
    principalId: developerPrincipalId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
    )
  }
}

resource openAiRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployRagServices && !empty(developerPrincipalId)) {
  name: guid(openAi.id, developerPrincipalId, 'cognitive-services-openai-user')
  scope: openAi
  properties: {
    principalId: developerPrincipalId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
    )
  }
}

output workspaceName string = workspace.name
output workspaceId string = workspace.id
output storageAccountName string = storage.name
output registryName string = registry.name
output keyVaultName string = keyVault.name
output applicationInsightsName string = appInsights.name
output searchServiceName string = deployRagServices ? search.name : ''
output openAiAccountName string = deployRagServices ? openAi.name : ''
output openAiDeploymentName string = deployRagServices ? chatDeployment.name : ''
