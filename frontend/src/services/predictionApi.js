import { apiRequest } from './apiClient'

const API_PREFIX = '/api/v1'

export function getHealth() {
  return apiRequest(`${API_PREFIX}/health`)
}

export function getModelInfo(modelId) {
  const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''
  return apiRequest(`${API_PREFIX}/model-info${query}`)
}

export function getModels() {
  return apiRequest(`${API_PREFIX}/models`)
}

export function predictUrl(url, modelId) {
  return apiRequest(`${API_PREFIX}/predict`, {
    method: 'POST',
    body: JSON.stringify({
      url,
      ...(modelId ? { model_id: modelId } : {}),
    }),
  })
}
