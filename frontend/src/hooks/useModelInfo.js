import { useCallback, useEffect, useState } from 'react'

import { getModelInfo, getModels } from '../services/predictionApi'

export function useModelInfo() {
  const [modelInfo, setModelInfo] = useState(null)
  const [modelOptions, setModelOptions] = useState([])
  const [selectedModelId, setSelectedModelId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadModelInfo = useCallback(async (modelId) => {
    setLoading(true)
    setError('')

    try {
      const payload = await getModelInfo(modelId)
      setModelInfo(payload)
      setSelectedModelId(payload.model_id)
      return payload
    } catch (requestError) {
      setError(requestError.message)
      throw requestError
    } finally {
      setLoading(false)
    }
  }, [])

  const loadInitialModelState = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const optionsPayload = await getModels()
      setModelOptions(optionsPayload.models)
      const modelId = optionsPayload.default_model_id
      const infoPayload = await getModelInfo(modelId)
      setModelInfo(infoPayload)
      setSelectedModelId(infoPayload.model_id)
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadInitialModelState()
  }, [loadInitialModelState])

  return {
    modelInfo,
    modelOptions,
    selectedModelId,
    loading,
    error,
    selectModel: loadModelInfo,
    refreshModelInfo: loadModelInfo,
  }
}
