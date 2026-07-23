import { useState } from 'react'

import { predictUrl } from '../services/predictionApi'

export function useUrlPrediction() {
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submitUrl = async (url, modelId) => {
    setLoading(true)
    setError('')
    setPrediction(null)

    try {
      const payload = await predictUrl(url, modelId)
      setPrediction(payload)
      return payload
    } catch (requestError) {
      setError(requestError.message)
      throw requestError
    } finally {
      setLoading(false)
    }
  }

  return {
    prediction,
    loading,
    error,
    submitUrl,
  }
}
