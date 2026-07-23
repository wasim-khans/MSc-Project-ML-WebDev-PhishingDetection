import { useState } from 'react'

export function usePredictionHistory(limit = 6) {
  const [history, setHistory] = useState([])

  const recordPrediction = (prediction) => {
    if (!prediction?.url) {
      return
    }

    setHistory((currentHistory) => {
      const nextEntry = {
        id: `${Date.now()}-${prediction.url}`,
        url: prediction.url,
        predictedClass: prediction.predicted_class,
        confidence: prediction.confidence,
        modelName: prediction.model_name,
        trainingScenario: prediction.training_scenario,
        createdAt: new Date().toISOString(),
      }

      return [nextEntry, ...currentHistory].slice(0, limit)
    })
  }

  return {
    history,
    recordPrediction,
  }
}
