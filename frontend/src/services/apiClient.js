const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    return null
  }

  return response.json()
}

export async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  })

  const payload = await parseResponse(response)

  if (!response.ok) {
    const message =
      payload?.detail || payload?.message || 'The API request could not be completed.'
    throw new Error(message)
  }

  return payload
}
