function fallbackLabel(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

export const FEATURE_ORDER = [
  'url_length',
  'domain_length',
  'path_length',
  'dot_count',
  'hyphen_count',
  'digit_count',
  'special_char_count',
  'has_https',
  'has_ip_address',
  'has_at_symbol',
  'subdomain_count',
  'query_param_count',
  'suspicious_word_count',
  'tld_length',
  'has_url_shortener',
]

const FEATURE_METADATA = {
  url_length: {
    label: 'URL length',
    description: 'Total number of characters in the full URL.',
  },
  domain_length: {
    label: 'Domain length',
    description: 'Number of characters in the hostname part of the URL.',
  },
  path_length: {
    label: 'Path length',
    description: 'Number of characters after the domain, before the query string.',
  },
  dot_count: {
    label: 'Dot count',
    description: 'How many full-stop characters appear in the URL.',
  },
  hyphen_count: {
    label: 'Hyphen count',
    description: 'How many hyphens appear in the URL.',
  },
  digit_count: {
    label: 'Digit count',
    description: 'How many numeric characters appear in the URL.',
  },
  special_char_count: {
    label: 'Special character count',
    description: 'Number of selected special characters such as ?, =, &, @, %, and _.',
  },
  has_https: {
    label: 'Uses HTTPS',
    description: 'Returns 1 when the URL starts with https://, otherwise 0.',
    isBoolean: true,
  },
  has_ip_address: {
    label: 'Uses IP address',
    description: 'Returns 1 when the hostname looks like an IP address, otherwise 0.',
    isBoolean: true,
  },
  has_at_symbol: {
    label: 'Has @ symbol',
    description: 'Returns 1 when the URL contains an @ symbol, otherwise 0.',
    isBoolean: true,
  },
  subdomain_count: {
    label: 'Subdomain count',
    description: 'Approximate number of labels before the main domain.',
  },
  query_param_count: {
    label: 'Query parameter count',
    description: 'Number of key=value style parameters in the query string.',
  },
  suspicious_word_count: {
    label: 'Suspicious word count',
    description: 'Count of keywords such as login, verify, secure, update, and bank.',
  },
  tld_length: {
    label: 'TLD length',
    description: 'Number of characters in the top-level domain such as com or org.',
  },
  has_url_shortener: {
    label: 'Uses URL shortener',
    description: 'Returns 1 when the domain matches a known URL-shortening service.',
    isBoolean: true,
  },
}

export function getFeatureMetadata(featureKey) {
  return (
    FEATURE_METADATA[featureKey] || {
      label: fallbackLabel(featureKey),
      description: 'Extracted URL-only feature used by the machine-learning model.',
    }
  )
}
