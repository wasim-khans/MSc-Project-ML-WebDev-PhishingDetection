import unittest

from machine_learning.scripts.core.url_feature_extractor import (
    FEATURE_NAMES,
    extract_features,
)


class FeatureExtractorTests(unittest.TestCase):
    def test_extracts_stable_feature_dictionary_for_https_login_url(self):
        features = extract_features("https://secure-login.example.com/verify/account?id=123&next=home")

        self.assertEqual(list(features.keys()), FEATURE_NAMES)
        self.assertEqual(features["has_https"], 1)
        self.assertEqual(features["has_ip_address"], 0)
        self.assertEqual(features["has_at_symbol"], 0)
        self.assertEqual(features["subdomain_count"], 1)
        self.assertEqual(features["query_param_count"], 2)
        self.assertEqual(features["suspicious_word_count"], 4)
        self.assertEqual(features["tld_length"], 3)
        self.assertEqual(features["has_url_shortener"], 0)

    def test_detects_ip_address_and_at_symbol(self):
        features = extract_features("http://user@example.com@192.168.1.10/login")

        self.assertEqual(features["has_https"], 0)
        self.assertEqual(features["has_ip_address"], 1)
        self.assertEqual(features["has_at_symbol"], 1)
        self.assertEqual(features["subdomain_count"], 0)
        self.assertEqual(features["tld_length"], 0)

    def test_handles_urls_without_scheme(self):
        features = extract_features("bit.ly/secure-update")

        self.assertGreater(features["url_length"], 0)
        self.assertEqual(features["domain_length"], len("bit.ly"))
        self.assertEqual(features["has_url_shortener"], 1)
        self.assertEqual(features["suspicious_word_count"], 2)


if __name__ == "__main__":
    unittest.main()
