import re

class FraudDetector:
    def __init__(self):
        self.suspicious_keywords = ["fake", "fraudulent", "counterfeit", "suspicious", "unverified"]
        self.missing_metadata_fields = ["/Author", "/Producer", "/CreationDate"]

    def detect_fraud(self, text, metadata):
        fraud_score = 0

        if self.contains_suspicious_keywords(text):
            fraud_score += 1

        if self.missing_required_metadata(metadata):
            fraud_score += 1

        if self.contains_suspicious_numbers(text):
            fraud_score += 1

        is_fraudulent = fraud_score >= 2
        return is_fraudulent

    def contains_suspicious_keywords(self, text):
        for keyword in self.suspicious_keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                return True
        return False

    def missing_required_metadata(self, metadata):
        for field in self.missing_metadata_fields:
            if field not in metadata or metadata[field] is None:
                return True
        return False

    def contains_suspicious_numbers(self, text):
        suspicious_patterns = [
            r"\b\d{10,12}\b",
            r"\b\d+\.\d{2}\b"
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                return True
        return False