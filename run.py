import re
import glob
import os
import pickle
import ipaddress
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# ======================================================================
# Configuration
# ======================================================================
LOG_PATHS = [
    '/srv/*/logs/access.log',
    '/srv/*/logs/error.log'
]
MODEL_PATH = 'malicious_detection_model.pkl'
BADURL_PATH = 'dataset1_bad_regex.csv'
STATIC_ASSETS = r'\.(woff2?|ttf|otf|eot|css|js|png|jpe?g|gif|svg|webp)(\?|$)'

# ======================================================================
# URL Feature Extractor
# ======================================================================
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract structural URL features"""
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, urls):
        features = []
        for url in urls:
            path = url.split('?')[0]
            query = url.split('?')[1] if '?' in url else ''
            
            features.append([
                len(url),
                path.count('/'),
                len(query),
                url.count('@'),
                url.count('//'),
                url.count('%'),
                len(query.split('&')),
                1 if '=' in query else 0,
                1 if '../' in url else 0,
                1 if any(x in url.lower() for x in ['wp-', 'admin', 'config']) else 0
            ])
        return np.array(features)

# ======================================================================
# Log Parser
# ======================================================================
class LogParser:
    """Nginx log parser with explicit field labels"""
    def __init__(self):
        self.regex = re.compile(
            r'IP: (\S+) \| '
            r'USER: (\S+) \| '
            r'DOMAIN: (\S+) \| '  # Added DOMAIN field
            r'STATUS: (\d+) \| '
            r'TIME: \[([^\]]+)\] \| '
            r'REQUEST: "(\S+) (\S+) (\S+)" \| '
            r'BYTES: (\d+) \| '
            r'REFERER: "([^"]*)" \| '
            r'UA: "([^"]*)" \| '
            r'XFF: "([^"]*)" \| '
            r'REQ_TIME: "([^"]*)" \| '
            r'UPSTREAM_TIME: "([^"]*)" \| '
            r'CACHE: (\S+);'
        )

    def parse(self, line):
        match = self.regex.match(line)
        if not match:
            return None

        try:
            return {
                'ip': self.validate_ip(match.group(1)),
                'user': match.group(2),
                'domain': match.group(3),  # Added DOMAIN field
                'status': int(match.group(4)),
                'timestamp': match.group(5),
                'method': match.group(6),
                'uri': match.group(7),
                'protocol': match.group(8),
                'bytes_sent': int(match.group(9)),
                'referer': match.group(10),
                'user_agent': match.group(11),
                'x_forwarded_for': match.group(12),
                'request_time': self.parse_float(match.group(13)),
                'upstream_time': self.parse_float(match.group(14)),
                'cache_status': match.group(15),
                'raw': line.strip()
            }
        except (ValueError, ipaddress.AddressValueError) as e:
            print(f"Parse error: {e} in line: {line.strip()}")
            return None

    def validate_ip(self, ip_str):
        try:
            return str(ipaddress.ip_address(ip_str))
        except ValueError:
            return "Invalid IP"

    def parse_float(self, value):
        try:
            return float(value) if value not in ['-', ''] else 0.0
        except ValueError:
            return 0.0

# ======================================================================
# Training System
# ======================================================================
class MaliciousDetectorTrainer:
    def __init__(self):
        self.parser = LogParser()
        self.static_assets_re = re.compile(STATIC_ASSETS, re.IGNORECASE)
        self.badurl_patterns = self.load_badurl_patterns()

    def load_badurl_patterns(self):
        patterns = []
        if not os.path.exists(BADURL_PATH):
            raise FileNotFoundError(f"Missing required file: {BADURL_PATH}")
        
        try:
            with open(BADURL_PATH, 'r') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            patterns.append(re.compile(line, re.IGNORECASE))
                        except re.error as e:
                            print(f"Invalid regex at line {line_number}: {line}\nError: {str(e)}")
        except Exception as e:
            print(f"Error loading patterns: {str(e)}")
            raise
        
        if not patterns:
            raise ValueError("No valid patterns found in badurl.csv")
        return patterns

    def is_malicious(self, log_entry):
        if any(pat.search(log_entry['uri']) for pat in self.badurl_patterns):
            return 1
        return 0

    def stream_logs(self, file_patterns, sample_size=100000):
        for pattern in file_patterns:
            for path in glob.glob(pattern):
                try:
                    with open(path, 'r') as f:
                        for line in f:
                            parsed = self.parser.parse(line)
                            if parsed and not self.static_assets_re.search(parsed['uri']):
                                yield parsed
                                sample_size -= 1
                                if sample_size <= 0: 
                                    return
                except Exception as e:
                    print(f"Error processing {path}: {e}")

    def build_feature_pipeline(self):
        return ColumnTransformer([
            ('text', Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(1, 3),
                    max_features=5000,
                    sublinear_tf=True
                ))
            ]), 'uri'),
            ('structured', Pipeline([
                ('url_features', URLFeatureExtractor()),
                ('scaler', StandardScaler())
            ]), 'uri'),
            ('method', OneHotEncoder(), ['method']),
            ('cache', OneHotEncoder(), ['cache_status']),
            ('protocol', OneHotEncoder(), ['protocol']),
            ('status', StandardScaler(), ['status']),
            ('bytes', StandardScaler(), ['bytes_sent'])
        ])

    def train(self, save_path=MODEL_PATH):
        print("Training model...")
        logs = list(self.stream_logs(LOG_PATHS))
        if not logs:
            raise ValueError("No logs processed! Check configuration and permissions")

        df = pd.DataFrame(logs)
        df['label'] = [self.is_malicious(log) for log in logs]
        
        # Check class distribution
        class_counts = df['label'].value_counts()
        print(f"\nClass distribution:\n{class_counts}")
        
        if len(class_counts) < 2:
            msg = f"""\nTraining failed: Need both malicious and benign samples
            Malicious samples: {class_counts.get(1, 0)}
            Benign samples: {class_counts.get(0, 0)}
            """
            raise ValueError(msg)

        try:
            feature_pipeline = self.build_feature_pipeline()
            
            # Use imblearn's Pipeline instead of sklearn's
            from imblearn.pipeline import Pipeline
            
            # Build pipeline with proper SMOTE integration
            full_pipeline = Pipeline([
                ('features', feature_pipeline),
                ('smote', SMOTE(random_state=42)),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced',
                    max_depth=10,
                    random_state=42
                ))
            ])

            # Train the model
            full_pipeline.fit(df, df['label'])

            # Save model
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'pipeline': full_pipeline,
                    'parser': self.parser,
                    'train_date': datetime.now().isoformat(),
                    'class_distribution': class_counts.to_dict()
                }, f)

            print(f"\nModel saved to {save_path}")
            print(f"Malicious detection rate: {class_counts[1]/len(df)*100:.1f}%")

        except Exception as e:
            print(f"\nTraining failed: {str(e)}")
            raise

# ======================================================================
# Detection System
# ======================================================================
class MaliciousDetector:
    def __init__(self):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        self.pipeline = data['pipeline']
        self.parser = data['parser']
        self.static_assets_re = re.compile(STATIC_ASSETS, re.IGNORECASE)
    
    def process_logs(self, threshold=0.7, batch_size=1000):
        malicious_ips = {}
        
        for pattern in LOG_PATHS:
            for path in glob.glob(pattern):
                try:
                    with open(path, 'r') as f:
                        batch = []
                        for line in f:
                            parsed = self.parser.parse(line)
                            if parsed and not self.static_assets_re.search(parsed['uri']):
                                batch.append(parsed)
                                if len(batch) >= batch_size:
                                    self.process_batch(batch, malicious_ips)
                                    batch = []
                        if batch:
                            self.process_batch(batch, malicious_ips)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
        
        return malicious_ips
    
    def process_batch(self, batch, result):
        df = pd.DataFrame(batch)
        try:
            probs = self.pipeline.predict_proba(df)[:, 1]
            for entry, prob in zip(batch, probs):
                if prob > 0.7:
                    ip = entry['ip']
                    result.setdefault(ip, []).append({
                        'timestamp': entry['timestamp'],
                        'method': entry['method'],
                        'uri': entry['uri'],
                        'domain': entry['domain'],  # Added domain field
                        'status': entry['status'],
                        'cache_status': entry['cache_status'],
                        'user_agent': entry['user_agent'],
                        'probability': prob,
                        'raw': entry['raw'],
                        'modsec_rule': self.generate_modsec_rule(entry)  # Generate ModSecurity rule
                    })
        except Exception as e:
            print(f"Error processing batch: {str(e)}")

    def generate_modsec_rule(self, entry):
        """Generate a ModSecurity rule to block the malicious request."""
        rule_template = (
            'SecRule REQUEST_URI "@contains {uri}" "id:{rule_id},phase:1,deny,log,msg:\'Detected malicious request to {uri}\'"'
        )
        rule_id = 100000 + hash(entry['uri']) % 900000  # Generate a unique rule ID
        return rule_template.format(uri=entry['uri'], rule_id=rule_id)

# ======================================================================
# Main Execution
# ======================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Web Log Analyzer')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--detect', action='store_true', help='Run detection')
    args = parser.parse_args()

    if args.train:
        print("Starting training process...")
        trainer = MaliciousDetectorTrainer()
        trainer.train()
        print("Training completed successfully")
    elif args.detect:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model missing. Run with --train first")
            
        print("Starting detection...")
        detector = MaliciousDetector()
        findings = detector.process_logs()
        
        if findings:
            print("\n=== Malicious Activity Detected ===")
            for ip, requests in findings.items():
                print(f"IP: {ip} - {len(requests)} suspicious requests")
                for req in requests[:3]:
                    print(f"  [{req['timestamp']}] {req['method']} {req['uri']}")
                    print(f"  Domain: {req['domain']}")
                    print(f"  Status: {req['status']} | Cache: {req['cache_status']}")
                    print(f"  User Agent: {req['user_agent'][:50]}")
                    print(f"  Probability: {req['probability']:.2f}")
                    print(f"  Suggested ModSecurity Rule: {req['modsec_rule']}\n")
        else:
            print("\nNo malicious activity detected")
    else:
        parser.print_help()