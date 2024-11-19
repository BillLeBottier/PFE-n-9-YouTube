# Configuration Gunicorn
workers = 1
threads = 4
timeout = 300  # 5 minutes
worker_class = 'gthread'
max_requests = 1000
max_requests_jitter = 50 