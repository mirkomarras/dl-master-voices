import numpy as np
import requests
import io
import base64

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# DEFAULT_TIMEOUT = 60  # seconds

# https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
# class TimeoutHTTPAdapter(HTTPAdapter):
#     def __init__(self, *args, **kwargs):
#         self.timeout = DEFAULT_TIMEOUT
#         if "timeout" in kwargs:
#             self.timeout = kwargs["timeout"]
#             del kwargs["timeout"]
#         super().__init__(*args, **kwargs)

#     def send(self, request, **kwargs):
#         timeout = kwargs.get("timeout")
#         if timeout is None:
#             kwargs["timeout"] = self.timeout
#         return super().send(request, **kwargs)


retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)
http = requests.Session()
http.mount("http://", HTTPAdapter(max_retries=retry_strategy))

# retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
# 
# http.mount("http://", TimeoutHTTPAdapter(max_retries=retries))


def clone_voice(embed, text):
    embed = embed.astype(np.float32)
    r = http.post('http://127.0.0.1:8080/clone_voice', data = {'embed': base64.b64encode(embed.tobytes()), 'text': text})
    if r.status_code == 200:
        return np.frombuffer(r.content, np.float32)
    else:
        raise ValueError(f'remote API Error: {r.status_code} {r.reason}')

def init_embedding(sample):
    sample = sample.astype(np.float32)
    r = http.post('http://127.0.0.1:8080/speaker_embedding', data={'embed': base64.b64encode(sample.tobytes())})
    if r.status_code == 200:
        return np.frombuffer(r.content, np.float32)
    else:
        raise ValueError(f'remote API Error: {r.status_code} {r.reason}')