"""Carer notifications. The dashboard always shows alerts; an external send happens only when the
carer has configured a channel (CARER_WEBHOOK_URL, e.g. an ntfy.sh topic or a Slack/Teams webhook)."""
import json
import os
import urllib.request


def send_carer(text: str, timeout_s: float = 4.0) -> bool:
    from sip.home import load
    url = os.environ.get("CARER_WEBHOOK_URL") or load().get("carer_webhook_url")
    if not url:
        return False
    try:
        if "ntfy" in url:
            req = urllib.request.Request(url, data=f"Sip care robot: {text}".encode(), headers={"Title": "Sip alert"})
        else:
            req = urllib.request.Request(url, data=json.dumps({"text": f"Sip care robot: {text}"}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=timeout_s).read()
        return True
    except Exception:
        return False
