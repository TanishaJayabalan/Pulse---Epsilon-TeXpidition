import requests

API = "http://localhost:8003/api"

# Get initial customers
customers = requests.get(f"{API}/customers").json()
c1 = customers[0]["customer_id"]
c2 = customers[1]["customer_id"]

print("Initial:")
r1 = requests.get(f"{API}/customers/{c1}").json()["recommendation"]
r2 = requests.get(f"{API}/customers/{c2}").json()["recommendation"]
print(c1, r1["status"], r1["confidence_zone"])
print(c2, r2["status"], r2["confidence_zone"])

# Simulate negative for c1
requests.post(f"{API}/simulate", json={
    "customer_id": c1,
    "type": "email_ignore",
    "channel": "email",
    "product": "Ferrari Team Cap",
    "sentiment": "negative",
    "offer_category": "sports"
})

# Get dashboard to trigger refresh_stale
requests.get(f"{API}/dashboard?force=true")

print("\nAfter simulating negative for c1:")
r1 = requests.get(f"{API}/customers/{c1}").json()["recommendation"]
r2 = requests.get(f"{API}/customers/{c2}").json()["recommendation"]
print(c1, r1["status"], r1["confidence_zone"])
print(c2, r2["status"], r2["confidence_zone"])

