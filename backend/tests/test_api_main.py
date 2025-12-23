def test_root_and_health(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["docs"] == "/docs"

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}
