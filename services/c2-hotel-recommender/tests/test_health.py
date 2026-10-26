from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_home_opens_the_hotel_finder():
    from src.main import APP_DIR
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    built = (APP_DIR / "index.html").exists()
    assert response.headers["location"] == ("/app" if built else "/chat")
    page = client.get("/chat")
    assert page.status_code == 200 and "Hotel finder" in page.text
    if built:
        page = client.get("/app")
        assert page.status_code == 200 and '<div id="root">' in page.text
        js = [w.split('"')[1] for w in page.text.split("src=")[1:] if "/app/assets/" in w][0]
        assert client.get(js).status_code == 200
