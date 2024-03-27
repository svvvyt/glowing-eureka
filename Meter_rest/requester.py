import requests


class Requester:
    def __init__(self, url, meter=None, qr=None, data=None, id=None):
        self.url = url
        self.meter = meter
        self.qr = qr
        self.data = {"meter": self.meter, "qr": self.qr}
        self.id = id

    def make_request(self, method, endpoint, data=None):
        full_url = f"{self.url}/{endpoint}"
        try:
            response = requests.request(method, full_url, data=data)
            response.raise_for_status()
            print(response.text)
        except requests.exceptions.RequestException as err:
            print(f"Error during {method} request to {full_url}: {err}")

    def create_post(self):
        self.make_request("POST", "", data=self.data)

    def get_all_posts(self):
        self.make_request("GET", "")

    def get_post_by_id(self):
        self.make_request("GET", str(self.id))

    def update_post_by_id(self):
        self.make_request("PUT", str(self.id), data=self.data)

    def delete_post_by_id(self):
        self.make_request("DELETE", str(self.id))


requester = Requester(url="http://127.0.0.1:8000/api/data", meter="sample_meter_upd", qr="sample_qr_upd", id=13)

# requester.get_all_posts()
# requester.get_post_by_id()
# requester.create_post()
# requester.update_post_by_id()
# requester.delete_post_by_id()
