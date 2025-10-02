import upstox_client
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import urllib.parse
from .config import API_KEY, API_SECRET, REDIRECT_URI
import json
import threading

# Global variable to store the authorization code
auth_code = None

class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        # Ignore favicon requests from browsers
        if self.path.startswith('/favicon.ico'):
            self.send_response(204)
            self.end_headers()
            return

        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get('code', [None])[0]

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        if code:
            auth_code = code
            self.wfile.write(b"<html><body><h1>Authorization successful! You can close this window.</h1></body></html>")
            # Shut down the server in a separate thread to avoid deadlocks
            threading.Thread(target=self.server.shutdown).start()
        else:
            self.wfile.write(b"<html><body><h1>Authorization failed. Please try again.</h1></body></html>")
            threading.Thread(target=self.server.shutdown).start()

def get_access_token():
    """
    Generates and saves an access token for the Upstox API.
    """
    global auth_code
    auth_code = None  # Reset global variable
    api_instance = upstox_client.LoginApi()

    # Manually construct the authorization URL with required scopes for trading
    base_url = "https://api-v2.upstox.com/login/authorization/dialog"
    params = {
        "client_id": API_KEY,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "interactive"  # Requesting interactive trading permissions
    }
    auth_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    print("Opening browser for login...")
    webbrowser.open(auth_url)

    server_address = ('', 8000)
    with HTTPServer(server_address, RedirectHandler) as httpd:
        print("Waiting for authorization... Please log in via the browser.")
        httpd.serve_forever()

    if not auth_code:
        raise Exception("Could not retrieve authorization code. Please try again.")

    access_token_response = api_instance.token(
        client_id=API_KEY,
        client_secret=API_SECRET,
        redirect_uri=REDIRECT_URI,
        grant_type='authorization_code',
        code=auth_code,
        api_version='v2'
    )

    token_data = {'access_token': access_token_response.access_token}
    with open("utils/access_token.json", "w") as f:
        json.dump(token_data, f)

    return access_token_response.access_token

if __name__ == "__main__":
    try:
        token = get_access_token()
        if token:
            print("\nAccess Token obtained and saved to utils/access_token.json")
    except Exception as e:
        print(f"\nAn error occurred: {e}")