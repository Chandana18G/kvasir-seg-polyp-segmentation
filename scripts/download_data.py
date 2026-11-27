"""Download Kvasir-SEG and the authors' split files into data/.

The images are NOT in this repository: the licence allows research and education
only, and the dataset must be cited (Jha et al., MMM 2020). This script gets them
from the official Simula page instead.

Run from the project folder:  python scripts/download_data.py
"""
import os
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile

ROOT = os.path.join(os.path.dirname(__file__), "..")       # the project folder
DATA = os.path.join(ROOT, "data")                          # data/ is in .gitignore: never on GitHub

ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"  # 46 MB
# The authors publish their split (train.txt 880 names, val.txt 120 names) on GitHub.
SPLIT_URL = "https://raw.githubusercontent.com/DebeshJha/Kvasir-SEG/master/Data-split/"

# The Simula server sends an incomplete certificate chain: it leaves out the
# certificate of the company that signed its own ("Sectigo ... DV R36"). Windows
# fetches that one by itself; Linux and Colab do not, and the download fails with
# CERTIFICATE_VERIFY_FAILED. This is the address the server's certificate names for it.
MISSING_CERT_URL = "http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt"


def download(url, path, context=None):
    """Save url to path. Writes to a .part file first, so a broken download
    never leaves a half file behind."""
    with urllib.request.urlopen(url, context=context) as response:
        with open(path + ".part", "wb") as f:
            shutil.copyfileobj(response, f)
    os.replace(path + ".part", path)


def context_with_missing_cert():
    """Normal certificate checking, plus the one certificate the server forgets.

    Checking stays ON: the extra certificate must itself be signed by a root that
    the computer already trusts (USERTrust), otherwise the connection still fails.
    """
    with urllib.request.urlopen(MISSING_CERT_URL) as response:
        der = response.read()
    context = ssl.create_default_context()                 # the normal, safe settings
    context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))   # add the missing link of the chain
    return context


# ---- the script starts here ----
os.makedirs(DATA, exist_ok=True)               # make data/ if it does not exist yet

zip_path = os.path.join(DATA, "kvasir-seg.zip")
if not os.path.exists(zip_path):               # download only once
    print("Downloading", ZIP_URL)
    try:
        download(ZIP_URL, zip_path)            # works on Windows directly
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise                                  # another error: stop and show it
        print("Certificate problem on the Simula server; adding its missing certificate ...")
        download(ZIP_URL, zip_path, context=context_with_missing_cert())   # Colab/Linux: retry with the fix

if not os.path.exists(os.path.join(DATA, "Kvasir-SEG", "images")):
    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA)

# The authors' split: train.txt (880 names) and val.txt (120 names, our test set).
for name in ["train.txt", "val.txt"]:
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        download(SPLIT_URL + name, path)

# Check: there must be 1,000 images and 1,000 masks.
n_images = len(os.listdir(os.path.join(DATA, "Kvasir-SEG", "images")))
n_masks = len(os.listdir(os.path.join(DATA, "Kvasir-SEG", "masks")))
print(f"Done: {n_images} images, {n_masks} masks in data/Kvasir-SEG/")
