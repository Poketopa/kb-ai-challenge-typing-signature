from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import numpy as np

KEY_SIZE = 32  # 256 bit
NONCE_SIZE = 12  # 96 bit (GCM 표준)


def generate_key():
    return AESGCM.generate_key(bit_length=256)

def save_key(key: bytes, path: str):
    with open(path, 'wb') as f:
        f.write(key)

def load_key(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()

def get_or_create_key(path: str = "secret.key") -> bytes:
    if not os.path.exists(path):
        key = generate_key()
        save_key(key, path)
        return key
    return load_key(path)

def encrypt_embedding(embedding: np.ndarray, key: bytes) -> bytes:
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_SIZE)
    data = embedding.astype(np.float32).tobytes()
    ct = aesgcm.encrypt(nonce, data, None)
    return nonce + ct

def decrypt_embedding(blob: bytes, key: bytes) -> np.ndarray:
    aesgcm = AESGCM(key)
    nonce = blob[:NONCE_SIZE]
    ct = blob[NONCE_SIZE:]
    data = aesgcm.decrypt(nonce, ct, None)
    arr = np.frombuffer(data, dtype=np.float32)
    return arr
