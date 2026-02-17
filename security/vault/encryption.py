"""
Advanced Encryption Module
==========================

Enterprise-grade encryption for sensitive data including API keys, credentials,
and personally identifiable information (PII).
"""

import os
import base64
import hashlib
import secrets
from typing import Union, Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class AdvancedEncryption:
    """
    Advanced encryption class providing both symmetric and asymmetric encryption
    capabilities with key derivation and secure random generation.
    """

    def __init__(self, key_file: Optional[str] = None):
        self.key_file = key_file or "security/keys/master.key"
        self.backend = default_backend()
        self._master_key = None
        self._private_key = None
        self._public_key = None

    def generate_master_key(self, save_to_file: bool = True) -> bytes:
        """
        Generate a new master key for symmetric encryption.

        Returns:
            bytes: The generated master key
        """
        master_key = secrets.token_bytes(32)  # 256-bit key

        if save_to_file:
            self._save_master_key(master_key)

        self._master_key = master_key
        logger.info("Master key generated successfully")
        return master_key

    def load_master_key(self) -> bytes:
        """
        Load the master key from file.

        Returns:
            bytes: The loaded master key
        """
        if not Path(self.key_file).exists():
            logger.warning(f"Master key file {self.key_file} not found, generating new key")
            return self.generate_master_key()

        try:
            with open(self.key_file, 'rb') as f:
                self._master_key = f.read()
            logger.info("Master key loaded successfully")
            return self._master_key
        except Exception as e:
            logger.error(f"Failed to load master key: {e}")
            raise

    def _save_master_key(self, key: bytes):
        """Save the master key to file with proper permissions."""
        Path(self.key_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.key_file, 'wb') as f:
            f.write(key)

        # Set restrictive file permissions (owner only)
        os.chmod(self.key_file, 0o600)

    def derive_key(self, password: str, salt: bytes = None) -> tuple[bytes, bytes]:
        """
        Derive a key from a password using PBKDF2.

        Args:
            password: The password to derive from
            salt: Optional salt (will be generated if not provided)

        Returns:
            tuple: (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )

        key = kdf.derive(password.encode())
        return key, salt

    def encrypt_data(self, data: Union[str, bytes], key: Optional[bytes] = None) -> Dict[str, str]:
        """
        Encrypt data using AES-256-GCM.

        Args:
            data: Data to encrypt (string or bytes)
            key: Optional encryption key (uses master key if not provided)

        Returns:
            dict: Encrypted data with metadata
        """
        if isinstance(data, str):
            data = data.encode()

        if key is None:
            key = self._master_key or self.load_master_key()

        # Generate a random nonce for GCM mode
        nonce = secrets.token_bytes(12)

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=self.backend
        )

        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return encrypted data with metadata
        return {
            'ciphertext': base64.b64encode(ciphertext).decode(),
            'nonce': base64.b64encode(nonce).decode(),
            'tag': base64.b64encode(encryptor.tag).decode(),
            'algorithm': 'AES-256-GCM'
        }

    def decrypt_data(self, encrypted_data: Dict[str, str], key: Optional[bytes] = None) -> bytes:
        """
        Decrypt data encrypted with encrypt_data.

        Args:
            encrypted_data: Dictionary containing encrypted data and metadata
            key: Optional decryption key (uses master key if not provided)

        Returns:
            bytes: Decrypted data
        """
        if key is None:
            key = self._master_key or self.load_master_key()

        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        tag = base64.b64decode(encrypted_data['tag'])

        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )

        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

    def generate_rsa_keys(self, key_size: int = 2048) -> tuple[bytes, bytes]:
        """
        Generate RSA key pair for asymmetric encryption.

        Args:
            key_size: Size of the RSA keys in bits

        Returns:
            tuple: (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        self._private_key = private_key
        self._public_key = public_key

        return private_pem, public_pem

    def load_rsa_keys(self, private_key_path: str, public_key_path: str):
        """Load RSA keys from PEM files."""
        with open(private_key_path, 'rb') as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=self.backend
            )

        with open(public_key_path, 'rb') as f:
            self._public_key = serialization.load_pem_public_key(
                f.read(),
                backend=self.backend
            )

    def encrypt_with_rsa(self, data: bytes) -> bytes:
        """
        Encrypt data with RSA public key.

        Args:
            data: Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        if not self._public_key:
            raise ValueError("No RSA public key loaded")

        return self._public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def decrypt_with_rsa(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data with RSA private key.

        Args:
            encrypted_data: Encrypted data

        Returns:
            bytes: Decrypted data
        """
        if not self._private_key:
            raise ValueError("No RSA private key loaded")

        return self._private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def hash_data(self, data: Union[str, bytes], algorithm: str = 'SHA256') -> str:
        """
        Create a cryptographic hash of data.

        Args:
            data: Data to hash
            algorithm: Hash algorithm to use

        Returns:
            str: Hex digest of the hash
        """
        if isinstance(data, str):
            data = data.encode()

        if algorithm.upper() == 'SHA256':
            hash_obj = hashlib.sha256(data)
        elif algorithm.upper() == 'SHA512':
            hash_obj = hashlib.sha512(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        return hash_obj.hexdigest()

    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.

        Args:
            length: Length of the token in bytes

        Returns:
            str: Base64-encoded secure token
        """
        token = secrets.token_bytes(length)
        return base64.urlsafe_b64encode(token).decode().rstrip('=')

    def constant_time_compare(self, a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks.

        Args:
            a: First string
            b: Second string

        Returns:
            bool: True if strings are equal
        """
        return secrets.compare_digest(a, b)

class FieldEncryption:
    """
    Field-level encryption for database records and sensitive data structures.
    """

    def __init__(self, encryption: AdvancedEncryption):
        self.encryption = encryption
        self.sensitive_fields = {
            'api_key', 'secret_key', 'private_key', 'password',
            'email', 'phone', 'ssn', 'account_number', 'routing_number'
        }

    def encrypt_record(self, record: Dict[str, Any], fields_to_encrypt: Optional[list] = None) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a record.

        Args:
            record: Dictionary containing record data
            fields_to_encrypt: List of field names to encrypt (uses defaults if not provided)

        Returns:
            dict: Record with encrypted sensitive fields
        """
        encrypted_record = record.copy()
        encrypt_fields = fields_to_encrypt or self.sensitive_fields

        for field_name, value in record.items():
            if field_name in encrypt_fields and value is not None:
                encrypted_data = self.encryption.encrypt_data(str(value))
                encrypted_record[field_name] = {
                    'encrypted': True,
                    'data': encrypted_data
                }

        return encrypted_record

    def decrypt_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt encrypted fields in a record.

        Args:
            record: Dictionary containing record data with encrypted fields

        Returns:
            dict: Record with decrypted fields
        """
        decrypted_record = record.copy()

        for field_name, value in record.items():
            if isinstance(value, dict) and value.get('encrypted'):
                decrypted_data = self.encryption.decrypt_data(value['data'])
                decrypted_record[field_name] = decrypted_data.decode()

        return decrypted_record

    def is_field_encrypted(self, field_value: Any) -> bool:
        """Check if a field value is encrypted."""
        return isinstance(field_value, dict) and field_value.get('encrypted', False)