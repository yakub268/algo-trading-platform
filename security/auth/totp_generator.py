"""
Time-based One-Time Password (TOTP) Generator
==============================================

RFC 6238 compliant TOTP implementation for multi-factor authentication.
"""

import hmac
import hashlib
import struct
import time
import secrets
import base64
import qrcode
from io import BytesIO
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TOTPGenerator:
    """
    Time-based One-Time Password generator implementing RFC 6238.

    Generates 6-digit codes that change every 30 seconds, compatible
    with authenticator apps like Google Authenticator, Authy, etc.
    """

    def __init__(self, issuer: str = "TradingBot", digits: int = 6, period: int = 30):
        """
        Initialize TOTP generator.

        Args:
            issuer: The issuer name shown in authenticator apps
            digits: Number of digits in the TOTP code (6 or 8)
            period: Time period in seconds for code validity (usually 30)
        """
        self.issuer = issuer
        self.digits = digits
        self.period = period

        if digits not in [6, 8]:
            raise ValueError("TOTP digits must be 6 or 8")

    def generate_secret(self, length: int = 20) -> str:
        """
        Generate a cryptographically secure random secret for TOTP.

        Args:
            length: Length of the secret in bytes (default 20 for 160-bit security)

        Returns:
            str: Base32 encoded secret
        """
        # Generate random bytes
        random_bytes = secrets.token_bytes(length)

        # Encode as base32 for compatibility with authenticator apps
        secret = base64.b32encode(random_bytes).decode('utf-8')

        # Remove padding characters for cleaner display
        return secret.rstrip('=')

    def generate_code(self, secret: str, timestamp: Optional[int] = None) -> str:
        """
        Generate a TOTP code for the given secret and timestamp.

        Args:
            secret: Base32 encoded secret
            timestamp: Unix timestamp (uses current time if None)

        Returns:
            str: TOTP code (6 or 8 digits)
        """
        if timestamp is None:
            timestamp = int(time.time())

        # Calculate time step
        time_step = timestamp // self.period

        # Convert secret from base32
        try:
            # Add padding if necessary
            padded_secret = secret + '=' * (8 - len(secret) % 8)
            key = base64.b32decode(padded_secret.upper())
        except Exception as e:
            logger.error(f"Invalid TOTP secret: {e}")
            raise ValueError("Invalid TOTP secret")

        # Convert time step to 8-byte big-endian
        time_bytes = struct.pack(">Q", time_step)

        # Generate HMAC-SHA1 hash
        hmac_hash = hmac.new(key, time_bytes, hashlib.sha1).digest()

        # Dynamic truncation
        offset = hmac_hash[-1] & 0x0f
        truncated = hmac_hash[offset:offset + 4]

        # Convert to integer and remove most significant bit
        code_int = struct.unpack(">I", truncated)[0] & 0x7fffffff

        # Generate the final code
        code = code_int % (10 ** self.digits)

        # Pad with leading zeros
        return f"{code:0{self.digits}d}"

    def verify_code(self, secret: str, code: str, window: int = 1) -> bool:
        """
        Verify a TOTP code against the secret with time window tolerance.

        Args:
            secret: Base32 encoded secret
            code: TOTP code to verify
            window: Number of time periods to check (±window)

        Returns:
            bool: True if code is valid
        """
        if not code.isdigit() or len(code) != self.digits:
            return False

        current_time = int(time.time())

        # Check current time and surrounding windows
        for i in range(-window, window + 1):
            timestamp = current_time + (i * self.period)
            expected_code = self.generate_code(secret, timestamp)

            if self._constant_time_compare(code, expected_code):
                return True

        return False

    def generate_qr_code(self, secret: str, account_name: str) -> bytes:
        """
        Generate QR code for easy setup in authenticator apps.

        Args:
            secret: Base32 encoded secret
            account_name: Account identifier (e.g., username or email)

        Returns:
            bytes: PNG image data of QR code
        """
        # Create the otpauth URL
        url = self._create_otpauth_url(secret, account_name)

        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")

        # Convert to bytes
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()

    def get_provisioning_uri(self, secret: str, account_name: str) -> str:
        """
        Get the provisioning URI for manual entry in authenticator apps.

        Args:
            secret: Base32 encoded secret
            account_name: Account identifier

        Returns:
            str: otpauth:// URI
        """
        return self._create_otpauth_url(secret, account_name)

    def get_backup_codes(self, count: int = 10) -> list[str]:
        """
        Generate backup codes for account recovery.

        Args:
            count: Number of backup codes to generate

        Returns:
            list: List of backup codes
        """
        backup_codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric code
            code = secrets.token_hex(4).upper()
            # Format as XXXX-XXXX for readability
            formatted_code = f"{code[:4]}-{code[4:]}"
            backup_codes.append(formatted_code)

        return backup_codes

    def _create_otpauth_url(self, secret: str, account_name: str) -> str:
        """Create the otpauth URL for authenticator apps."""
        from urllib.parse import quote

        # Encode parameters
        issuer_encoded = quote(self.issuer)
        account_encoded = quote(account_name)
        secret_encoded = quote(secret)

        # Create URL
        url = (
            f"otpauth://totp/{issuer_encoded}:{account_encoded}"
            f"?secret={secret_encoded}"
            f"&issuer={issuer_encoded}"
            f"&digits={self.digits}"
            f"&period={self.period}"
            f"&algorithm=SHA1"
        )

        return url

    def _constant_time_compare(self, a: str, b: str) -> bool:
        """
        Compare two strings in constant time to prevent timing attacks.

        Args:
            a: First string
            b: Second string

        Returns:
            bool: True if strings are equal
        """
        if len(a) != len(b):
            return False

        result = 0
        for x, y in zip(a, b):
            result |= ord(x) ^ ord(y)

        return result == 0

    def get_current_counter(self) -> int:
        """
        Get the current time counter for TOTP calculation.

        Returns:
            int: Current time counter
        """
        return int(time.time()) // self.period

    def get_remaining_time(self) -> int:
        """
        Get remaining seconds until the current TOTP code expires.

        Returns:
            int: Remaining seconds
        """
        current_time = int(time.time())
        return self.period - (current_time % self.period)

# Convenience function for quick TOTP operations
def generate_totp_secret() -> str:
    """Generate a new TOTP secret."""
    generator = TOTPGenerator()
    return generator.generate_secret()

def verify_totp_code(secret: str, code: str, issuer: str = "TradingBot") -> bool:
    """Verify a TOTP code."""
    generator = TOTPGenerator(issuer=issuer)
    return generator.verify_code(secret, code)