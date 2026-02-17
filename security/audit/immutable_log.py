"""
Immutable Logging System
========================

Cryptographically secure logging system that ensures log integrity
and prevents tampering - essential for regulatory compliance.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from ..vault.encryption import AdvancedEncryption

logger = logging.getLogger(__name__)

@dataclass
class LogEntry:
    """Represents an immutable log entry."""
    id: str
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    previous_hash: str
    current_hash: str
    sequence_number: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ImmutableLogger:
    """
    Cryptographically secure immutable logging system.

    Creates a blockchain-like structure where each log entry contains
    the hash of the previous entry, making tampering detectable.
    """

    def __init__(self, log_file: str, encryption: Optional[AdvancedEncryption] = None):
        self.log_file = Path(log_file)
        self.encryption = encryption
        self._lock = threading.Lock()
        self._sequence_number = 0
        self._last_hash = "0" * 64  # Genesis hash
        self._initialized = False

    def initialize(self):
        """Initialize the immutable logger."""
        with self._lock:
            if self._initialized:
                return

            # Create directory if it doesn't exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing entries to get the latest hash and sequence
            if self.log_file.exists():
                self._load_chain_state()

            self._initialized = True
            logger.info(f"Immutable logger initialized with sequence number {self._sequence_number}")

    def log_event(self, event_type: str, data: Dict[str, Any], event_id: Optional[str] = None) -> str:
        """
        Log an immutable event.

        Args:
            event_type: Type of event (e.g., 'TRADE_EXECUTED', 'USER_LOGIN')
            data: Event data
            event_id: Optional custom event ID

        Returns:
            str: The event ID
        """
        if not self._initialized:
            self.initialize()

        with self._lock:
            # Generate unique event ID if not provided
            if not event_id:
                event_id = self._generate_event_id()

            # Create log entry
            entry = LogEntry(
                id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                data=data,
                previous_hash=self._last_hash,
                current_hash="",  # Will be calculated
                sequence_number=self._sequence_number + 1
            )

            # Calculate hash for this entry
            entry.current_hash = self._calculate_hash(entry)

            # Write to file
            self._write_entry(entry)

            # Update state
            self._last_hash = entry.current_hash
            self._sequence_number = entry.sequence_number

            logger.debug(f"Logged immutable event '{event_type}' with ID '{event_id}'")
            return event_id

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of the entire log chain.

        Returns:
            dict: Verification results
        """
        if not self.log_file.exists():
            return {
                'valid': True,
                'total_entries': 0,
                'message': 'No log file exists'
            }

        entries = self._read_all_entries()
        if not entries:
            return {
                'valid': True,
                'total_entries': 0,
                'message': 'Log file is empty'
            }

        # Verify chain integrity
        expected_hash = "0" * 64  # Genesis hash
        invalid_entries = []

        for i, entry in enumerate(entries):
            # Verify previous hash matches
            if entry.previous_hash != expected_hash:
                invalid_entries.append({
                    'entry_id': entry.id,
                    'sequence': entry.sequence_number,
                    'error': f'Previous hash mismatch. Expected: {expected_hash}, Got: {entry.previous_hash}'
                })

            # Verify current hash is correct
            calculated_hash = self._calculate_hash(entry, exclude_current_hash=True)
            if entry.current_hash != calculated_hash:
                invalid_entries.append({
                    'entry_id': entry.id,
                    'sequence': entry.sequence_number,
                    'error': f'Hash mismatch. Expected: {calculated_hash}, Got: {entry.current_hash}'
                })

            # Verify sequence number
            if entry.sequence_number != i + 1:
                invalid_entries.append({
                    'entry_id': entry.id,
                    'sequence': entry.sequence_number,
                    'error': f'Sequence number mismatch. Expected: {i + 1}, Got: {entry.sequence_number}'
                })

            expected_hash = entry.current_hash

        return {
            'valid': len(invalid_entries) == 0,
            'total_entries': len(entries),
            'invalid_entries': invalid_entries,
            'first_entry_time': entries[0].timestamp.isoformat() if entries else None,
            'last_entry_time': entries[-1].timestamp.isoformat() if entries else None
        }

    def get_entries(self, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_types: Optional[List[str]] = None,
                   limit: Optional[int] = None) -> List[LogEntry]:
        """
        Retrieve log entries with optional filtering.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            event_types: Optional list of event types to filter by
            limit: Optional limit on number of entries

        Returns:
            List[LogEntry]: Filtered log entries
        """
        entries = self._read_all_entries()

        # Apply filters
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        if event_types:
            entries = [e for e in entries if e.event_type in event_types]

        # Apply limit
        if limit:
            entries = entries[-limit:]  # Get most recent entries

        return entries

    def search_events(self, search_term: str, field: Optional[str] = None) -> List[LogEntry]:
        """
        Search for events containing a specific term.

        Args:
            search_term: Term to search for
            field: Optional specific field to search in

        Returns:
            List[LogEntry]: Matching log entries
        """
        entries = self._read_all_entries()
        matching_entries = []

        for entry in entries:
            if field:
                # Search in specific field
                if field in entry.data and search_term.lower() in str(entry.data[field]).lower():
                    matching_entries.append(entry)
            else:
                # Search in all data fields
                data_str = json.dumps(entry.data).lower()
                if search_term.lower() in data_str:
                    matching_entries.append(entry)

        return matching_entries

    def export_to_json(self, output_file: str, start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> bool:
        """
        Export log entries to JSON file for external analysis.

        Args:
            output_file: Path to output file
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            bool: Success status
        """
        try:
            entries = self.get_entries(start_time, end_time)
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_entries': len(entries),
                'integrity_check': self.verify_integrity(),
                'entries': [entry.to_dict() for entry in entries]
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported {len(entries)} log entries to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export log entries: {e}")
            return False

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
        return f"evt_{timestamp}_{self._sequence_number + 1}"

    def _calculate_hash(self, entry: LogEntry, exclude_current_hash: bool = False) -> str:
        """Calculate SHA-256 hash for a log entry."""
        # Create a copy of the entry for hashing
        hash_data = {
            'id': entry.id,
            'timestamp': entry.timestamp.isoformat(),
            'event_type': entry.event_type,
            'data': entry.data,
            'previous_hash': entry.previous_hash,
            'sequence_number': entry.sequence_number
        }

        # Include current hash if not excluded (for verification)
        if not exclude_current_hash and entry.current_hash:
            hash_data['current_hash'] = entry.current_hash

        # Convert to JSON and hash
        json_str = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _write_entry(self, entry: LogEntry):
        """Write a log entry to the file."""
        entry_data = entry.to_dict()

        # Encrypt if encryption is enabled
        if self.encryption:
            json_str = json.dumps(entry_data)
            encrypted_data = self.encryption.encrypt_data(json_str)
            entry_data = {
                'encrypted': True,
                'data': encrypted_data
            }

        # Append to file
        with open(self.log_file, 'a') as f:
            json.dump(entry_data, f, separators=(',', ':'))
            f.write('\n')

    def _read_all_entries(self) -> List[LogEntry]:
        """Read all log entries from the file."""
        if not self.log_file.exists():
            return []

        entries = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry_data = json.loads(line)

                    # Decrypt if encrypted
                    if entry_data.get('encrypted'):
                        decrypted_data = self.encryption.decrypt_data(entry_data['data'])
                        entry_data = json.loads(decrypted_data.decode())

                    entry = LogEntry.from_dict(entry_data)
                    entries.append(entry)

        except Exception as e:
            logger.error(f"Failed to read log entries: {e}")
            raise

        return entries

    def _load_chain_state(self):
        """Load the current state of the blockchain from existing entries."""
        entries = self._read_all_entries()
        if entries:
            last_entry = entries[-1]
            self._last_hash = last_entry.current_hash
            self._sequence_number = last_entry.sequence_number
        else:
            self._last_hash = "0" * 64
            self._sequence_number = 0