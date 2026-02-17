"""
Access Control Module
====================

Role-based access control (RBAC) system with fine-grained permissions,
session management, and security policy enforcement.
"""

from .rbac import RoleBasedAccess
from .permissions import PermissionManager
from .session_manager import SessionManager

__all__ = ['RoleBasedAccess', 'PermissionManager', 'SessionManager']